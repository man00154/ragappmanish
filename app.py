import streamlit as st
import os
import requests
import tempfile
import base64
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config ---
PDF_DOWNLOAD_DIR = "downloaded_pdfs"
CHROMA_DB_DIR = "chroma_db"
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Predefined PDFs per company, with all your provided links included:
PREDEFINED_PDF_LINKS = {
    "Dell": [
        "https://i.dell.com/sites/csdocuments/Product_Docs/en/Dell-EMC-PowerEdge-Rack-Servers-Quick-Reference-Guide.pdf",
        "https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-r660xs-technical-guide.pdf",
        "https://i.dell.com/sites/csdocuments/shared-content_data-sheets_documents/en/aa/poweredge_r740_r740xd_technical_guide.pdf",
        "https://dl.dell.com/topicspdf/openmanage-server-administrator-v95_users-guide_en-us.pdf",
        "https://dl.dell.com/manuals/common/dellemc-server-config-profile-refguide.pdf",
    ],
    "IBM": [
        "https://www.redbooks.ibm.com/redbooks/pdfs/sg248513.pdf",
        "https://www.ibm.com/docs/SSLVMB_28.0.0/pdf/IBM_SPSS_Statistics_Server_Administrator_Guide.pdf",
        "https://public.dhe.ibm.com/software/webserver/appserv/library/v60/ihs_60.pdf",
        "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-files",
    ],
    "Cisco": [
        "https://www.cisco.com/c/dam/global/shared/assets/pdf/cisco_enterprise_campus_infrastructure_design_guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_Wireless_LAN_Design_Guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_IP_Addressing_Best_Practices.pdf",
        "https://www.cisco.com/c/en/us/td/docs/net_mgmt/network_registrar/7-2/user/guide/cnr72book.pdf",
    ],
    "Juniper": [
        "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/junos-overview.pdf",
        "https://archive.org/download/junos-srxsme/JunOS%20SRX%20Documentation%20Set/network-management.pdf",
        "https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3779.pdf",
    ],
    "Fortinet (FortiGate)": [
        "https://fortinetweb.s3.amazonaws.com/docs.fortinet.com/v2/attachments/b94274f8-1a11-11e9-9685-f8bc1258b856/FortiOS-5.6-Firewall.pdf",
        "https://docs.fortinet.com/document/fortiweb/6.0.7/administration-guide-pdf",
        "https://www.andovercg.com/datasheets/fortigate-fortinet-200.pdf",
        "https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf",
    ],
    "EUC": [
        # These are websites, so will be skipped gracefully in ingestion
        "https://www.dell.com/en-us/lp/dt/end-user-computing",
        "https://www.nutanix.com/solutions/end-user-computing",
        "https://eucscore.com/docs/tools.html",
        "https://apparity.com/euc-resources/spreadsheet-euc-documents/",
    ],
}


# --- Helper Functions ---

def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

def load_and_split_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        st.success(f"Processed {len(texts)} chunks from {os.path.basename(file_path)}")
        return texts
    except Exception as e:
        st.error(f"Error processing PDF {os.path.basename(file_path)}: {e}")
        return []

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0
    )

def initialize_vector_store(documents, embeddings):
    if documents:
        if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
            st.session_state.vector_store.add_documents(documents)
            st.info("Added to existing vector store.")
        else:
            st.session_state.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
            st.info("Created new vector store.")
        st.session_state.vector_store.persist()
        st.success("Vector store updated!")
    else:
        st.warning("No documents to add.")

def get_rag_chain(vector_store, llm):
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not display PDF: {e}")

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="RAG App with Groq")
st.title("📄 MANISH SINGH- RAG Application with Document Chat (Groq)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_display_path" not in st.session_state:
    st.session_state.pdf_display_path = None

# Initialize models
try:
    embeddings = get_embeddings()
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to initialize models. Error: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Upload & Ingest")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process Uploaded PDFs"):
        all_new_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            new_docs = load_and_split_pdf(temp_file_path)
            all_new_docs.extend(new_docs)
            if not st.session_state.pdf_display_path:
                st.session_state.pdf_display_path = temp_file_path
        if all_new_docs:
            initialize_vector_store(all_new_docs, embeddings)

    st.subheader("Predefined PDF Ingestion")
    selected_company = st.selectbox("Select a company", [""] + list(PREDEFINED_PDF_LINKS.keys()))
    if st.button("Ingest Predefined PDFs"):
        if selected_company:
            all_docs = []
            for url in PREDEFINED_PDF_LINKS[selected_company]:
                if url.lower().endswith(".pdf"):
                    file_name = os.path.basename(urlparse(url).path)
                    output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)
                    if download_pdf(url, output_path):
                        docs = load_and_split_pdf(output_path)
                        all_docs.extend(docs)
                        if not st.session_state.pdf_display_path:
                            st.session_state.pdf_display_path = output_path
                else:
                    st.warning(f"Skipping non-PDF: {url}")
            if all_docs:
                initialize_vector_store(all_docs, embeddings)

# Layout
col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.subheader("📑 PDF Viewer")
    if st.session_state.pdf_display_path:
        display_pdf(st.session_state.pdf_display_path)
    else:
        st.info("Upload or ingest a PDF.")

with col2:
    st.subheader("💬 Chat with Documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            if st.session_state.vector_store is None:
                reply = "Please upload or ingest documents first."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                with st.spinner("Thinking..."):
                    qa_chain = get_rag_chain(st.session_state.vector_store, llm)
                    if qa_chain:
                        try:
                            response = qa_chain({"question": prompt})
                            ai_response = response["answer"]
                            st.markdown(ai_response)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        except Exception as e:
                            st.error(f"Error: {e}")
