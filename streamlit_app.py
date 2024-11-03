import sys

if sys.platform.startswith('linux'):
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import PyPDF2

APP_NAME = "📄 StatementWise 📄"
OPEN_AI_KEY = st.secrets.API_KEY

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    
    if uploaded_file is not None:
        pdf_file = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page in pdf_file.pages:
            text += page.extract_text()
        documents = [text]
        # documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.invoke(query_text)

# Page title
st.set_page_config(page_title=APP_NAME)
# st.title(APP_NAME)
st.markdown(f"<h1 style='text-align: center'>{APP_NAME}</h1>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
# Query text

# Form input and query
result = []
with st.form('myform', clear_on_submit=True): 
    col1, col2 = st.columns([7, 1])
    with col1:
        query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)
    with col2:
        st.markdown("<p style='margin-top: 27.5px;'></p>", unsafe_allow_html=True)
        submitted = st.form_submit_button('Submit', disabled=not (uploaded_file))
    if submitted and OPEN_AI_KEY.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, OPEN_AI_KEY, query_text)
            response_result = response["result"]
            result.append(response_result)
if len(result):
    st.info(response_result)
