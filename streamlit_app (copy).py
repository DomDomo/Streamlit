import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import PyPDF2

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        pdf_file = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page in pdf_file.pages:
            text += page.extract_text()
        documents = [text]
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
st.set_page_config(page_title='Statement Wise')
st.title('Statement Wise')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    openai_api_key = st.secrets.API_KEY
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            response_result = response["result"]
            result.append(response_result)
            del openai_api_key

# Display the answer to the query
if len(result):
    st.info(response_result)

# Summarize button
if uploaded_file is not None:
    if st.button("Summarise"):
        with st.spinner("Summarizing the document..."):
            openai_api_key = st.secrets.API_KEY
            # Prompt for summarizing the document
            summary_prompt = "Please provide a short summary of the following document."
            summary_response = generate_response(uploaded_file, openai_api_key, summary_prompt)
            summary_result = summary_response["result"]
            st.success(summary_result)
