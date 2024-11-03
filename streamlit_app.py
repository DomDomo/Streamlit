import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import PyPDF2

def generate_response(uploaded_file, openai_api_key, query_text):
    try:
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
    except Exception as e:
        st.error(f"An error occurred while processing the document: {e}")
        return None

# Page title
st.set_page_config(page_title='Statement Wise')
st.title('Statement Wise')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form input and actions
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.secrets.API_KEY
    
    # Columns for buttons in the same row
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Submit button
    with col1:
        submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    
    # Summarise button
    with col2:
        summarize_clicked = st.form_submit_button('Summarise')
    
    # Recommend button
    with col3:
        recommend_clicked = st.form_submit_button('Recommend')
    
    # Perform actions based on the button clicked
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            if response and "result" in response:
                response_result = response["result"]
                result.append(response_result)
            else:
                st.error("No response received for the query.")

    elif summarize_clicked and openai_api_key.startswith('sk-'):
        with st.spinner("Summarizing the document..."):
            summary_prompt = "Please provide a short summary of the following document."
            summary_response = generate_response(uploaded_file, openai_api_key, summary_prompt)
            if summary_response and "result" in summary_response:
                summary_result = summary_response["result"]
                st.success(summary_result)
            else:
                st.error("No summary could be generated.")

    elif recommend_clicked and openai_api_key.startswith('sk-'):
        with st.spinner("Generating recommendations..."):
            recommendation_prompt = "Based on the content of this document, please provide recommendations."
            recommendation_response = generate_response(uploaded_file, openai_api_key, recommendation_prompt)
            if recommendation_response and "result" in recommendation_response:
                recommendation_result = recommendation_response["result"]
                st.success(recommendation_result)
            else:
                st.error("No recommendations could be generated.")

# Display the answer to the query
if len(result):
    st.info(result[0])
