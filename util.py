from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()


# Function to get the API key
def get_api_key():
    # Try to get the API key from st.secrets first
    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        return groq_api_key
    except Exception as e:
        print(e)

def get_inference_api_key():
    try:
        inference_api_key = os.getenv("INFERENCE_API_KEY", "")

        return inference_api_key
    except Exception as e:
        print(e)


# Function for API configuration at sidebar
def sidebar_api_key_configuration():
    groq_api_key = get_api_key()
    if groq_api_key == '':
        st.sidebar.warning('Enter the API Key(s) 🗝️')
        st.session_state.prompt_activation = False
    elif (groq_api_key.startswith('gsk_') and (len(groq_api_key) == 56)):
        st.sidebar.success('Lets Proceed!', icon='️👉')
        st.session_state.prompt_activation = True
    else:
        st.sidebar.warning('Please enter the correct API Key 🗝️!', icon='⚠️')
        st.session_state.prompt_activation = False
    return groq_api_key


def sidebar_groq_model_selection():
    st.sidebar.subheader("Model Selection")
    model = st.sidebar.selectbox('Select the Model', ('Llama3-8b-8192', 'Llama3-70b-8192', 'Mixtral-8x7b-32768',
                                                      'Gemma-7b-it'), label_visibility="collapsed")
    return model


# Read PDF data
def read_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")
    inference_api_key = get_inference_api_key()

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings


# Create vectorstore
def create_vectorstore(pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks
    embeddings = get_embedding_function()  # Get the embedding function
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Get response from llm of user asked question
def get_llm_response(llm, prompt, question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response
