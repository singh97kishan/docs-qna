
import streamlit as st
import os
from PyPDF2 import PdfReader
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS, Chroma
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_data 
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=google_embeddings)
    vector_store.save_local("faiss_db")
    return vector_store

def get_llm_chain(user_question, vectorstore):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the asnwer is not in provided context just respond, "Answer is not available in the context", don't provide
    the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0.1)


    google_embeddings =  GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #vectorstore = FAISS.load_local("faiss_db", google_embeddings)
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    llm_response = str(rag_chain.invoke(user_question))
    return llm_response

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

import time

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.set_page_config("Document QnA", layout="centered")
st.title("Document QnA Chatbot :robot_face:")

with st.sidebar:
    uploaded_file = st.file_uploader("**Upload PDFs**", type=['pdf'], accept_multiple_files=True)
    if st.button("Generate and Save as Vectos"):
        if uploaded_file:
            with st.spinner("Processing..."):
                text = get_pdf_text(uploaded_file)
                chunks = get_text_chunks(text)
                vectorstore = get_vector_store(chunks)
                if 'vectorstore' not in st.session_state.keys():
                    st.session_state['vectorstore'] = vectorstore

        st.success("Vectors saved successfully ✅")
    st.button('Clear Chat History', on_click=clear_chat_history)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today??"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message['content'])

if user_prompt := st.chat_input("Type your query here"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Please wait.."):
            response = str(get_llm_chain(user_prompt, st.session_state['vectorstore']))
            st.write_stream(response_generator(response))

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)


