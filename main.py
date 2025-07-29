import os
import streamlit as st
import pickle
import time

from langchain_community.llms import Together
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

st.title("News Research Bot")
st.sidebar.title("🔍 News Article URLs")

llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.9,
    max_tokens=512,
    top_p=0.7,
    together_api_key="068e12af3d4d3228e1763d2065910ef9ef34fcebbab26610433742976476a2ec"  # ← Replace with your actual key
)

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked=st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading in progress...")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Data loaded successfully. Splitting into chunks...")
    docs = text_splitter.split_documents(data)
    #create embeddings and save it to FAISS index or vectorstore
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    #save the vectorstore to disk
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    
query= main_placeholder.text_input("Ask a question about the news articles:")
if query:
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            # Create retriever
            retriever = vectorstore.as_retriever()
            # Build RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
            )
            response = qa_chain.invoke({"query": query})
            st.header("Answer:")
            st.subheader(response['result'])
            
