import os
import pickle
import streamlit as st

from langchain_community.llms import Together
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Page config
st.set_page_config(page_title="News Research Bot", layout="wide")
st.title("🗞️ News Research Bot")

# Sidebar for URL inputs
st.sidebar.title("🔗 Enter News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

if st.sidebar.button("🚀 Process URLs"):
    if urls:
        with st.spinner("🔄 Fetching and processing documents..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(data)

            # Create HuggingFace embeddings and vectorstore
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embedding_model)

            # Save vectorstore
            with open("vectorstore.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            st.success("✅ Documents processed and saved successfully.")
    else:
        st.warning("⚠️ Please enter at least one valid URL.")

# Input query
query = st.text_input("💬 Ask a question based on the articles:")

if query:
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()

        # Setup Together LLM
        llm = Together(
            model="meta-llama/Llama-3-8b-chat-hf",
            temperature=0.9,
            max_tokens=512,
            top_p=0.7,
            together_api_key="068e12af3d4d3228e1763d2065910ef9ef34fcebbab26610433742976476a2ec"  # Replace for production
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        with st.spinner("🔎 Searching for the answer..."):
            response = qa_chain.invoke({"query": query})

        st.markdown("### ✅ Answer")
        st.markdown(response["result"])

        # Display sources
        if response.get("source_documents"):
            st.markdown("---")
            st.markdown("### 📚 Sources")
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i + 1}:** [{doc.metadata.get('source', 'Unknown')}]")
                st.code(doc.page_content[:300] + "...")

        # Offer download
        with open("vectorstore.pkl", "rb") as f:
            st.download_button("📦 Download Vectorstore", f, "vectorstore.pkl")

    else:
        st.error("❌ Vectorstore not found. Please process URLs first.")
