import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdfs(pdf_folder="pdfs", vector_store_path="faiss_index"):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            print(f"Loading {file}...")
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    if not documents:
        print("No PDFs found in the folder.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vector_store_path)
    print(f"Vector store saved to {vector_store_path}")

if __name__ == "__main__":
    ingest_pdfs()
