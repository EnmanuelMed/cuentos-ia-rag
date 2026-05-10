import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración básica
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

st.title("📚 Consulta de Cuentos")

pregunta = st.text_input("¿Qué quieres saber de los cuentos?")

if pregunta:
    # Buscamos los trozos de texto más parecidos en tu faiss_index
    documentos_relevantes = vectorstore.similarity_search(pregunta, k=3)
    contexto = "\n".join([doc.page_content for doc in documentos_relevantes])
    
    # Le pasamos el contexto a Groq manualmente
    prompt = f"Basándote en este contenido: {contexto}\n\nResponde a la pregunta: {pregunta}"
    respuesta = llm.invoke(prompt)
    
    st.write(respuesta.content)
