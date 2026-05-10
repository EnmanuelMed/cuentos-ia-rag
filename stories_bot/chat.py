import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Mágico Cuentos Bot", page_icon="📖", layout="centered")

# Cargar CSS externo
def local_css(file_name):
    with open(f"stories_bot/static/{file_name}") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Configuración de IA
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    return vectorstore, llm

vectorstore, llm = load_resources()

# Barra Lateral
with st.sidebar:
    st.title("🌟 Panel Mágico")
    st.markdown("---")
    
    if st.button("🗑️ Borrar Conversación"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    with st.expander("📖 Sobre el Bot"):
        st.write("Este asistente utiliza inteligencia artificial para explorar el mundo de tus cuentos en PDF.")

# Interfaz Principal
st.title("📚 Consulta de Cuentos")
st.caption("Tu guía mágico a través de las historias")

# Historial de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de Usuario
if prompt := st.chat_input("Hazme una pregunta sobre los cuentos..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los libros..."):
            # Búsqueda RAG
            documentos_relevantes = vectorstore.similarity_search(prompt, k=3)
            contexto = "\n".join([doc.page_content for doc in documentos_relevantes])
            
            full_prompt = (
                f"Eres un bibliotecario mágico y experto en cuentos. "
                f"Basándote estrictamente en este contenido: {contexto}\n\n"
                f"Responde de forma amable y detallada a la pregunta: {prompt}"
            )
            
            respuesta = llm.invoke(full_prompt)
            st.markdown(respuesta.content)
            
            # Guardar respuesta
            st.session_state.messages.append({"role": "assistant", "content": respuesta.content})
