import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Cuentos Bot", page_icon="📚")

# CSS Personalizado para el estilo de chat
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    h1 {
        color: #2e4053;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .bot-bubble {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 15px;
        padding: 10px;
        color: #333;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #d1e7dd;
        border-radius: 15px;
        padding: 10px;
        color: #0f5132;
        margin-bottom: 10px;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

def get_qa_chain():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Por favor, configura GROQ_API_KEY en el archivo .env")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists("faiss_index"):
        st.warning("El índice de cuentos no existe. Por favor, procesa los PDFs primero.")
        return None

    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

def main():
    st.title("📚 Cuentos Inteligentes")
    st.markdown("<p style='text-align: center;'>Haz preguntas sobre tus cuentos favoritos</p>", unsafe_allow_html=True)

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes previos
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("¿Qué quieres saber de los cuentos?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            qa_chain = get_qa_chain()
            if qa_chain:
                with st.spinner("Consultando los cuentos..."):
                    response = qa_chain.invoke(prompt)
                    answer = response["result"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Error al cargar el sistema de respuesta.")

if __name__ == "__main__":
    main()
