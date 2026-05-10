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
        background-color: #f0f2f6;
    }
    .stChatMessage {
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stChatInputContainer {
        padding-bottom: 30px;
    }
    h1 {
        color: #1E3A8A;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #3B82F6;
        color: white;
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
    # Sidebar interactiva
    with st.sidebar:
        st.title("⚙️ Configuración")
        
        # Menú desplegable novedoso
        personalidad = st.selectbox(
            "Elige la personalidad del bot:",
            ["Narrador Clásico", "Sabio Anciano", "Hada Curiosa", "Crítico Literario"],
            help="Cambia la forma en que el bot responde a tus preguntas"
        )
        
        st.divider()
        
        # Botones de acción
        if st.button("🗑️ Limpiar Historial", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("✨ Sugerir Pregunta", use_container_width=True):
            st.info("Prueba con: '¿Cuál es la moraleja de la historia?'")
            
        st.divider()
        
        # Sección 'Acerca de'
        with st.expander("📖 Acerca de este Bot"):
            st.write("""
            Este es un bot inteligente diseñado para interactuar con colecciones de cuentos en formato PDF.
            Utiliza **RAG (Retrieval Augmented Generation)** para dar respuestas precisas basadas en los documentos cargados.
            """)

    st.title("📚 Cuentos Inteligentes")
    st.markdown("<p style='text-align: center;'>Haz preguntas sobre tus cuentos favoritos</p>", unsafe_allow_html=True)
    
    # Mostrar la personalidad seleccionada
    st.caption(f"Personalidad actual: **{personalidad}**")

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
