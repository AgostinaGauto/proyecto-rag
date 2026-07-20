"""
Interfaz de Usuario con Streamlit para el Sistema RAG Corporativo.
Conectado al pipeline local de Ollama con sintaxis moderna de LangChain y FAISS.
Fase 12: Atribución y visualización de fuentes consultadas en tiempo real.
"""

import os
import sys

# Desactivar advertencia de OpenMP en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Agregar la raíz del proyecto al path de Python para evitar ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from app.rag import crear_o_cargar_vectorstore, formatear_documentos

# Configuración básica de la página web
st.set_page_config(
    page_title="Asistente RAG Corporativo",
    page_icon="🤖",
    layout="centered"
)

st.title(" Asistente RAG Corporativo")
st.caption("Consulta las políticas y productos de la empresa en tiempo real (100% Local con Ollama)")

# Cargar la base de datos vectorial de forma eficiente (usando caché de Streamlit)
@st.cache_resource(show_spinner=False)
def obtener_cadena_rag():
    """Inicializa la base de datos y la cadena RAG devolviendo tanto la respuesta como las fuentes."""
    db = crear_o_cargar_vectorstore()
    
    # Búsqueda por Relevancia Mínima (MMR) con k=4
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 8}
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.0)
    
    # Prompt estricto para evitar deducciones erróneas
    prompt = ChatPromptTemplate.from_template("""
    Eres un asistente corporativo preciso y literal. Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto provisto.
    
    Reglas strictly:
    1. Sé directo y aférrate exactamente a la información del texto.
    2. NO asumas, deduzcas ni inventes detalles de tiempo (como meses o semanas) si el texto no los especifica explícitamente.
    3. Si la respuesta no está en el contexto, di exactamente que no dispones de esa información.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta en español:""")

    # Cadena que recupera los documentos en paralelo para exponerlos como fuente
    cadena_rag = RunnableParallel({
        "docs": retriever,
        "question": RunnablePassthrough()
    }).assign(
        answer=(
            {"context": lambda x: formatear_documentos(x["docs"]), "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
        )
    )
    
    return cadena_rag

# Inicializar el estado del historial de chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "¡Hola! Soy tu asistente corporativo. ¿En qué puedo ayudarte hoy sobre las políticas o productos?",
            "sources": []
        }
    ]

# Renderizar los mensajes del historial en la pantalla
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Mostrar fuentes si existen en el mensaje
        if msg.get("sources"):
            with st.expander("Ver fuentes consultadas"):
                for i, doc in enumerate(msg["sources"], 1):
                    origen = os.path.basename(doc.metadata.get("source", "Desconocido"))
                    st.markdown(f"**Fuente {i}:** `{origen}`")
                    st.caption(f"_{doc.page_content.strip()}_")

# Entrada de texto del usuario
if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí..."):
    # Guardar y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": pregunta_usuario, "sources": []})
    st.chat_message("user").write(pregunta_usuario)

    # Generar y mostrar la respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Consultando la base de conocimiento local..."):
            try:
                cadena_rag = obtener_cadena_rag()
                resultado = cadena_rag.invoke(pregunta_usuario)
                
                respuesta_texto = resultado["answer"]
                documentos_fuente = resultado["docs"]
                
                st.write(respuesta_texto)
                
                # Mostrar el desplegable de fuentes inmediatamente
                if documentos_fuente:
                    with st.expander("Ver fuentes consultadas"):
                        for i, doc in enumerate(documentos_fuente, 1):
                            origen = os.path.basename(doc.metadata.get("source", "Desconocido"))
                            st.markdown(f"**Fuente {i}:** `{origen}`")
                            st.caption(f"_{doc.page_content.strip()}_")

                # Guardar la respuesta y sus fuentes en el historial
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": respuesta_texto,
                    "sources": documentos_fuente
                })
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")