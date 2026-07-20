"""
Interfaz de Usuario con Streamlit para el Sistema RAG Corporativo.
Conectado al pipeline local de Ollama con sintaxis moderna de LangChain y FAISS.
Fase 13: Métricas de latencia y telemetría en tiempo real.
"""

import os
import sys
import time

# Desactivar advertencia de OpenMP en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Agregar la raíz del proyecto al path de Python para evitar ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag import crear_o_cargar_vectorstore, formatear_documentos

# Configuración básica de la página web
st.set_page_config(
    page_title="Asistente RAG Corporativo",
    page_icon="🤖",
    layout="centered"
)

st.title(" Asistente RAG Corporativo")
st.caption("Consulta las políticas y productos de la empresa en tiempo real (100% Local con Ollama)")

# Cargar la base de datos vectorial y modelo de forma eficiente
@st.cache_resource(show_spinner=False)
def obtener_componentes_rag():
    """Inicializa la base de datos vectorial y el modelo LLM."""
    db = crear_o_cargar_vectorstore()
    
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 8}
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_template("""
    Eres un asistente corporativo preciso y literal. Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto provisto.
    
    Reglas estrictas:
    1. Sé directo y aférrate exactamente a la información del texto.
    2. NO asumas, deduzcas ni inventes detalles de tiempo (como meses o semanas) si el texto no los especifica explicítamente.
    3. Si la respuesta no está en el contexto, di exactamente que no dispones de esa información.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta en español:""")
    
    return retriever, llm, prompt

# Inicializar el estado del historial de chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "¡Hola! Soy tu asistente corporativo. ¿En qué puedo ayudarte hoy sobre las políticas o productos?",
            "sources": [],
            "metrics": None
        }
    ]

# Renderizar los mensajes del historial en la pantalla
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Mostrar métricas de rendimiento si existen
        if msg.get("metrics"):
            m = msg["metrics"]
            st.caption(
                f" **Métricas:** Total: `{m['total_time']:.2f}s` | "
                f"Búsqueda FAISS: `{m['retrieval_time']:.2f}s` | "
                f"Generación LLM: `{m['llm_time']:.2f}s`"
            )
            
        # Mostrar fuentes si existen
        if msg.get("sources"):
            with st.expander(" Ver fuentes consultadas"):
                for i, doc in enumerate(msg["sources"], 1):
                    origen = os.path.basename(doc.metadata.get("source", "Desconocido"))
                    st.markdown(f"**Fuente {i}:** `{origen}`")
                    st.caption(f"_{doc.page_content.strip()}_")

# Entrada de texto del usuario
if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí..."):
    # Guardar y mostrar el mensaje del usuario
    st.session_state.messages.append({
        "role": "user", 
        "content": pregunta_usuario, 
        "sources": [], 
        "metrics": None
    })
    st.chat_message("user").write(pregunta_usuario)

    # Generar y mostrar la respuesta de la IA cronometrando cada paso
    with st.chat_message("assistant"):
        with st.spinner("Consultando la base de conocimiento local..."):
            try:
                retriever, llm, prompt = obtener_componentes_rag()
                
                t0_inicio = time.perf_counter()
                
                # 1. Medir tiempo de Búsqueda Semántica en FAISS
                t_busqueda_inicio = time.perf_counter()
                documentos_fuente = retriever.invoke(pregunta_usuario)
                t_busqueda_fin = time.perf_counter()
                tiempo_retrieval = t_busqueda_fin - t_busqueda_inicio
                
                # Formatear contexto
                contexto_texto = formatear_documentos(documentos_fuente)
                
                # 2. Medir tiempo de Generación en Llama 3.2
                t_llm_inicio = time.perf_counter()
                cadena_generacion = prompt | llm | StrOutputParser()
                respuesta_texto = cadena_generacion.invoke({
                    "context": contexto_texto,
                    "question": pregunta_usuario
                })
                t_llm_fin = time.perf_counter()
                tiempo_llm = t_llm_fin - t_llm_inicio
                
                tiempo_total = time.perf_counter() - t0_inicio
                
                # Guardar métricas
                metricas = {
                    "total_time": tiempo_total,
                    "retrieval_time": tiempo_retrieval,
                    "llm_time": tiempo_llm
                }

                # Mostrar respuesta en pantalla
                st.write(respuesta_texto)
                
                # Mostrar métricas de tiempo
                st.caption(
                    f"⏱️ **Métricas:** Total: `{tiempo_total:.2f}s` | "
                    f"Búsqueda FAISS: `{tiempo_retrieval:.2f}s` | "
                    f"Generación LLM: `{tiempo_llm:.2f}s`"
                )
                
                # Mostrar el desplegable de fuentes inmediatamente
                if documentos_fuente:
                    with st.expander("📚 Ver fuentes consultadas"):
                        for i, doc in enumerate(documentos_fuente, 1):
                            origen = os.path.basename(doc.metadata.get("source", "Desconocido"))
                            st.markdown(f"**Fuente {i}:** `{origen}`")
                            st.caption(f"_{doc.page_content.strip()}_")

                # Guardar en el historial
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": respuesta_texto,
                    "sources": documentos_fuente,
                    "metrics": metricas
                })
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")