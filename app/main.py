"""
Interfaz de Usuario con Streamlit para el Sistema RAG Corporativo.
Conectado al pipeline local de Ollama con sintaxis moderna de LangChain y FAISS.
Fase 14: Carga dinámica de documentos y actualización del Vectorstore.
"""

import os
import sys
import time
import shutil

# Desactivar advertencia de OpenMP en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Agregar la raíz del proyecto al path de Python para evitar ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag import crear_o_cargar_vectorstore, formatear_documentos, RUTA_VECTORSTORE
from app.embeddings import obtener_modelo_embeddings
from app.loader import cargar_pdf, cargar_csv, dividir_documentos
from langchain_community.vectorstores import FAISS

# Configuración básica de la página web
st.set_page_config(
    page_title="Asistente RAG Corporativo",
    page_icon="",
    layout="centered"
)

st.title(" Asistente RAG Corporativo")
st.caption("Consulta las políticas y productos de la empresa en tiempo real (100% Local con Ollama)")

# Función para regenerar el vectorstore desde cero
def regenerar_vectorstore():
    """Elimina la base de datos actual y la vuelve a construir con todos los documentos."""
    if os.path.exists(RUTA_VECTORSTORE):
        shutil.rmtree(RUTA_VECTORSTORE)
    
    constructor_embeddings = obtener_modelo_embeddings()
    ruta_pdf_dir = os.path.join("data", "pdf")
    ruta_csv_dir = os.path.join("data", "csv")
    
    documentos_cargados = []
    
    # Cargar todos los PDFs en la carpeta
    if os.path.exists(ruta_pdf_dir):
        for archivo in os.listdir(ruta_pdf_dir):
            if archivo.endswith(".pdf"):
                documentos_cargados.extend(cargar_pdf(os.path.join(ruta_pdf_dir, archivo)))
                
    # Cargar todos los CSVs en la carpeta
    if os.path.exists(ruta_csv_dir):
        for archivo in os.listdir(ruta_csv_dir):
            if archivo.endswith(".csv"):
                documentos_cargados.extend(cargar_csv(os.path.join(ruta_csv_dir, archivo)))

    if not documentos_cargados:
        raise ValueError("No hay documentos guardados para procesar.")

    chunks = dividir_documentos(documentos_cargados, chunk_size=300, chunk_overlap=50)
    db_vectorial = FAISS.from_documents(chunks, constructor_embeddings)
    db_vectorial.save_local(RUTA_VECTORSTORE)
    
    # Limpiar caché de Streamlit para recargar la nueva base de datos
    st.cache_resource.clear()

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
    2. NO asumas, deduzcas ni inventes detalles de tiempo (como meses o semanas) si el texto no los especifica explícitamente.
    3. Si la respuesta no está en el contexto, di exactamente que no dispones de esa información.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta en español:""")
    
    return retriever, llm, prompt


# -------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR): CARGA DINÁMICA DE DOCUMENTOS
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Base de Conocimiento")
    st.write("Sube nuevos archivos PDF o CSV para incorporar al sistema RAG.")
    
    archivos_subidos = st.file_uploader(
        "Selecciona un archivo:",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("Procesar e Indexar Documentos", use_container_width=True):
        if archivos_subidos:
            with st.spinner("Guardando e indexando nuevos documentos..."):
                try:
                    for archivo in archivos_subidos:
                        ext = archivo.name.split(".")[-1].lower()
                        if ext == "pdf":
                            destino_dir = os.path.join("data", "pdf")
                        else:
                            destino_dir = os.path.join("data", "csv")
                            
                        os.makedirs(destino_dir, exist_ok=True)
                        ruta_destino = os.path.join(destino_dir, archivo.name)
                        
                        # Guardar el archivo localmente
                        with open(ruta_destino, "wb") as f:
                            f.write(archivo.getbuffer())
                    
                    # Reconstruir el vectorstore
                    regenerar_vectorstore()
                    st.success("¡Base de conocimiento actualizada con éxito!")
                except Exception as e:
                    st.error(f"Error al procesar los archivos: {e}")
        else:
            st.warning("Por favor, selecciona al menos un archivo primero.")
            
    st.divider()
    st.caption("Archivos actuales en el sistema:")
    
    # Listar los archivos subidos actualmente
    for folder, icon in [("pdf", ""), ("csv", "")]:
        folder_path = os.path.join("data", folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if not f.startswith(".")]
            for f in files:
                st.write(f"{icon} `{f}`")


# -------------------------------------------------------------------
# HISTORIAL DE CHAT Y STREAMLIT UI
# -------------------------------------------------------------------
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
        
        # Mostrar métricas si existen
        if msg.get("metrics"):
            m = msg["metrics"]
            st.caption(
                f"⏱️ **Métricas:** Total: `{m['total_time']:.2f}s` | "
                f"Búsqueda FAISS: `{m['retrieval_time']:.2f}s` | "
                f"Generación LLM: `{m['llm_time']:.2f}s`"
            )
            
        # Mostrar fuentes si existen
        if msg.get("sources"):
            with st.expander("Ver fuentes consultadas"):
                for i, doc in enumerate(msg["sources"], 1):
                    origen = os.path.basename(doc.metadata.get("source", "Desconocido"))
                    st.markdown(f"**Fuente {i}:** `{origen}`")
                    st.caption(f"_{doc.page_content.strip()}_")

# Entrada de texto del usuario
if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({
        "role": "user", 
        "content": pregunta_usuario, 
        "sources": [], 
        "metrics": None
    })
    st.chat_message("user").write(pregunta_usuario)

    # Generar respuesta cronometrando
    with st.chat_message("assistant"):
        with st.spinner("Consultando la base de conocimiento local..."):
            try:
                retriever, llm, prompt = obtener_componentes_rag()
                
                t0_inicio = time.perf_counter()
                
                # Búsqueda FAISS
                t_busqueda_inicio = time.perf_counter()
                documentos_fuente = retriever.invoke(pregunta_usuario)
                tiempo_retrieval = time.perf_counter() - t_busqueda_inicio
                
                contexto_texto = formatear_documentos(documentos_fuente)
                
                # Generación LLM
                t_llm_inicio = time.perf_counter()
                cadena_generacion = prompt | llm | StrOutputParser()
                respuesta_texto = cadena_generacion.invoke({
                    "context": contexto_texto,
                    "question": pregunta_usuario
                })
                tiempo_llm = time.perf_counter() - t_llm_inicio
                tiempo_total = time.perf_counter() - t0_inicio
                
                metricas = {
                    "total_time": tiempo_total,
                    "retrieval_time": tiempo_retrieval,
                    "llm_time": tiempo_llm
                }

                st.write(respuesta_texto)
                
                st.caption(
                    f" **Métricas:** Total: `{tiempo_total:.2f}s` | "
                    f"Búsqueda FAISS: `{tiempo_retrieval:.2f}s` | "
                    f"Generación LLM: `{tiempo_llm:.2f}s`"
                )
                
                if documentos_fuente:
                    with st.expander("📚 Ver fuentes consultadas"):
                        for i, doc in enumerate(documentos_fuente, 1):
                            origen = os.path.basename(doc.metadata.get("source", "Desconocido"))
                            st.markdown(f"**Fuente {i}:** `{origen}`")
                            st.caption(f"_{doc.page_content.strip()}_")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": respuesta_texto,
                    "sources": documentos_fuente,
                    "metrics": metricas
                })
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")