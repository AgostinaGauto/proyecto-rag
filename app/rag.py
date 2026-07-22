"""
Módulo RAG - Componente Vectorial y Generación con LCEL:
Soporta generación híbrida: Groq en la nube (Render) y Ollama local.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from app.loader import cargar_pdf, cargar_csv, dividir_documentos
from app.embeddings import obtener_modelo_embeddings

# Desactivar advertencia de OpenMP en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ruta estándar de persistencia
RUTA_VECTORSTORE = "vectorstore"


def obtener_llm():
    """Retorna ChatGroq si la API Key está presente; si no, usa Ollama local."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        print("[INFO] Conectando con LLM en la nube (Groq: llama-3.1-8b-instant)...")
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            groq_api_key=groq_api_key
        )
    else:
        print("[INFO] Conectando con LLM local (Ollama: llama3.2)...")
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="llama3.2", temperature=0.0)


def crear_o_cargar_vectorstore() -> FAISS:
    """Busca una base de datos vectorial FAISS existente o crea una nueva."""
    constructor_embeddings = obtener_modelo_embeddings()

    if os.path.exists(os.path.join(RUTA_VECTORSTORE, "index.faiss")):
        print(f"[INFO] Detectada base de datos local. Cargando FAISS desde '{RUTA_VECTORSTORE}'...")
        return FAISS.load_local(RUTA_VECTORSTORE, constructor_embeddings, allow_dangerous_deserialization=True)

    print("[INFO] No se encontró una base de datos vectorial previa. Creando una nueva...")
    ruta_pdf = os.path.join("data", "pdf", "politicas_empresa.pdf")
    ruta_csv = os.path.join("data", "csv", "productos.csv")
    
    documentos_cargados = []
    if os.path.exists(ruta_pdf): documentos_cargados.extend(cargar_pdf(ruta_pdf))
    if os.path.exists(ruta_csv): documentos_cargados.extend(cargar_csv(ruta_csv))
        
    if not documentos_cargados:
        raise ValueError("No se encontraron archivos válidos en data/pdf/ o data/csv/.")

    chunks = dividir_documentos(documentos_cargados, chunk_size=300, chunk_overlap=50)
    db_vectorial = FAISS.from_documents(chunks, constructor_embeddings)
    db_vectorial.save_local(RUTA_VECTORSTORE)
    return db_vectorial


def formatear_documentos(docs) -> str:
    """Une el contenido de los fragmentos recuperados en un único bloque de texto."""
    return "\n\n".join(doc.page_content for doc in docs)


def ejecutar_sistema_rag(pregunta: str):
    """Ejecuta el pipeline RAG completo usando la arquitectura pura LCEL de LangChain."""
    print(f"\n[SISTEMA RAG] Pregunta: {pregunta}")
    
    try:
        # 1. Obtener base de datos de conocimiento y configurar el recuperador con MMR
        db = crear_o_cargar_vectorstore()
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 8}
        )
        
        # 2. Configurar el LLM (Groq en la nube u Ollama local)
        llm = obtener_llm()
        
        # 3. Definir el prompt del sistema
        prompt = ChatPromptTemplate.from_template("""
        Eres un asistente corporativo preciso y literal. Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto provisto.
        
        Reglas estrictas:
        1. Sé directo y aférrate exactamente a la información del texto.
        2. NO asumas, deduzcas ni inventes detalles de tiempo si el texto no los especifica explícitamente.
        3. Si la respuesta no está en el contexto, di exactamente que no dispones de esa información.

        Contexto:
        {context}

        Pregunta: {question}

        Respuesta en español:""")
        
        # 4. Construir la cadena RAG con LCEL
        cadena_rag = (
            {"context": retriever | formatear_documentos, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 5. Ejecutar la consulta
        respuesta_final = cadena_rag.invoke(pregunta)
        
        print("\n" + "="*50)
        print("[RESPUESTA DE LA IA]:")
        print(respuesta_final)
        print("="*50 + "\n")
        
        return respuesta_final
        
    except Exception as e:
        print(f"[ERROR EN RAG]: {e}")
        return f"Error al procesar la solicitud: {e}"


if __name__ == "__main__":
    consulta = "¿Cuánto presupuesto tengo para equipar mi oficina en casa y qué días debo ir a trabajar físicamente?"
    ejecutar_sistema_rag(consulta)