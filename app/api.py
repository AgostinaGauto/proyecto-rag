"""
API REST con FastAPI para el Sistema RAG Corporativo.
Expone un endpoint para realizar consultas al pipeline de Ollama + FAISS.
Fase 12: Integración de API REST.
"""

import os
import sys
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# Agregar la raíz del proyecto al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag import crear_o_cargar_vectorstore, formatear_documentos
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Inicializar FastAPI
app = FastAPI(
    title="API RAG Corporativo",
    description="Endpoint REST para consultar políticas y productos usando Ollama y FAISS.",
    version="1.0.0"
)

# Modelos de Pydantic para la validación de entrada/salida
class ConsultaRequest(BaseModel):
    pregunta: str = Field(..., example="¿Cuál es el presupuesto para la oficina en casa?")

class FuenteResponse(BaseModel):
    archivo: str
    contenido: str

class ConsultaResponse(BaseModel):
    respuesta: str
    fuentes: List[FuenteResponse]

# Carga e inicialización global de la cadena RAG
def obtener_cadena_rag():
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
    2. NO asumas, deduzcas ni inventes detalles de tiempo si el texto no los especifica explícitamente.
    3. Si la respuesta no está en el contexto, di exactamente que no dispones de esa información.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta en español:""")

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

# Instancia global de la cadena
cadena_rag = None

@app.on_event("startup")
def startup_event():
    global cadena_rag
    print("Cargando base de datos vectorial y modelo RAG...")
    cadena_rag = obtener_cadena_rag()
    print("¡Sistema RAG listo para recibir solicitudes!")

@app.get("/")
def read_root():
    return {"mensaje": "Bienvenido a la API del Asistente RAG Corporativo. Visita /docs para la documentación."}

@app.post("/query", response_model=ConsultaResponse)
def consultar_rag(request: ConsultaRequest):
    if not request.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    try:
        resultado = cadena_rag.invoke(request.pregunta)
        
        fuentes_formateadas = [
            FuenteResponse(
                archivo=os.path.basename(doc.metadata.get("source", "Desconocido")),
                contenido=doc.page_content.strip()
            )
            for doc in resultado["docs"]
        ]
        
        return ConsultaResponse(
            respuesta=resultado["answer"],
            fuentes=fuentes_formateadas
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno procesando la consulta: {str(e)}")