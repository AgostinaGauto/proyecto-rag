

#  Challenge Alura - Agente Inteligente RAG Corporativo

Este repositorio contiene la solución desarrollada para el **Challenge Alura Agente**. Consiste en un sistema de **Retrieval-Augmented Generation (RAG)** estructurado como una API REST que permite realizar consultas a un agente inteligente alimentado con la información contenida en documentos internos (archivos PDF y CSV).

---

##  Descripción General del Proyecto

El **Asistente RAG Corporativo** es un agente de IA diseñado para actuar como un canal de consulta unificado sobre políticas internas de la empresa y catálogo de productos. El sistema procesa archivos PDF y CSV, genera índices vectoriales y utiliza un modelo de lenguaje avanzado para responder preguntas de forma precisa y fundamentada exclusivamente en la documentación provista, citando las fuentes consultadas.

---

##  Arquitectura de la Solución

El flujo de procesamiento de la información y generación de respuestas sigue la siguiente arquitectura:

1. **Ingesta y Procesamiento de Documentos:** Se leen archivos PDF y CSV almacenados en el directorio `data/`. Los textos se dividen en fragmentos mediante *text splitters* optimizados.
2. **Generación de Embeddings:** Se utiliza el modelo `BAAI/bge-small-en-v1.5` de **FastEmbed** (ejecutado en ONNX) para transformar el texto en vectores de alto rendimiento con muy bajo uso de memoria.
3. **Almacenamiento Vectorial (Retriever):** Los vectores se almacenan en un índice **FAISS**. Para las consultas, se utiliza una estrategia de búsqueda por similitud con *Maximal Marginal Relevance* (MMR) para garantizar diversidad y relevancia en los contextos recuperados.
4. **Agente RAG / LLM:** Los datos recuperados se envían junto con la consulta del usuario a un modelo **Llama 3.1 (8B Instant)** a través de la API de **Groq**, generando la respuesta final.
5. **Interfaz API:** Todo el pipeline está expuesto a través de una API REST construida con **FastAPI**.

---

##  Tecnologías y Herramientas Utilizadas

* **Lenguaje:** Python 3.10+
* **Framework Web:** FastAPI + Uvicorn
* **Orquestación RAG:** LangChain
* **LLM:** ChatGroq (`llama-3.1-8b-instant`)
* **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
* **Base de Datos Vectorial:** FAISS
* **Procesamiento de Documentos:** PyPDF, Pandas
* **Despliegue & Contenerización:** Docker, Render / OCI (Oracle Cloud Infrastructure)

Ejemplos de Uso del Agente
Preguntas que el agente puede responder:
"¿Cuál es el presupuesto asignado para equipar la oficina en casa?"

"¿Cuáles son los productos disponibles en el catálogo y sus especificaciones?"

"¿Qué pasos debo seguir para solicitar un reembolso?"

