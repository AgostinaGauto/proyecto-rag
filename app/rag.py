"""
Módulo RAG - Componente Vectorial: Responsable de la creación, almacenamiento
y recuperación semántica utilizando FAISS.
Sigue las buenas prácticas de diseño modular y PEP8.
"""

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.loader import cargar_pdf, cargar_csv, dividir_documentos
from app.embeddings import obtener_modelo_embeddings

# Ruta estándar de persistencia definida en la arquitectura del proyecto
RUTA_VECTORSTORE = "vectorstore"


def crear_o_cargar_vectorstore() -> FAISS:
    """
    Busca una base de datos vectorial FAISS existente en el disco.
    Si existe, la carga; si no, procesa los documentos desde cero,
    genera los embeddings y la guarda para futuros usos.

    Returns:
        FAISS: Instancia de la base de datos vectorial lista para buscar.
    """
    # Inicializamos el motor de embeddings local
    constructor_embeddings = obtener_modelo_embeddings()

    # Escenario A: La base de datos ya fue creada previamente
    if os.path.exists(os.path.join(RUTA_VECTORSTORE, "index.faiss")):
        print(f"[INFO] Detectada base de datos local. Cargando FAISS desde '{RUTA_VECTORSTORE}'...")
        # 'allow_dangerous_deserialization=True' es necesario para cargar archivos FAISS locales mediante pickle
        db_vectorial = FAISS.load_local(
            RUTA_VECTORSTORE, 
            constructor_embeddings, 
            allow_dangerous_deserialization=True
        )
        print("[INFO] Base de datos vectorial cargada exitosamente.")
        return db_vectorial

    # Escenario B: Primera ejecución, la base de datos no existe
    print("[INFO] No se encontró una base de datos vectorial previa. Creando una nueva...")
    
    # 1. Definimos las rutas de nuestros archivos creados en la Fase 3
    ruta_pdf = os.path.join("data", "pdf", "politicas_empresa.pdf")
    ruta_csv = os.path.join("data", "csv", "productos.csv")
    
    # 2. Cargamos los documentos usando nuestro módulo loader
    documentos_cargados = []
    if os.path.exists(ruta_pdf):
        documentos_cargados.extend(cargar_pdf(ruta_pdf))
    if os.path.exists(ruta_csv):
        documentos_cargados.extend(cargar_csv(ruta_csv))
        
    if not documentos_cargados:
        raise ValueError("No se encontraron archivos válidos en las carpetas data/pdf/ o data/csv/ para indexar.")

    # 3. Fragmentamos los documentos en chunks inteligentes
    chunks = dividir_documentos(documentos_cargados, chunk_size=500, chunk_overlap=50)

    # 4. Construimos el índice FAISS pasando los textos y el modelo de embeddings
    print("[INFO] Indexando fragmentos en FAISS (esto generará los embeddings numéricos)...")
    db_vectorial = FAISS.from_documents(chunks, constructor_embeddings)
    
    # 5. Guardamos en el disco local para que la próxima vez sea instantáneo
    db_vectorial.save_local(RUTA_VECTORSTORE)
    print(f"[INFO] Base de datos guardada exitosamente en la carpeta '{RUTA_VECTORSTORE}'.")
    
    return db_vectorial


# Bloque de prueba local para verificar la búsqueda semántica
if __name__ == "__main__":
    try:
        # Inicializamos o cargamos la base de datos
        db = crear_o_cargar_vectorstore()
        
        # Realizamos una prueba de búsqueda semántica (Similitud)
        pregunta_prueba = "¿Cuánto dinero dan para equipar la oficina en casa?"
        print(f"\n[TEST] Ejecutando búsqueda semántica para: '{pregunta_prueba}'")
        
        # k=2 le indica a FAISS que nos traiga los 2 fragmentos con mayor coincidencia de significado
        resultados = db.similarity_search(pregunta_prueba, k=2)
        
        print(f"\n[RESULTADOS ENCONTRADOS: {len(resultados)}]")
        for i, doc in enumerate(resultados):
            print(f"\n--- Fragmento Coincidente #{i+1} (Origen: {doc.metadata['source']}) ---")
            print(doc.page_content)
            print("-" * 40)
            
    except Exception as e:
        print(f"[ERROR]: Ocurrió un fallo en el módulo RAG Vectorial: {e}")