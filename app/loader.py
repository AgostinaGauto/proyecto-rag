"""
Módulo Loader: Responsable de la carga y extracción de texto de documentos

"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def cargar_pdf(ruta_archivo: str) -> List[Document]:
    """
    Carga un archivo PDF y extrae su contenido en una lista de objetos Document.
    
    Un 'Document' es la estructura estándar de LangChain que contiene:
    - page_content (str): El texto extraído de la página.
    - metadata (dict): Datos sobre el origen (ruta del archivo, número de página).

    Args:
        ruta_archivo (str): Ruta local donde se encuentra el PDF.

    Returns:
        List[Document]: Lista de documentos extraídos por página.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta especificada.
    """
    # Verificación de seguridad básica antes de procesar
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")

    print(f"[INFO] Cargando y procesando PDF: {ruta_archivo}...")
    
    # Inicializamos el cargador oficial de LangChain para PyPDF
    loader = PyPDFLoader(ruta_archivo)
    
    # load() lee el archivo y genera un Document por cada página del PDF
    documentos = loader.load()
    
    print(f"[INFO] Carga exitosa. Se procesaron {len(documentos)} página(s).")
    return documentos


# Este bloque solo se ejecuta si corremos este archivo directamente para hacer pruebas
if __name__ == "__main__":
    # Ruta de prueba relativa para verificar el funcionamiento local
    ruta_prueba = os.path.join("data", "pdf", "politicas_empresa.pdf")
    
    try:
        docs = cargar_pdf(ruta_prueba)
        # Mostramos la primera página extraída para verificar
        if docs:
            print("\n--- Contenido de la primera página ---")
            print(docs[0].page_content[:300] + "...")
            print("---------------------------------------")
            print("Metadatos asociados:", docs[0].metadata)
    except Exception as e:
        print(f"[ERROR] Ocurrió un fallo en la prueba: {e}")