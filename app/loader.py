"""
Módulo Loader: Responsable de la carga y extracción de texto de documentos

"""

import os
from typing import List
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def cargar_pdf(ruta_archivo: str) -> List[Document]:
    """Carga un archivo PDF y extrae su contenido."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")

    print(f"[INFO] Cargando PDF: {ruta_archivo}...")
    loader = PyPDFLoader(ruta_archivo)
    documentos = loader.load()
    return documentos


def cargar_csv(ruta_archivo: str) -> List[Document]:
    """Carga un archivo CSV y transforma cada fila en un texto narrativo."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")

    print(f"[INFO] Cargando CSV: {ruta_archivo}...")
    df = pd.read_csv(ruta_archivo)
    documentos = []

    for index, fila in df.iterrows():
        contenido_narrativo = (
            f"Información del Producto:\n"
            f"- ID: {fila.get('id', 'N/A')}\n"
            f"- Nombre: {fila.get('nombre', 'N/A')}\n"
            f"- Categoría: {fila.get('categoria', 'N/A')}\n"
            f"- Precio: ${fila.get('precio', 'N/A')} USD\n"
            f"- Unidades en stock: {fila.get('stock', 'N/A')}"
        )
        metadatos = {
            "source": ruta_archivo,
            "row": index,
            "id": str(fila.get('id', ''))
        }
        documentos.append(Document(page_content=contenido_narrativo, metadata=metadatos))
    return documentos


def dividir_documentos(documentos: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Divide una lista de documentos en fragmentos (chunks) más pequeños y homogéneos
    utilizando una estrategia recursiva inteligente.

    Args:
        documentos (List[Document]): Documentos originales cargados.
        chunk_size (int): Tamaño máximo de caracteres por fragmento.
        chunk_overlap (int): Cantidad de caracteres que se solapan entre fragmentos.

    Returns:
        List[Document]: Lista de nuevos documentos fragmentados.
    """
    print(f"[INFO] Iniciando división de documentos (Tamaño: {chunk_size}, Solapamiento: {chunk_overlap})...")
    
    # Configuramos el divisor recursivo profesional
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documentos)
    print(f"[INFO] División completada. Generados {len(chunks)} fragmentos en total.")
    return chunks


# Bloque de prueba local
if __name__ == "__main__":
    ruta_pdf = os.path.join("data", "pdf", "politicas_empresa.pdf")
    ruta_csv = os.path.join("data", "csv", "productos.csv")
    
    try:
        # 1. Cargar fuentes
        docs_pdf = cargar_pdf(ruta_pdf)
        docs_csv = cargar_csv(ruta_csv)
        
        # Combinamos ambas listas de documentos cargados
        todos_los_documentos = docs_pdf + docs_csv
        
        # 2. Fragmentar de forma unificada
        fragmentos_totales = dividir_documentos(todos_los_documentos, chunk_size=300, chunk_overlap=30)
        
        # 3. Mostrar una muestra del procesamiento
        if len(fragmentos_totales) > 0:
            print("\n" + "="*40)
            print(f"--- Muestra del Chunk #0 (Origen: {fragmentos_totales[0].metadata['source']}) ---")
            print(fragmentos_totales[0].page_content)
            print("="*40)
            
    except Exception as e:
        print(f"[ERROR]: {e}")