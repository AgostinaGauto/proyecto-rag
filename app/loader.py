"""
Módulo Loader: Responsable de la carga y extracción de texto de documentos

"""

import os
from typing import List
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def cargar_pdf(ruta_archivo: str) -> List[Document]:
    """
    Carga un archivo PDF y extrae su contenido en una lista de objetos Document.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")

    print(f"[INFO] Cargando y procesando PDF: {ruta_archivo}...")
    loader = PyPDFLoader(ruta_archivo)
    documentos = loader.load()
    print(f"[INFO] Carga exitosa. Se procesaron {len(documentos)} página(s).")
    return documentos


def cargar_csv(ruta_archivo: str) -> List[Document]:
    """
    Carga un archivo CSV utilizando Pandas y transforma cada fila
    en un texto narrativo estructurado en un objeto Document de LangChain.

    Args:
        ruta_archivo (str): Ruta local donde se encuentra el CSV.

    Returns:
        List[Document]: Lista de documentos creados (uno por cada fila).
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")

    print(f"[INFO] Cargando y procesando CSV: {ruta_archivo}...")
    
    # Leemos el archivo CSV usando Pandas
    df = pd.read_csv(ruta_archivo)
    documentos = []

    # Iteramos sobre cada fila de la tabla
    for index, fila in df.iterrows():
        # Construimos un texto narrativo descriptivo para que el LLM lo entienda fácilmente
        contenido_narrativo = (
            f"Información del Producto:\n"
            f"- ID: {fila.get('id', 'N/A')}\n"
            f"- Nombre: {fila.get('nombre', 'N/A')}\n"
            f"- Categoría: {fila.get('categoria', 'N/A')}\n"
            f"- Precio: ${fila.get('precio', 'N/A')} USD\n"
            f"- Unidades en stock: {fila.get('stock', 'N/A')}"
        )
        
        # Guardamos metadatos útiles para poder filtrar u organizar la información después
        metadatos = {
            "source": ruta_archivo,
            "row": index,
            "id": str(fila.get('id', ''))
        }
        
        # Creamos el objeto Document estándar de LangChain
        doc = Document(page_content=contenido_narrativo, metadata=metadatos)
        documentos.append(doc)

    print(f"[INFO] Carga exitosa. Se procesaron {len(documentos)} fila(s) del CSV.")
    return documentos


# Este bloque solo se ejecuta si corremos este archivo directamente para hacer pruebas
if __name__ == "__main__":
    # 1. Prueba de carga de PDF (ya verificada)
    ruta_pdf = os.path.join("data", "pdf", "politicas_empresa.pdf")
    try:
        docs_pdf = cargar_pdf(ruta_pdf)
    except Exception as e:
        print(f"[ERROR PDF]: {e}")

    # 2. Nueva prueba de carga de CSV
    ruta_csv = os.path.join("data", "csv", "productos.csv")
    try:
        print("\n" + "="*40 + "\n")
        docs_csv = cargar_csv(ruta_csv)
        if docs_csv:
            print("\n--- Contenido del primer registro transformado ---")
            print(docs_csv[0].page_content)
            print("---------------------------------------")
            print("Metadatos asociados:", docs_csv[0].metadata)
    except Exception as e:
        print(f"[ERROR CSV]: {e}")