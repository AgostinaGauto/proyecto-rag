"""
Módulo de Embeddings Local: Configura el modelo de vectores utilizando Ollama.
Elimina dependencias externas conflictivas y permite ejecución 100% local.
"""

from langchain_ollama import OllamaEmbeddings


def obtener_modelo_embeddings() -> OllamaEmbeddings:
    """
    Inicializa y retorna el constructor de embeddings local de Ollama.
    
    Returns:
        OllamaEmbeddings: Instancia lista para vectorizar documentos.
    """
    print("[INFO] Inicializando modelo de embeddings local: nomic-embed-text...")
    return OllamaEmbeddings(model="nomic-embed-text")