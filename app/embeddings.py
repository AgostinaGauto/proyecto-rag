"""
Módulo de Embeddings:
Utiliza HuggingFace en lugar de Ollama para que funcione tanto en local como en la nube (Render).
"""

from langchain_huggingface import HuggingFaceEmbeddings


def obtener_modelo_embeddings():
    """Retorna un modelo de embeddings ligero de HuggingFace optimizado para CPU."""
    print("[INFO] Cargando modelo de embeddings HuggingFace (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )