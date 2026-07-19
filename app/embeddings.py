"""
Módulo Embeddings: Responsable de inicializar el modelo de representación 
vectorial semántica utilizando Hugging Face.
Sigue los lineamientos de PEP8 y modularidad profesional.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def obtener_modelo_embeddings(nombre_modelo: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Inicializa y retorna el modelo de embeddings de Hugging Face.
    Este modelo correrá localmente en la CPU.

    Args:
        nombre_modelo (str): Nombre del modelo en el hub de Hugging Face.

    Returns:
        HuggingFaceEmbeddings: Instancia del modelo lista para vectorizar texto.
    """
    print(f"[INFO] Inicializando modelo de embeddings: {nombre_modelo}...")
    
    # Especificamos parámetros de configuración
    # 'cpu' asegura que funcione en cualquier computadora sin requerir placa de video NVIDIA (GPU)
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # Normaliza para facilitar el cálculo de distancia cósica
    
    embeddings = HuggingFaceEmbeddings(
        model_name=nombre_modelo,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("[INFO] Modelo de embeddings cargado correctamente en memoria.")
    return embeddings


# Bloque de prueba local
if __name__ == "__main__":
    try:
        # 1. Instanciar el modelo
        constructor_embeddings = obtener_modelo_embeddings()
        
        # 2. Prueba conceptual: Vectorizar dos frases similares y una diferente
        frase_1 = "El precio de la laptop es de 1200 dólares."
        frase_2 = "¿Cuánto cuesta la computadora portátil?"
        
        print("\n[INFO] Generando embeddings de prueba...")
        vector_1 = constructor_embeddings.embed_query(frase_1)
        vector_2 = constructor_embeddings.embed_query(frase_2)
        
        print(f"✓ Embedding 1 generado. Dimensiones del vector: {len(vector_1)}")
        print(f"✓ Embedding 2 generado. Dimensiones del vector: {len(vector_2)}")
        print(f"Muestra de los primeros 5 números del Vector 1: {vector_1[:5]}")
        
    except Exception as e:
        print(f"[ERROR]: Ocurrió un fallo al cargar los embeddings: {e}")