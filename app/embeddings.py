from langchain_community.embeddings import FastEmbedEmbeddings

def obtener_modelo_embeddings():
    print("[INFO] Cargando FastEmbed (optimizado para CPU y bajo consumo de RAM)...")
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )