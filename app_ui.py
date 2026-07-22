import os
import requests
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="RAG Corporativo - TechNova",
    page_icon="🤖",
    layout="wide"
)

# URL del backend de FastAPI (dentro de Docker o localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.title("Asistente Virtual Corporativo - TechNova")
st.markdown("Consulta información sobre políticas, productos y procesos internos.")

# Historial de chat en la sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(" Fuentes consultadas"):
                for src in message["sources"]:
                    st.write(f"- **Archivo:** `{src.get('archivo', 'N/A')}`")

# Entrada del usuario
if prompt := st.chat_input("Escribe tu consulta aquí..."):
    # Guardar y mostrar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Llamada al backend RAG
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento..."):
            try:
                response = requests.post(API_URL, json={"pregunta": prompt}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("respuesta", "No se obtuvo respuesta.")
                    sources = data.get("fuentes", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander(" Fuentes consultadas"):
                            for src in sources:
                                st.write(f"- **Archivo:** `{src.get('archivo', 'N/A')}`")

                    # Guardar respuesta del asistente en el historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error en la API ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"No se pudo conectar con el backend: {e}")