import streamlit as st
import re
from dotenv import load_dotenv
from rag_chain import DEFAULT_WEB_PATHS, build_agent

# Carga las variables del archivo .env, incluida la API key de OpenRouter.
load_dotenv()

# Configuración general de la página de Streamlit.
st.set_page_config(page_title="RAG documental", layout="wide")

st.title("RAG documental")
st.caption("Pregunta, pega URLs o adjunta documentos desde el propio chat.")

# Estado de la conversación que se muestra en pantalla.
if "messages" not in st.session_state:
    st.session_state.messages = []

# URLs que forman parte del índice vectorial del RAG.
if "web_paths" not in st.session_state:
    st.session_state.web_paths = list(DEFAULT_WEB_PATHS)

# Documentos adjuntos guardados en memoria para poder reconstruir el índice.
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

# Expresión regular sencilla para detectar URLs pegadas en el chat.
URL_PATTERN = re.compile(r"https?://[^\s,;]+")


def rebuild_agent(web_paths, files):
    """Reconstruye el agente RAG con las URLs y archivos actuales."""
    files = files or []
    with st.spinner("Cargando documentos y construyendo el agente..."):
        st.session_state.agent = build_agent(web_paths=web_paths, uploaded_files=files)

    # Guardamos las fuentes usadas para que la columna derecha siempre refleje el índice activo.
    st.session_state.web_paths = web_paths
    st.session_state.uploaded_documents = files

# Construye el agente una sola vez al abrir la aplicación.
if "agent" not in st.session_state:
    try:
        rebuild_agent(st.session_state.web_paths, st.session_state.uploaded_documents)
    except Exception as exc:
        st.error(f"No se pudo iniciar el RAG: {exc}")
        st.stop()

# Pantalla principal: chat a la izquierda y fuentes cargadas a la derecha.
left_col, right_col = st.columns([2, 1], vertical_alignment="top")

with left_col:
    # Muestra el historial de la conversación.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Campo de chat con clip para adjuntar documentos.
    chat_value = st.chat_input(
        "Pregunta, pega una URL o adjunta documentos...",
        accept_file="multiple",
        file_type=["txt", "md", "csv", "html", "htm", "pdf"],
    )

    if chat_value:
        # Streamlit devuelve un string si solo hay texto, o un objeto si hay archivos adjuntos.
        if isinstance(chat_value, str):
            pregunta = chat_value
            attached_files = []
        else:
            pregunta = chat_value.get("text", "")
            attached_files = chat_value.get("files", [])

        # Preparamos nuevas fuentes sin duplicar URLs ni nombres de archivo ya cargados.
        next_web_paths = list(st.session_state.web_paths)
        next_documents = list(st.session_state.uploaded_documents)
        added_sources = 0

        for url in URL_PATTERN.findall(pregunta):
            if url not in next_web_paths:
                next_web_paths.append(url)
                added_sources += 1

        for uploaded_file in attached_files:
            already_loaded = any(doc["name"] == uploaded_file.name for doc in next_documents)
            if not already_loaded:
                next_documents.append({"name": uploaded_file.name, "content": uploaded_file.getvalue()})
                added_sources += 1

        # Si el usuario pegó URLs o adjuntó archivos, se reconstruye el RAG con esas fuentes.
        if added_sources:
            try:
                rebuild_agent(next_web_paths, next_documents)
                st.toast(f"Fuentes añadidas: {added_sources}")
            except Exception as exc:
                st.error(f"No se pudieron añadir las fuentes: {exc}")
                st.stop()

        # Si solo se adjuntaron fuentes y no hay pregunta, refrescamos para mostrar la lista actualizada.
        if not pregunta.strip():
            st.rerun()

        # Añade la pregunta del usuario al historial visible.
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.write(pregunta)

        # Invoca el agente RAG y muestra la última respuesta generada.
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": pregunta}]}
                )
                respuesta_texto = respuesta["messages"][-1].content
            st.write(respuesta_texto)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})

with right_col:
    # Panel resumen con las URLs y archivos que forman parte del índice actual.
    st.markdown("### Fuentes cargadas")
    active_sources = [
        *st.session_state.web_paths,
        *[file["name"] for file in st.session_state.uploaded_documents],
    ]
    st.write(f"{len(active_sources)} fuente(s) en el índice actual.")
    for source in active_sources:
        st.code(source, language=None)
