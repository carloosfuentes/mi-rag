import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_agent

# Cargando las variables de entorno
load_dotenv()

st.title("RAG con LangChain")
st.caption("Pregunta sobre el artículo de Lilian Weng sobre agentes")

# Construir el agente una sola vez
if "agent" not in st.session_state:
    with st.spinner("Cargando documentos y construyendo el agente..."):
        st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input del usuario y respuesta del agente
if pregunta := st.chat_input("¿Qué quieres saber?"):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": pregunta}]}
            )
            respuesta_texto = respuesta["messages"][-1].content
        st.write(respuesta_texto)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})