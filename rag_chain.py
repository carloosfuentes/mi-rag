import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os

# Construir el agente
def build_agent():
    # Definiendo el modelo LLM
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

    # Cargar y dividir documentos
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever()

    # Definir herramienta de búsqueda
    @tool
    def buscador(query: str) -> str:
        """Busca información relevante en la base de datos de vectores."""
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs])

    return create_react_agent(model, tools=[buscador])