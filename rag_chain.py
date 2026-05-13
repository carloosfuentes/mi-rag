import bs4
from io import BytesIO
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os

# URL inicial que se indexa si el usuario no adjunta nada desde la interfaz.
DEFAULT_WEB_PATHS = ("https://lilianweng.github.io/posts/2024-07-07-hallucination/",)

# Función para cargar nuevos archivos dentro del RAG
def _extract_uploaded_document(uploaded_file) -> Document | None:
    """Convierte un archivo subido desde Streamlit en un documento indexable."""
    # La app guarda los archivos como diccionarios para conservarlos entre reruns.
    file_name = uploaded_file["name"] if isinstance(uploaded_file, dict) else uploaded_file.name
    raw_content = uploaded_file["content"] if isinstance(uploaded_file, dict) else uploaded_file.getvalue()
    extension = file_name.rsplit(".", 1)[-1].lower()

    # Cada tipo de archivo necesita una forma distinta de extraer texto limpio.
    if extension == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Instala pypdf para poder cargar archivos PDF.") from exc

        reader = PdfReader(BytesIO(raw_content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif extension in {"html", "htm"}:
        # En HTML nos quedamos solo con el texto visible.
        soup = bs4.BeautifulSoup(raw_content, "html.parser")
        text = soup.get_text("\n", strip=True)
    else:
        # Para txt, md o csv basta con decodificar el contenido como texto.
        text = raw_content.decode("utf-8", errors="ignore")

    # Si el archivo no aporta texto, no se añade al índice.
    if not text.strip():
        return None

    return Document(page_content=text, metadata={"source": file_name})


def _load_web_documents(web_paths: tuple[str, ...]):
    """Carga documentos web y reduce el HTML a las zonas principales del artículo."""
    # Este filtro funciona bien con los artículos del blog de Lilian Weng.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    if any(doc.page_content.strip() for doc in docs):
        return docs

    # Si una URL no usa esas clases CSS, cargamos la web completa como respaldo.
    return WebBaseLoader(web_paths=web_paths).load()


def build_agent(web_paths=None, uploaded_files=None):
    """Construye todo el flujo RAG: carga, troceado, vectorstore, retriever y agente."""
    # Modelo conversacional que responderá usando el contexto recuperado.
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

    # Si no se pasan URLs, se usa la URL por defecto para que la app arranque con contenido.
    web_paths = tuple(path.strip() for path in (web_paths or DEFAULT_WEB_PATHS) if path.strip())
    uploaded_files = uploaded_files or []

    # Carga todas las fuentes que formarán parte del RAG.
    docs = _load_web_documents(web_paths) if web_paths else []
    for uploaded_file in uploaded_files:
        document = _extract_uploaded_document(uploaded_file)
        if document is not None:
            docs.append(document)

    if not docs:
        raise ValueError("No hay documentos con contenido para construir el RAG.")

    # Divide documentos largos en fragmentos manejables para el buscador semántico.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # Convierte cada fragmento en vectores y los guarda en FAISS.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever()

    # Herramienta que el agente puede llamar para buscar contexto en el índice.
    @tool
    def buscador(query: str) -> str:
        """Busca información relevante en la base de datos de vectores."""
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs])

    # ReAct agent: decide cuándo usar el buscador y cuándo redactar la respuesta final.
    return create_react_agent(model, tools=[buscador])
