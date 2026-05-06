# mi-rag

Implementación sencilla de un sistema RAG (Retrieval-Augmented Generation) desplegado mediante Docker. La aplicación se ejecuta en local usando Streamlit.

## Descripción

Este proyecto implementa un sistema de generación de respuestas basado en recuperación de información (RAG). Combina la búsqueda de contexto relevante con un modelo de lenguaje para generar respuestas más precisas.

La aplicación está contenida en un entorno Docker y se expone a través de una interfaz web construida con Streamlit.

## Características

- Sistema RAG básico con LangChain
- Interfaz web con Streamlit
- Despliegue sencillo con Docker
- Entorno reproducible mediante Docker Compose

## Tecnologías

- Python
- Streamlit
- LangChain
- Docker
- Docker Compose
- OpenRouter (API LLM)

## Estructura del proyecto

```text
mi-rag/
├── app.py
├── rag_chain.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Configuración

Antes de levantar el proyecto, crea un archivo `.env` en la raíz del repositorio con tu API Key de OpenRouter:

## Instalación y despliegue

1. Clonar el repositorio:

```bash
git clone https://github.com/carloosfuentes/mi-rag.git
cd mi-rag
docker-compose up --build
```

2. Una vez desplegado, la aplicación está disponible en:

```
http://localhost:8501
```

## Notas
- Es necesario tener Docker y Docker Compose instalados
- El sistema está preparado para ejecutarse en entorno local
- La configuración puede modificarse desde el Dockerfile o docker-compose.yml

# Observaciones
En caso de que se quiera publicar este RAG, podremos usar Streamlit Cloud, el cual es un servicio gratuito el cual es capaz de publicar el proyecto desde un repositorio público de GitHub.

A la hora de publicar el RAG, deberemos de compartir nuestra clave API con Streamlit (en mi caso es de OpenRouter). De lo contrario, no se podrá utilizar ningún LLM, lo que provocará que el RAG no funcione.