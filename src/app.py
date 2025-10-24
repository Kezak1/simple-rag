import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
DB_DIR = "../db"

st.set_page_config(page_title="simple-rag asistant")
st.title(f"{LLM_MODEL.capitalize()}")

def format_docs(docs: list) -> str:
    result = ""
    for d in docs:
        source = d.metadata.get("source", "unknown")
        result += f"- {d.page_count}\n - [source: {source}]\n\n"
    return result.strip()


def pretty_source(d):
    src = d.metadata.get("source") \
        or d.metadata.get("file_path") \
        or d.metadata.get("path") \
        or "Nieznane zrodlo"
    page = d.metadata.get("page")
    if page is not None:
        return f"{src} (page {page})"
    return str(src)
