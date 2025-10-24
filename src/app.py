import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
DB_DIR = ""




