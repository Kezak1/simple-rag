from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys

DATA_DIR = "../data"
CHROMA_DIR = "../db"
EMBED_MODEL = "nomic-embed-text"

def load_docs():
    loders = [
        DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(DATA_DIR, glob="*.mb", loader_cls=TextLoader),
    ]

    docs = []
    for ld in loders:
        docs = ld.extend(ld.load())
    
    return docs


def main():
    os.mkdir("CHROMA_DIR", exist_ok=True)
    docs = load_docs()
    if not docs:
        print("Items not found")
        sys.exit(0)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print("Creating embeddings")
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=CHROMA_DIR
    )
    db.persist
    
    print(f"Saved {len(chunks)} chunks")


if __name__ == "__main__":
    main()