from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("DB_DIR", "db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def load_docs():
    loaders = [
        DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader),
    ]

    docs = []
    for loader in loaders:
        try:
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"Warning: loader {loader} failed with error {e}")

    return docs

def main():
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        docs = load_docs()
        if not docs:
            print("No documents found in", DATA_DIR)
            sys.exit(0)
        
        total_size = 0
        for doc in docs:
            total_size += len(doc.page_content)
        
        if total_size > 100_000_000:
            print(f"Total document size ({total_size/1_000_000:.1f}MB) exceeds limit")
            sys.exit(1)

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        print("Creating embeddings")
        emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

        try:
            _ = emb.embed_query("warmup")
        except Exception as e:
            print(f"Warm-up failed: {e}")
            sys.exit(1)

        print("Saving to Chroma…")
        Chroma.from_documents(
            documents=chunks,           
            embedding=emb,                
            persist_directory=CHROMA_DIR, 
        )
        
        print(f"Saved {len(chunks)} chunks to {CHROMA_DIR}")
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        if os.path.exists(CHROMA_DIR):
            print(f"Cleaning up {CHROMA_DIR}")
            import shutil
            shutil.rmtree(CHROMA_DIR)
        sys.exit(1)

if __name__ == "__main__":
    main()