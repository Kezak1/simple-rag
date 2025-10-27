import os
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3"
DB_DIR = "db"

st.set_page_config(page_title="simple-rag assistant")
st.title(f"{LLM_MODEL.capitalize()}")

st.sidebar.header("parameters")
top_k = st.sidebar.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=4, step=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
use_sources = st.sidebar.checkbox("Give source", value=True)

def source(d):
    src = d.metadata.get("source") or d.metadata.get("file_path") or d.metadata.get("path") or "not found source"
    page = d.metadata.get("page")
    if page is not None:
        return f"{src} (page {page})"
    return str(src)

def format_docs_for_prompt(docs, give_source=True, max_chars=1200):
    chunks = []
    used = 0
    for d in docs:
        text = (d.page_content or "").strip()
        snippet = text[:400].replace("\n", " ")

        if give_source:
            lab = source(d)
            candidate = f"- {snippet}\n[source: {lab}]\n"
        else:
            candidate = f"- {snippet}\n"

        if used + len(candidate) > max_chars:
            break
        chunks.append(candidate)
        used += len(candidate)

    return "\n".join(chunks) if chunks else "(no retrieved context)"


def history_to_text(chat_history):
    lines = []
    for m in chat_history:
        if isinstance(m, HumanMessage):
            role = "Human"
        else:
            role = "Assistant"

        lines.append(f"{role}: {m.content}")

    if lines:
        return "\n".join(lines)
    return "(empty)"


def load_vectorstore():
    if not os.path.isdir(DB_DIR):
        st.warning(f"Chroma directory not found at: {DB_DIR}. Run your process_data.py first")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def load_llm(temp: float):
    return ChatOllama(model=LLM_MODEL, temperature=temp)


PROMPT = ChatPromptTemplate.from_template(
"""You are a helpful assistant. Use the provided context when relevant.
If the answer isn't in the context, say you aren't sure!

Chat history:
{chat_history}

Context:
{context}

User question:
{question}
"""
)

def query(user_query, chat_history, vectorstore, temp, k, give_source):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(user_query)

    context_text = format_docs_for_prompt(docs, give_source=give_source)
    llm = load_llm(temp)
    
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "chat_history": history_to_text(chat_history),
        "context": context_text,
        "question": user_query
    })
    return answer, docs

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I am simple RAG. What do you want?")
    ]


vectorstore = load_vectorstore()

for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)

user_query = st.chat_input("Type your message...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            answer, docs = query(
                user_query=user_query,
                chat_history=st.session_state.chat_history,
                vectorstore=vectorstore,
                temp=temperature,
                k=top_k,
                give_source=use_sources,
            )
        st.markdown(answer)

        if docs and use_sources:
            with st.expander("Sources"):
                for d in docs:
                    st.write(f"• {source(d)}")

    st.session_state.chat_history.append(AIMessage(content=answer))