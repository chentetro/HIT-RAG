import os
import hashlib
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from bs4 import BeautifulSoup


# -----------------------------
# Configuration
# -----------------------------

# Replace these URLs with the real HIT College URLs.
HIT_URLS: Dict[str, str] = {
    # "https://www.hit.ac.il/students/regulations-newsletters/": "regulations",
    # "https://academic.hit.ac.il/en/rnd/Student_Exchange/Course_Catalogue/Computer_Science": "syllabus",
}

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_hit_content(text: str) -> str:
    """Lightly clean HIT HTML text before chunking and deduplication.

    This keeps the logic centralized so that duplicate detection always runs
    on normalized content.
    """

    if not text:
        return ""

    # Basic normalization – strip leading/trailing whitespace
    cleaned = text.strip()
    # Collapse very long runs of whitespace into single spaces
    cleaned = " ".join(cleaned.split())
    return cleaned


def load_and_chunk_documents(url_category_map: Dict[str, str]) -> List[Document]:
    """Load HTML content from multiple URLs and assign a `category` metadata field.

    For the regulations newsletters page we use RecursiveUrlLoader so we also capture
    linked newsletter pages, while keeping the crawl bounded and restricted to the
    HIT domain. All other URLs are loaded with WebBaseLoader.
    """

    all_docs: List[Document] = []
    seen_hashes: set[str] = set()

    for url, category in url_category_map.items():
        # Special case: recursively load linked newsletters under the regulations page
        if url == "https://www.hit.ac.il/students/regulations-newsletters/":
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=2,
                prevent_outside=True,
                extractor=lambda html: BeautifulSoup(html, "html.parser").get_text(),
            )
            docs = loader.load()
        else:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs={"features": "html.parser"},
            )
            docs = loader.load()

        for d in docs:
            # Clean content first so deduplication compares normalized text
            d.page_content = clean_hit_content(d.page_content)

            # MD5 hash-based deduplication on cleaned content
            content_bytes = d.page_content.encode("utf-8", errors="ignore")
            content_hash = hashlib.md5(content_bytes).hexdigest()
            if content_hash in seen_hashes:
                source = d.metadata.get("source", url) if d.metadata else url
                print(f"Skipping duplicate content from {source}")
                continue

            seen_hashes.add(content_hash)

            d.metadata = d.metadata or {}
            d.metadata["source"] = url
            d.metadata["category"] = category
            all_docs.append(d)

    if not all_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    split_docs = splitter.split_documents(all_docs)
    return split_docs


def build_vectorstore(url_category_map: Dict[str, str]) -> Chroma:
    """Create or load a ChromaDB vector store populated with categorized documents."""

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

    docs = load_and_chunk_documents(url_category_map)
    if not docs:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


vectorstore: Chroma | None = None


@tool
def search_academic_regulations(query: str) -> str:
    """Search HIT academic regulations and policies. Use for questions about rules, grading, attendance, and exams."""

    if vectorstore is None:
        return "Vector store is not initialized."

    docs = vectorstore.similarity_search(
        query, k=4, filter={"category": "regulations"}
    )
    if not docs:
        return "No relevant regulations were found."

    parts = []
    for d in docs:
        parts.append(
            f"Source: {d.metadata.get('source', 'unknown')}\n"
            f"Category: {d.metadata.get('category', '')}\n"
            f"Content:\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def get_course_info(query: str) -> str:
    """Search HIT course syllabi and descriptions. Use for questions about courses, prerequisites, and curriculum."""

    if vectorstore is None:
        return "Vector store is not initialized."

    docs = vectorstore.similarity_search(
        query, k=4, filter={"category": "syllabus"}
    )
    if not docs:
        return "No relevant course information was found."

    parts = []
    for d in docs:
        parts.append(
            f"Source: {d.metadata.get('source', 'unknown')}\n"
            f"Category: {d.metadata.get('category', '')}\n"
            f"Content:\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def create_rag_agent() -> AgentExecutor:
    """Create a LangChain 0.3 tool-calling agent backed by ChatGroq."""

    tools = [search_academic_regulations, get_course_info]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful academic assistant for HIT College.\n"
                    "You have access to tools that let you search:\n"
                    " - academic regulations and policies (category: 'regulations')\n"
                    " - course syllabi and descriptions (category: 'syllabus')\n\n"
                    "When a user question involves rules, grading, or exams, prefer "
                    "the regulations tool. When it involves course content, prerequisites "
                    "or curriculum, prefer the syllabus tool. If the question touches both "
                    "areas (for example, 'I got a 60 in Data Science, did I pass?'), call "
                    "both tools as needed and combine the information in your answer.\n\n"
                    "Always show your final answer clearly and concisely for students."
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True)


@st.cache_resource
def initialize_app() -> AgentExecutor:
    """Initialize vector store and agent once per Streamlit session."""

    global vectorstore
    load_dotenv()  # to read GROQ_API_KEY from .env if present

    if not HIT_URLS:
        st.warning(
            "No HIT URLs configured. Please edit `HIT_URLS` in `app.py` "
            "to point to the correct regulations and syllabus pages."
        )

    vectorstore = build_vectorstore(HIT_URLS)
    return create_rag_agent()


def main() -> None:
    st.set_page_config(page_title="HIT RAG Assistant", page_icon="📚")

    st.title("HIT College RAG Assistant")
    st.caption(
        "Ask questions about academic regulations and course information. "
        "Powered by LangChain 0.3, ChromaDB, and ChatGroq."
    )

    agent_executor = initialize_app()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about regulations or course information...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

        # We pass the whole chat history to allow the agent to reason over context.
        history_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history_messages.append(("human", msg["content"]))
            else:
                history_messages.append(("ai", msg["content"]))

        result = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": history_messages,
            }
        )

        answer = result.get("output", "")
        response_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

