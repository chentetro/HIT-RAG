import json
import os

import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. הגדרת נתיב העבודה - מוודא שהקוד ימצא את הקבצים שלו
working_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(working_dir, ".hf_cache")
os.makedirs(cache_dir, exist_ok=True)

# 2. טעינת המפתחות מקובץ ה-JSON (ודאי שיש לך קובץ config.json בתיקייה)
with open(f"{working_dir}/config.json", encoding="utf-8") as config_file:
    config_data = json.load(config_file)
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def setup_vectorstore():
    """
    פונקציה שטוענת את בסיס הנתונים הווקטורי מהזיכרון המקומי
    """
    # הנתיב לתיקייה שבה שמרנו את המידע של HIT
    persist_directory = f"{working_dir}/hit_db"
    
    # הגדרת מודל ה-Embeddings (זהה לזה שהשתמשנו ב-Ingestion)
    # תיקנתי כאן את טעות הכתיב מהצילום (embeddings עם 2 d)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder=cache_dir
    )
    
    # חיבור ל-ChromaDB הקיים
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

def format_chat_history(chat_history):
    recent_messages = chat_history[-6:]
    history_lines = []

    for message in recent_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {message['content']}")

    return "\n".join(history_lines)


def ask_groq(question, vectorstore, chat_history):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    documents = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in documents)
    history_text = format_chat_history(chat_history)

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful HIT assistant. "
                    "Answer only from the provided context. "
                    "If the answer is not in the context, say you do not know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chat history:\n{history_text or 'No previous messages.'}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content or "No answer was returned."
    sources = [
        doc.metadata.get("source", "Unknown source")
        for doc in documents
        if doc.metadata.get("source")
    ]

    return answer, sources

# הגדרות תצורת הדף של Streamlit
st.set_page_config(
    page_title="Multi Doc Chat",
    page_icon="📚",
    layout="centered"
)

# כותרת ראשית לאפליקציה
st.title("📚 HIT Assistant")

# אתחול היסטוריית הצ'אט בזיכרון של הדפדפן (Session State)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# אתחול בסיס הנתונים הווקטורי - רץ רק פעם אחת
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

# Display chat history from session state
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_input = st.chat_input("Ask AI...")

if user_input:
    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_response, sources = ask_groq(
                user_input,
                st.session_state.vectorstore,
                st.session_state.chat_history[:-1],
            )

        st.markdown(assistant_response)
        if sources:
            unique_sources = list(dict.fromkeys(sources))
            st.caption("Sources: " + ", ".join(unique_sources))

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
