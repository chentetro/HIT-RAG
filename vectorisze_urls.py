from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv

# טעינת מפתחות ה-API מהקובץ הסודי
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(working_dir, ".hf_cache")
os.makedirs(cache_dir, exist_ok=True)

# 1. הגדרת רשימת כתובות האתרים של HIT
urls = [
    "https://www.hit.ac.il/academic/sciences/bsc/"
   
]

# 2. טעינת התוכן מהאתרים
loader = WebBaseLoader(urls)
documents = loader.load()

# --- WebBaseLoader scraping preview (check if content is suitable) ---
preview_path = os.path.join(working_dir, "webbase_scrape_preview.txt")
with open(preview_path, "w", encoding="utf-8") as f:
    f.write("=== WebBaseLoader scraping preview ===\n\n")
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "?")
        f.write(f"--- Doc {i} | source: {source} | length: {len(doc.page_content)} chars ---\n\n")
        f.write(doc.page_content)
        f.write("\n\n")
print(f"Scraping preview: {len(documents)} doc(s) saved to {preview_path}")
for i, doc in enumerate(documents):
    print(f"  Doc {i}: {len(doc.page_content)} chars from {doc.metadata.get('source', '?')}")

# 3. פירוק הטקסט לנתחים קטנים (Chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# 4. הגדרת מודל ה-Embeddings (הפיכת טקסט למספרים)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=cache_dir
)

# 5. יצירת בסיס הנתונים הווקטורי ושמירתו בתיקייה מקומית
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./hit_db"
)

print(f"Done! Saved {len(text_chunks)} chunks to hit_db.")