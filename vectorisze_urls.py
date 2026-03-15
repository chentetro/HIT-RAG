import os
import re

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(working_dir, ".hf_cache")
os.makedirs(cache_dir, exist_ok=True)

# ---------- 1. URLs to scrape ----------
urls = [
    "https://www.hit.ac.il/academic/sciences/bsc/",
]

# ---------- 2. Load pages with WebBaseLoader (no browser needed) ----------
# SoupStrainer targets only the <main> content area, skipping nav/footer/forms
content_filter = SoupStrainer("main")

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs={"parse_only": content_filter},
    encoding="utf-8",
)
documents = loader.load()


def clean_page_text(text: str) -> str:
    """Strip navigation noise, form junk, duplicate paragraphs, and excess whitespace."""
    # Remove known UI / layout artifacts
    for noise in [
        "header mobile",
        "דלג לתוכן המרכזי",
        "דלג על איזור זה",
        ".accordion-answer",
    ]:
        text = text.replace(noise, "")

    # Cut everything from the contact-form section onward
    text = re.sub(r"חושבים ללמוד ב-HIT\?.*$", "", text, flags=re.DOTALL)

    # Deduplicate identical paragraphs (keeps first occurrence)
    lines = text.split("\n")
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            unique.append(line)
            continue
        if stripped not in seen:
            seen.add(stripped)
            unique.append(line)
    text = "\n".join(unique)

    # Collapse runs of blank lines / spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


for doc in documents:
    doc.page_content = clean_page_text(doc.page_content)

# ---------- Preview (for debugging) ----------
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

# ---------- 3. Split into chunks ----------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# ---------- 4. Embeddings ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    cache_folder=cache_dir,
)

# ---------- 5. Persist to ChromaDB ----------
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./hit_db",
)

print(f"Done! Saved {len(text_chunks)} chunks to hit_db.")