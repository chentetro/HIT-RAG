# HIT Assistant — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about HIT (Holon Institute of Technology) using content scraped from the HIT website.

Built with **LangChain**, **ChromaDB**, **Groq (LLaMA 3.3)**, and **Streamlit**.

## Project Structure

```
RAG/
├── main.py                 # Streamlit chat app (the UI)
├── vectorisze_urls.py      # Scrapes HIT pages → chunks → stores in ChromaDB
├── config.json             # API keys (not committed — see setup below)
├── requirements.txt        # Python dependencies
├── hit_db/                 # ChromaDB vector store (auto-generated)
└── webbase_scrape_preview.txt  # Debug preview of scraped content
```

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.\venv\Scripts\activate.bat`
- **macOS / Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `config.json` file in the project root:

```json
{
  "GROQ_API_KEY": "your-groq-api-key-here"
}
```

> This file is listed in `.gitignore` and will not be committed.

## Running the Project

There are **two steps**: first build the vector database, then launch the chat UI.

### Step 1 — Scrape & vectorize (run once, or whenever you update URLs)

```bash
python vectorisze_urls.py
```

This will:
- Fetch the HIT web pages listed in the `urls` array
- Clean the scraped text (remove navigation, forms, duplicates)
- Split the text into chunks and embed them
- Save everything to the `hit_db/` ChromaDB directory
- Write a `webbase_scrape_preview.txt` for you to inspect the scraped content

### Step 2 — Launch the chatbot

```bash
streamlit run main.py
```

The app will open at [http://localhost:8501](http://localhost:8501). Ask any question about HIT and the assistant will answer based on the scraped content.

## Adding More Pages

Edit the `urls` list in `vectorisze_urls.py` to include additional HIT pages:

```python
urls = [
    "https://www.hit.ac.il/academic/sciences/bsc/",
    "https://www.hit.ac.il/academic/sciences/msc/",
    # add more URLs here
]
```

Then re-run `python vectorisze_urls.py` to rebuild the vector database.
