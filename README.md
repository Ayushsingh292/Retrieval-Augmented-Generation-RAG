# Mini RAG Web — Website-Grounded Q&A using Mistral + Chroma

A lightweight Retrieval-Augmented Generation (RAG) system that crawls any website, builds embeddings, and answers questions grounded strictly in crawled content.  
Built with Flask, SentenceTransformers, ChromaDB, and local Mistral (Ollama).

---

## Setup & Run

1. Clone & install dependencies
   ```bash
   git clone <repo_url>
   cd rag-min2
   pip install -r requirements.txt
   ```

2. Start Ollama with Mistral
   ```bash
   ollama run mistral
   ```

3. Run the Flask web interface
   ```bash
   cd web
   python app.py
   ```
   Opens at http://127.0.0.1:5000

4. (Optional) Run from CLI
   ```bash
   python main.py crawl https://www.python.org/psf/ --max-pages 2
   python main.py index
   python main.py ask "What is the Python Software Foundation?"
   ```

---

## Architecture Overview

- Crawl Stage → Recursively collects text from a target website (via `requests` + `BeautifulSoup`) within the same domain.  
- Index Stage →  
  - Encodes text into embeddings using `all-MiniLM-L6-v2` from SentenceTransformers.  
  - Stores vectors persistently using ChromaDB (`PersistentClient(path="./chroma_db")`).  
- Ask Stage →  
  - Embeds the question and retrieves top-k relevant passages.  
  - Sends context + query to local Mistral (Ollama) for grounded answer generation.  
- Frontend (Flask) → Minimal 3-step interface: Crawl → Index → Ask.

---

## Evaluation & Behavior

- Produces factually grounded answers with explicit sources.  
- Retrieval accuracy verified through cosine similarity in embedding space.  
- Works fully offline using local models — no API key or network required.  
- Validated across sites such as Python.org, NASA.gov, Wikipedia, and OpenAI.com.

---

## Design Trade-offs

- MiniLM chosen for compact size (~100 MB) and strong semantic recall.  
- Persistent ChromaDB ensures local reusability and fast lookups.  
- No fine-grained chunking keeps pipeline simple but slightly lowers recall on long pages.  
- Mistral via Ollama avoids API costs but may be slower than cloud LLMs.  
- Requests + BeautifulSoup only captures static HTML (no JS rendering).

---

## Limitations

- Fails on dynamic/JavaScript-heavy sites (React, Next.js, etc.).  
- No deduplication or crawl scheduling; may re-index similar text.  
- Embedding model dimension (384 d) limits vector expressiveness.  
- Single-user, sequential pipeline (demo-scale, not production).

---

## Attribution

Developed by Ayush Singh (Manipal Institute of Technology Bengaluru).  
Inspired by open-source examples from  
[ChromaDB Docs](https://docs.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/examples/).  
Local text generation powered by [Ollama](https://ollama.ai) using the Mistral model.

---

## Example Demo Queries

| Website | Example Questions |
|----------|-------------------|
| Python.org | What is the Python Software Foundation?<br>Who created Python?<br>What are Python’s main features? |
| NASA.gov | What is NASA’s mission?<br>When was NASA established?<br>What are NASA’s current space programs? |
| Wikipedia – AI | What is Artificial Intelligence?<br>Who is the father of AI?<br>What are the goals of AI research? |
| Mozilla.org | What is Mozilla’s mission?<br>Who founded Mozilla?<br>What products are built by Mozilla? |

---

**License:** MIT License  
(c) 2025 Ayush Singh. All rights reserved.
