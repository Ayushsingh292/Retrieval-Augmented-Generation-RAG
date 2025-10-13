
import typer
import requests
import json
import time
from bs4 import BeautifulSoup
import chromadb
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer

app = typer.Typer(help="Mini RAG: Crawl → Index → Ask")


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("pages")


model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- 1. CODE FOR CRAWL ----
@app.command()
def crawl(
    start_url: str = typer.Argument(..., help="Starting URL to crawl"),
    max_pages: int = typer.Option(5, "--max-pages", help="Maximum number of pages to crawl")
):
    """Crawl up to N pages within the same domain and save clean text."""
    visited, to_visit = set(), [start_url]
    pages = []
    domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            res = requests.get(url, timeout=10, headers={"User-Agent": "MiniRAGBot/1.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            text = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
            if len(text.strip()) < 50:
                visited.add(url)
                continue
            pages.append({"url": url, "text": text})
            visited.add(url)

            
            for link in soup.find_all("a", href=True):
                new_url = urljoin(url, link["href"])
                if urlparse(new_url).netloc == domain and new_url not in visited:
                    to_visit.append(new_url)

        except Exception as e:
            print(json.dumps({"error": f"Error crawling {url}: {str(e)}"}))
            return

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "page_count": len(pages),
        "urls": [p["url"] for p in pages]
    }, indent=2))


# ---- 2. CODE FOR INDEX ----
@app.command()
def index():
    """Embed crawled pages using MiniLM and store them in Chroma."""
    try:
        data = json.load(open("data.json", encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"error": "data.json not found. Run crawl first."}))
        return

    
    try:
        chroma_client.delete_collection("pages")
    except:
        pass
    
    global collection
    collection = chroma_client.create_collection("pages")

    count = 0

    for d in data:
        try:
            emb = model.encode(d["text"]).tolist()
            
            collection.add(
                ids=[d["url"]],
                embeddings=[emb],
                documents=[d["text"]],  
                metadatas=[{"url": d["url"]}]
            )
            count += 1
        except Exception as e:
            print(json.dumps({"error": f"Embedding failed for {d['url']}: {str(e)}"}))
            return

    print(json.dumps({
        "vector_count": count,
        "status": "Index built successfully with MiniLM"
    }, indent=2))


# ---- 3. CODE FOR ASK ----
@app.command()
def ask(question: str, top_k: int = 3):
    """Retrieve top-k chunks and generate a grounded answer with citations via local Mistral.
    Includes retrieval, generation, and total latency timings (ms).
    """
    import time, json, requests

    # ---------- Start total timer ----------
    total_start = time.time()

    try:
        # ---------- Retrieval ----------
        retrieval_start = time.time()
        question_embedding = model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        retrieval_ms = int((time.time() - retrieval_start) * 1000)
    except Exception as e:
        print(json.dumps({"error": f"Query failed: {str(e)}"}))
        return

    # ---------- Validate results ----------
    if not results or not results.get("documents") or not results["documents"][0]:
        print(json.dumps({
            "answer": "No data indexed yet. Run index first.",
            "timings": {"retrieval_ms": 0, "llm_ms": 0, "total_ms": int((time.time() - total_start) * 1000)}
        }))
        return

    # ---------- Build context ----------
    context_docs = results["documents"][0]
    context = "\n\n".join(context_docs[:top_k])
    sources = [m["url"] for m in results["metadatas"][0] if "url" in m]

    prompt = f"""You are a helpful assistant.
Answer ONLY using the context below.
If the context does not contain enough information, say "not found in crawled content".

Context:
{context}

Question: {question}
Answer:"""

    # ---------- LLM Generation ----------
    try:
        llm_start = time.time()
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=120
        )
        llm_ms = int((time.time() - llm_start) * 1000)

        data = resp.json()
        answer = data.get("response", "").strip()
    except Exception as e:
        llm_ms = int((time.time() - llm_start) * 1000)
        print(json.dumps({
            "error": f"Ollama (Mistral) generation failed: {str(e)}",
            "timings": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": int((time.time() - total_start) * 1000)}
        }))
        return

    total_ms = int((time.time() - total_start) * 1000)

    # ---------- Output ----------
    print(json.dumps({
        "answer": answer,
        "sources": sources,
        "retrieval_count": len(context_docs),
        "timings": {
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms
        }
    }, indent=2))




if __name__ == "__main__":
    app()