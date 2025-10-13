import os, time, orjson, re, statistics, typer
import faiss
from sentence_transformers import SentenceTransformer
from utils.io_utils import ensure_dirs, read_jsonl
from utils.model_utils import ensure_ollama_model

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
CHUNK_META_PATH = os.path.join(DATA_DIR, "chunk_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(model: SentenceTransformer, texts):
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def ask(
    question: str = typer.Option(..., help="Your question"),
    top_k: int = typer.Option(5, help="Top-k chunks to retrieve"),
    min_support: float = typer.Option(0.25, help="Minimum top similarity to avoid refusal (0..1)"),
    model_name: str = typer.Option("mistral", help="Ollama model name (e.g., mistral, llama3)"),
):
    """
    Retrieve top-k chunks and use a local Ollama model for phrased answers.
    """
    import ollama

    t0 = time.time()
    ensure_dirs()

    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(CHUNK_META_PATH)):
        typer.echo(orjson.dumps({"error": "Index artifacts missing. Run index first."}).decode())
        raise typer.Exit(1)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_vec = embed_texts(model, [question])

    index = faiss.read_index(FAISS_INDEX_PATH)
    D, I = index.search(q_vec, top_k)
    sims = D[0].tolist()
    idxs = I[0].tolist()

    chunks = read_jsonl(CHUNKS_PATH)
    results = []
    for rank, (sim, idx) in enumerate(zip(sims, idxs), start=1):
        if 0 <= idx < len(chunks):
            ch = chunks[idx]
            snippet = ch["text"].strip()
            url = ch["url"]
            results.append({"rank": rank, "score": float(sim), "url": url, "snippet": snippet[:400]})

    retrieval_ms = int((time.time() - t0) * 1000)
    top_score = sims[0] if sims else 0.0

    if not results or top_score < min_support:
        out = {
            "answer": "Not enough information in the crawled content to answer this question.",
            "refused": True,
            "sources": [{"url": r["url"], "snippet": r["snippet"], "score": r["score"]} for r in results],
            "timings": {"retrieval_ms": retrieval_ms, "total_ms": int((time.time() - t0) * 1000)},
        }
        typer.echo(orjson.dumps(out).decode())
        return

    context_text = "\n\n".join([f"Source {i+1} ({r['url']}): {r['snippet']}" for i, r in enumerate(results)])
    prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the context does not contain enough information, respond with "Not enough information."

Question: {question}

Context:
{context_text}

Answer clearly and concisely using only the information above:"""

    try:
        ensure_ollama_model(model_name)
        llm_start = time.time()
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        answer_text = response["message"]["content"].strip()
        llm_ms = int((time.time() - llm_start) * 1000)
    except Exception as e:
        answer_text = f"Ollama generation failed: {e}"
        llm_ms = 0

    total_ms = int((time.time() - t0) * 1000)
    out = {
        "answer": answer_text,
        "refused": False,
        "sources": [{"url": r["url"], "score": r["score"]} for r in results],
        "timings": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": total_ms},
    }
    typer.echo(orjson.dumps(out).decode())

def eval(
    file: str = typer.Option(..., help="Path to a text file with one question per line"),
    top_k: int = typer.Option(5),
    model_name: str = typer.Option("mistral"),
    min_support: float = typer.Option(0.25),
):
    """
    Run multiple questions and report p50/p95 latencies, refusal rate, and avg answer length.
    """
    import ollama
    ensure_dirs()
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        typer.echo(orjson.dumps({"error": "Index missing. Run crawl/index first."}).decode())
        raise typer.Exit(1)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(FAISS_INDEX_PATH)
    questions = [q.strip() for q in open(file, "r", encoding="utf-8") if q.strip()]

    retrieval_times, llm_times, total_times = [], [], []
    refusals, answers_len = 0, 0
    results = []

    try:
        ensure_ollama_model(model_name)
    except Exception:
        pass

    for q in questions:
        t0 = time.time()
        q_vec = embed_texts(model, [q])
        D, I = index.search(q_vec, top_k)
        sims, idxs = D[0].tolist(), I[0].tolist()
        chunks = read_jsonl(CHUNKS_PATH)

        top_score = sims[0] if sims else 0.0
        retrieval_ms = int((time.time() - t0) * 1000)
        retrieval_times.append(retrieval_ms)

        refused = False
        answer_text = ""
        llm_ms = 0
        srcs = []

        if not sims or top_score < min_support:
            refused = True
            answer_text = "Not enough information in the crawled content to answer this question."
        else:
            ctx = []
            for rank, (sim, idx) in enumerate(zip(sims, idxs), start=1):
                if 0 <= idx < len(chunks):
                    ch = chunks[idx]
                    srcs.append({"url": ch["url"], "score": float(sim)})
                    ctx.append(f"Source {rank} ({ch['url']}): {ch['text'][:400]}")
            prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer.
If the context lacks the answer, reply exactly: Not enough information.

Question: {q}

Context:
{chr(10).join(ctx)}

Answer:"""
            llm_start = time.time()
            try:
                resp = ollama.chat(model=model_name, messages=[{"role":"user","content":prompt}])
                answer_text = resp["message"]["content"].strip()
            except Exception as e:
                answer_text = f"Ollama generation failed: {e}"
            llm_ms = int((time.time() - llm_start) * 1000)

        total_ms = retrieval_ms + llm_ms
        total_times.append(total_ms)
        llm_times.append(llm_ms)
        refusals += 1 if refused or answer_text.lower().startswith("not enough information") else 0
        answers_len += len(answer_text)

        results.append({"question": q, "answer": answer_text, "refused": refused, "sources": srcs,
                        "timings": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": total_ms}})

    def p50(xs): return int(statistics.median(xs)) if xs else 0
    def p95(xs):
        if not xs: return 0
        xs_sorted = sorted(xs)
        idx = min(len(xs_sorted)-1, int(round(0.95*(len(xs_sorted)-1))))
        return int(xs_sorted[idx])

    report = {
        "n": len(questions),
        "refusal_rate": round(refusals / max(1, len(questions)), 3),
        "answer_len_avg": int(answers_len / max(1, len(questions))),
        "latency_ms": {
            "retrieval_p50": p50(retrieval_times), "retrieval_p95": p95(retrieval_times),
            "llm_p50": p50(llm_times),             "llm_p95": p95(llm_times),
            "total_p50": p50(total_times),         "total_p95": p95(total_times),
        },
        "samples": results[:3]
    }
    typer.echo(orjson.dumps(report).decode())
