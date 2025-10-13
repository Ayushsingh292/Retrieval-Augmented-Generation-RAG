import os, hashlib, orjson, time
import numpy as np
import faiss
import typer
from sentence_transformers import SentenceTransformer

from utils.io_utils import ensure_dirs, read_jsonl, write_jsonl
from utils.text_utils import chunk_text

DATA_DIR = "data"
PAGES_PATH = os.path.join(DATA_DIR, "pages.jsonl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
CHUNK_META_PATH = os.path.join(DATA_DIR, "chunk_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(model: SentenceTransformer, texts):
    vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def index(
    chunk_size: int = typer.Option(600, help="Chunk size (chars)"),
    chunk_overlap: int = typer.Option(100, help="Chunk overlap (chars)"),
    embedding_model: str = typer.Option(EMBED_MODEL_NAME, help="Sentence-Transformers model"),
):
    """
    Chunk pages, embed, and build a FAISS index. Saves to data/*.
    """
    t0 = time.time()
    ensure_dirs()

    pages = read_jsonl(PAGES_PATH)
    if not pages:
        typer.echo(orjson.dumps({"error": "No pages found. Run crawl first."}).decode())
        raise typer.Exit(1)

    for p in [CHUNKS_PATH, CHUNK_META_PATH, FAISS_INDEX_PATH]:
        if os.path.exists(p):
            os.remove(p)

    errors = []
    total_chunks = 0
    for p in pages:
        try:
            chunks = chunk_text(p["text"], chunk_size=chunk_size, overlap=chunk_overlap)
            for (i, j, ctext) in chunks:
                cid = hashlib.md5(f'{p["url"]}:{i}:{j}'.encode()).hexdigest()
                write_jsonl(CHUNK_META_PATH, {"chunk_id": cid, "url": p["url"], "start": i, "end": j})
                write_jsonl(CHUNKS_PATH, {"url": p["url"], "chunk_id": cid, "text": ctext})
            total_chunks += len(chunks)
        except Exception as e:
            errors.append(str(e))

    model = SentenceTransformer(embedding_model)
    chunk_rows = read_jsonl(CHUNKS_PATH)
    texts = [r["text"] for r in chunk_rows]

    vecs = embed_texts(model, texts)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    faiss.write_index(index, FAISS_INDEX_PATH)

    out = {
        "vector_count": int(index.ntotal),
        "chunk_count": total_chunks,
        "errors": errors,
        "timings": {"index_ms": int((time.time() - t0) * 1000)},
    }
    typer.echo(orjson.dumps(out).decode())
