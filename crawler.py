import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os, time, queue, requests, urllib.robotparser, orjson
from bs4 import BeautifulSoup
import trafilatura
import tldextract
import typer
from utils.io_utils import ensure_dirs, write_jsonl
from utils.text_utils import normalize_url, get_links, same_registrable_domain

DATA_DIR = "data"
PAGES_PATH = os.path.join(DATA_DIR, "pages.jsonl")

def polite_sleep(ms: int):
    time.sleep(max(ms, 0) / 1000.0)

def simple_robot_ok(target_url: str) -> bool:
    try:
        parts = requests.utils.urlparse(target_url)
        robots = f"{parts.scheme}://{parts.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots)
        rp.read()
        return rp.can_fetch("*", target_url)
    except Exception:
        return True

def fetch(url: str, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "mini-rag-bot"})
        if "text/html" in r.headers.get("Content-Type",""):
            return r.status_code, r.url, r.text
        return r.status_code, r.url, ""
    except Exception:
        return 0, url, ""

def extract_main_text(html: str, url: str):
    downloaded = trafilatura.extract(html, url=url, include_tables=False, favor_recall=True)
    if downloaded:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else url)
        return title, downloaded.strip()
    return "", ""

def crawl(
    start_url: str = typer.Option(..., help="Starting URL"),
    max_pages: int = typer.Option(40, help="Hard cap on pages to crawl"),
    max_depth: int = typer.Option(2, help="BFS depth limit"),
    crawl_delay_ms: int = typer.Option(500, help="Politeness delay (ms)"),
):
    """
    Crawl within the registrable domain, respect robots.txt, extract main text, and save to data/pages.jsonl.
    """
    t0 = time.time()
    ensure_dirs()

    visited, urls_out = set(), []
    q = queue.Queue()
    q.put((start_url, 0))

    page_count = 0
    skipped = 0

    if os.path.exists(PAGES_PATH):
        os.remove(PAGES_PATH)

    while not q.empty() and page_count < max_pages:
        url, depth = q.get()
        if url in visited:
            continue
        visited.add(url)

        if not same_registrable_domain(start_url, url):
            skipped += 1
            continue

        if not simple_robot_ok(url):
            skipped += 1
            continue

        status, final_url, html = fetch(url)
        if status != 200 or not html:
            skipped += 1
            continue

        title, text = extract_main_text(html, final_url)
        if not text:
            skipped += 1
            continue

        write_jsonl(PAGES_PATH, {"url": final_url, "title": title, "text": text})
        urls_out.append(final_url)
        page_count += 1

        if depth < max_depth:
            for nxt in get_links(html, final_url):
                if nxt not in visited and same_registrable_domain(start_url, nxt):
                    q.put((nxt, depth + 1))

        polite_sleep(crawl_delay_ms)

    out = {
        "page_count": page_count,
        "skipped_count": skipped,
        "urls": urls_out,
        "timings": {"crawl_ms": int((time.time() - t0) * 1000)},
    }
    typer.echo(orjson.dumps(out).decode())
