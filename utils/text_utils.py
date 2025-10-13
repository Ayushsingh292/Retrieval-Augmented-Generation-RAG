import re
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
import tldextract

def same_registrable_domain(a: str, b: str) -> bool:
    ea, eb = tldextract.extract(a), tldextract.extract(b)
    return (ea.domain, ea.suffix) == (eb.domain, eb.suffix)

def normalize_url(base: str, href: str) -> str:
    try:
        url = urljoin(base, href.strip())
        url, _ = urldefrag(url)
        return url
    except Exception:
        return ""

def get_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(base_url, a["href"])
        if u:
            out.append(u)
    return out

def chunk_text(text: str, chunk_size=600, overlap=100):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append((i, j, text[i:j]))
        if j == n:
            break
        i = j - overlap
    return chunks
