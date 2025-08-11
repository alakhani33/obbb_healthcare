# ingest.py — pdfplumber/bs4-only ingestion (no unstructured)
import os, sys, pathlib, re
from typing import List, Tuple, Optional
from dotenv import load_dotenv
load_dotenv()

import requests
import pdfplumber
from bs4 import BeautifulSoup

from rag_core import init_chroma, get_embeddings, chunk_text, add_to_vectorstore

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUM_DIR = DATA_DIR / "summaries"
DATA_DIR.mkdir(exist_ok=True, parents=True)
SUM_DIR.mkdir(exist_ok=True, parents=True)

UA = {"User-Agent": "Mozilla/5.0 (OBBB-Advisor; +https://caliber360ai.com)"}
RESET = "--reset" in sys.argv  # optional: wipe Chroma collection before indexing


def log(msg: str) -> None:
    print(f"[ingest] {msg}")


def http_get(url: str) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r


def save_bytes(path: pathlib.Path, content: bytes) -> None:
    path.write_bytes(content)
    log(f"Saved -> {path} ({len(content)} bytes)")


def download_direct(url: str, target: pathlib.Path) -> None:
    log(f"Downloading file: {url}")
    r = http_get(url)
    save_bytes(target, r.content)


def discover_first_pdf_on_page(url: str) -> Optional[str]:
    """Generic: look for the first PDF link on any HTML page."""
    log(f"Scanning page for PDF: {url}")
    try:
        r = http_get(url)
    except Exception as e:
        log(f"Failed to fetch page: {e}")
        return None
    soup = BeautifulSoup(r.text, "lxml")
    # Prefer obvious PDF links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") or ".pdf?" in href.lower():
            return requests.compat.urljoin(url, href)
    return None


def normalize_sources(cli_args: List[str]) -> list[tuple[str, str]]:
    """
    Accepts CLI args or .env BILL_SOURCES (comma-separated).
    You can prefix with 'text=' to label something as a summary.
    Example:
      BILL_SOURCES=https://.../bill.pdf,text=https://.../summary.html
    """
    env_val = os.getenv("BILL_SOURCES", "")
    raw = cli_args if cli_args else [s.strip() for s in env_val.split(",") if s.strip()]
    out: list[tuple[str, str]] = []
    for src in raw:
        kind = "bill"
        if src.lower().startswith("text="):
            src = src.split("=", 1)[1].strip()
            kind = "summary"
        out.append((kind, src))
    return out


def fetch_sources(sources: list[tuple[str, str]]) -> int:
    """Download URLs into data/ (and mark summaries into data/ as well). Returns count saved."""
    n = 0
    for idx, (kind, url) in enumerate(sources, 1):
        try:
            lower = url.lower()
            guess_ext = ".pdf" if ".pdf" in lower else (".txt" if lower.endswith(".txt") else ".html")
            if lower.endswith((".pdf", ".html", ".htm", ".txt")) or any(s in lower for s in [".pdf?", ".html?", ".htm?"]):
                target = DATA_DIR / (f"{'bill' if kind=='bill' else 'summary'}{idx}{guess_ext}")
                download_direct(url, target)
                n += 1
                continue

            # Page → try to find a PDF on it
            pdf_url = discover_first_pdf_on_page(url)
            if pdf_url:
                target = DATA_DIR / (f"{'bill' if kind=='bill' else 'summary'}{idx}.pdf")
                download_direct(pdf_url, target)
                n += 1
            else:
                # Fallback: save page HTML
                r = http_get(url)
                target = DATA_DIR / (f"{'bill' if kind=='bill' else 'summary'}{idx}.html")
                save_bytes(target, r.content)
                n += 1
        except Exception as e:
            log(f"Download failed for {url}: {e}")
    return n


def read_pdf(path: pathlib.Path) -> List[Tuple[int, str]]:
    """Extract text per page with pdfplumber. Returns [(page_num, text), ...]."""
    texts: List[Tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                texts.append((i, txt))
            else:
                # Likely a scanned page; note it for possible OCR later.
                log(f"Page {i} had no extractable text (scanned or image-based).")
    return texts


def read_html(path: pathlib.Path) -> List[Tuple[Optional[int], str]]:
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n", strip=True)
    return [(None, text)]


def read_txt(path: pathlib.Path) -> List[Tuple[Optional[int], str]]:
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def parse_and_add(paths: List[pathlib.Path], coll, embeddings) -> None:
    for p in paths:
        title = p.name
        log(f"Parsing: {p}")
        try:
            if p.suffix.lower() == ".pdf":
                pages = read_pdf(p)
                all_chunks = []
                for page_num, ptext in pages:
                    chunks = chunk_text(ptext, doc_title=title, source_path=str(p))
                    for c in chunks:
                        c["metadata"]["page"] = page_num
                        m = re.search(r"(Sec\.?\s*\d+[A-Za-z\-]*)", c["text"])
                        if m:
                            c["metadata"]["section"] = m.group(1)
                    all_chunks.extend(chunks)
                if all_chunks:
                    add_to_vectorstore(coll, embeddings, all_chunks)
                    log(f"Indexed chunks: {len(all_chunks)}")
                else:
                    log("No extractable text in this PDF. Consider OCR fallback.")
            elif p.suffix.lower() in {".html", ".htm"}:
                items = read_html(p)
                total = 0
                for _, ptext in items:
                    chunks = chunk_text(ptext, doc_title=title, source_path=str(p))
                    add_to_vectorstore(coll, embeddings, chunks)
                    total += len(chunks)
                log(f"Indexed chunks: {total}")
            elif p.suffix.lower() == ".txt":
                items = read_txt(p)
                total = 0
                for _, ptext in items:
                    chunks = chunk_text(ptext, doc_title=title, source_path=str(p))
                    add_to_vectorstore(coll, embeddings, chunks)
                    total += len(chunks)
                log(f"Indexed chunks: {total}")
        except Exception as e:
            log(f"ERROR parsing {p.name}: {e}")


def collect_files(directory: pathlib.Path) -> List[pathlib.Path]:
    files = sorted(directory.glob("*"))
    return [p for p in files if p.suffix.lower() in {".pdf", ".html", ".htm", ".txt"} and p.is_file()]


def reset_collection():
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.Client(Settings(persist_directory="./chroma_db", is_persistent=True))
        client.delete_collection("big_bill_collection")
        log("Reset: deleted existing Chroma collection.")
    except Exception:
        # Ok if it didn't exist
        pass


def main():
    # 1) Optional reset
    if RESET:
        reset_collection()

    # 2) Pull sources from CLI or .env
    sources = normalize_sources([a for a in sys.argv[1:] if not a.startswith("--")])
    if sources:
        count = fetch_sources(sources)
        log(f"Downloaded {count} source file(s).")
    else:
        log("No BILL_SOURCES or CLI URLs provided; will index any existing local files.")

    # 3) Index local files
    main_files = collect_files(DATA_DIR)
    sum_files = collect_files(SUM_DIR)
    if not main_files and not sum_files:
        log("No input files found. Place files under /data or /data/summaries, or set BILL_SOURCES/CLI URLs.")
        log(f"Expected folders:\n - {DATA_DIR}\n - {SUM_DIR}")
        return

    log(f"Found {len(main_files)} main file(s), {len(sum_files)} summary file(s).")
    _, coll = init_chroma()
    embeddings = get_embeddings()
    if main_files:
        parse_and_add(main_files, coll, embeddings)
    if sum_files:
        parse_and_add(sum_files, coll, embeddings)

    log("Ingestion complete.")


if __name__ == "__main__":
    main()
