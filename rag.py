"""Core RAG pipeline for the AI Study Assistant."""
import os
import re
import hashlib
import unicodedata
from collections import Counter
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

_embedder = None


def get_embedder(name=EMBED_MODEL):
    global _embedder
    if _embedder is None or _embedder[0] != name:
        _embedder = (name, SentenceTransformer(name))
    return _embedder[1]


# ---------- 1. Load PDFs ----------
def ocr_available():
    """True if Tesseract + pytesseract are installed (needed for image-only slides)."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page(page, dpi=170):
    import io
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    w, h = img.size
    img = img.crop((0, int(0.03 * h), w, int(0.94 * h)))  # skip slide header/footer strip
    return pytesseract.image_to_string(img).strip()


_OCR_CACHE = {}


def load_pdf(path_or_bytes, source_name, ocr=True):
    """Return cleaned pages [{'text','source','page'}].
    Pages with no text layer (image-only slides) are read with OCR if Tesseract is installed."""
    if not isinstance(path_or_bytes, (bytes, bytearray)):
        with open(path_or_bytes, "rb") as f:
            path_or_bytes = f.read()
    doc = fitz.open(stream=path_or_bytes, filetype="pdf")
    raw = [{"text": p.get_text().strip(), "source": source_name, "page": i + 1}
           for i, p in enumerate(doc)]
    pages = clean_pages(raw)
    if ocr and ocr_available():
        have = {p["page"] for p in pages}
        key = hashlib.md5(path_or_bytes).hexdigest()
        ocr_raw = []
        for i, page in enumerate(doc):
            if i + 1 in have:
                continue
            if (key, i) not in _OCR_CACHE:  # OCR once per PDF, even if settings change
                _OCR_CACHE[(key, i)] = _ocr_page(page)
            ocr_raw.append({"text": _OCR_CACHE[(key, i)], "source": source_name, "page": i + 1})
        pages = sorted(pages + clean_pages(ocr_raw), key=lambda p: p["page"])
    return pages


# ---------- 1b. Clean slide-deck noise ----------
def _strip_junk_chars(s):
    out = []
    for ch in s:
        if ch in "\n\t":
            out.append(ch)
        elif ch == "\ufffd" or unicodedata.category(ch).startswith("C"):
            continue  # replacement chars, private-use glyphs, control chars
        else:
            out.append(ch)
    return "".join(out)


def _line_key(line):
    # ignore spaces; for SHORT lines also ignore digits so "ML 10" and "ML 20"
    # count as the same footer, while long content lines are compared exactly
    key = re.sub(r"\s+", "", line).lower()
    if len(key) <= 60:
        key = re.sub(r"\d+", "", key)
    return key


def clean_pages(pages, repeat_ratio=0.25, min_chars=40):
    """Remove headers/footers that repeat across pages and junk symbols."""
    if not pages:
        return pages
    split = []
    counts = Counter()
    for p in pages:
        lines = [re.sub(r"[ \t]+", " ", l).strip() for l in _strip_junk_chars(p["text"]).split("\n")]
        lines = [l for l in lines if l]
        split.append(lines)
        counts.update({_line_key(l) for l in lines})
    limit = max(3, repeat_ratio * len(pages))
    cleaned = []
    for p, lines in zip(pages, split):
        keep = [l for l in lines if counts[_line_key(l)] < limit and _line_key(l)]
        text = re.sub(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}", "", " ".join(keep)).strip()
        if len(text) >= min_chars:
            cleaned.append({**p, "text": text})
    return cleaned


# ---------- 2. Chunk ----------
def chunk_pages(pages, chunk_size=800, overlap=150):
    """Split each page into overlapping character chunks, keeping page metadata."""
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "source": p["source"], "page": p["page"]})
    return chunks


# ---------- 3. Index + retrieve ----------
class NotesIndex:
    def __init__(self, chunks, embed_model=EMBED_MODEL):
        self.chunks = chunks
        self.embed_model = embed_model
        model = get_embedder(embed_model)
        vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                            show_progress_bar=False)
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim (vectors normalized)
        self.index.add(np.asarray(vecs, dtype="float32"))

    def search(self, query, k=4):
        q = get_embedder(self.embed_model).encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [dict(self.chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


# ---------- 4. LLM ----------
def ask_llm(prompt, max_tokens=None):
    # Thinking level "low" keeps answers fast; set THINKING_LEVEL=medium/high for harder questions.
    # No output cap: thinking tokens count toward the limit and used to cut answers off.
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    level = os.getenv("THINKING_LEVEL", "low")
    try:
        resp = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=level)),
        )
    except Exception as e:
        if "thinking" not in str(e).lower():
            raise
        # this model/SDK doesn't accept thinking_level: retry with defaults
        resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    if not resp.text:
        raise RuntimeError("Gemini returned an empty response (blocked or cut off). Try again.")
    return resp.text


def _context(hits):
    return "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)


def answer_question(index, question, k=4):
    hits = index.search(question, k)
    prompt = (
        "You are a study assistant. Answer the question using ONLY the notes below. "
        "If the answer is not in the notes, say so. Mention page numbers like (p.3).\n\n"
        f"NOTES:\n{_context(hits)}\n\nQUESTION: {question}"
    )
    return ask_llm(prompt), hits


def summarize_topic(index, topic, k=6):
    hits = index.search(topic, k)
    prompt = (
        "Summarize the topic below for exam revision using ONLY these notes. "
        "Use short bullet points and highlight key terms.\n\n"
        f"NOTES:\n{_context(hits)}\n\nTOPIC: {topic}"
    )
    return ask_llm(prompt), hits


def generate_quiz(index, topic, n=5, k=6):
    hits = index.search(topic, k)
    prompt = (
        f"Create {n} multiple-choice questions (4 options each) from ONLY these notes. "
        "Test course concepts only (not names, dates or slide headers) and make wrong options plausible. "
        "Use EXACTLY this markdown format for every question, with a blank line between every line:\n\n"
        "**Q1. question text** (p.X)\n\n- A) option\n\n- B) option\n\n- C) option\n\n- D) option\n\n"
        "**Answer:** B. one-line explanation\n\n---\n\n"
        f"NOTES:\n{_context(hits)}\n\nTOPIC: {topic}"
    )
    return ask_llm(prompt), hits
