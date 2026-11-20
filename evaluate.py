"""Measure retrieval accuracy (top-k hit rate) on your own test questions.

Usage:
    python evaluate.py notes.pdf eval.json

eval.json format (write 15-20 questions from YOUR notes):
[
  {"question": "What is overfitting?", "page": 12},
  {"question": "Define gradient descent", "page": 15}
]
"page" can be a number or a list of acceptable pages (useful for slides that repeat content).
A retrieval counts as a HIT if any of the top-k chunks comes from an expected page.
"""
import json
import sys
from rag import load_pdf, chunk_pages, NotesIndex

K = 3
CONFIGS = [
    # (chunk_size, overlap, embedding model)
    (500, 100, "all-MiniLM-L6-v2"),
    (800, 150, "all-MiniLM-L6-v2"),
    (1200, 200, "all-MiniLM-L6-v2"),
    (800, 150, "BAAI/bge-small-en-v1.5"),
]


def evaluate(pages, questions, chunk_size, overlap, model):
    index = NotesIndex(chunk_pages(pages, chunk_size, overlap), embed_model=model)
    hits = 0
    for item in questions:
        results = index.search(item["question"], K)
        expected = item["page"] if isinstance(item["page"], list) else [item["page"]]
        if any(r["page"] in expected for r in results):
            hits += 1
    return hits / len(questions)


if __name__ == "__main__":
    pdf_path, eval_path = sys.argv[1], sys.argv[2]
    pages = load_pdf(pdf_path, pdf_path)
    questions = json.load(open(eval_path))

    print(f"\nEvaluating {len(questions)} questions, top-{K} retrieval\n")
    print(f"{'chunk':>6} {'overlap':>8} {'model':<28} {'top-%d acc' % K}")
    print("-" * 60)
    for size, ov, model in CONFIGS:
        acc = evaluate(pages, questions, size, ov, model)
        print(f"{size:>6} {ov:>8} {model:<28} {acc:.0%}")
