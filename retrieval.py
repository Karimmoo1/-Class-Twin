"""FAISS + multilingual-e5-small. Heavy imports are lazy so the other modules load fast."""
import json
import threading
from functools import lru_cache

import config

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def _encode(texts):
    with _lock:
        return _embedder().encode(texts, normalize_embeddings=True,
                                  convert_to_numpy=True, batch_size=32).astype("float32")


@lru_cache(maxsize=256)
def _encode_query(q: str):
    return _encode(["query: " + q])


class Index:
    """doc_prefix: 'passage: ' for notes. For question-to-question matching e5 recommends
    'query: ' on both sides (symmetric task)."""

    def __init__(self, items, text_key="text", doc_prefix="passage: "):
        import faiss
        self.items = items
        embs = _encode([doc_prefix + it[text_key] for it in items])
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(self, query: str, k: int = 3):
        k = min(k, len(self.items))
        scores, ids = self.index.search(_encode_query(query), k)
        return [(self.items[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def chunk_text(text: str, max_chars: int = 700):
    paras = []
    for blk in text.split("\n\n"):
        blk = blk.strip()
        if not blk:
            continue
        if len(blk) <= max_chars:
            paras.append(blk)
        else:  # long block (e.g. the glossary list): one piece per line
            paras.extend(l.strip() for l in blk.splitlines() if l.strip())
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


@lru_cache(maxsize=1)
def notes_index() -> Index:
    items = []
    for p in sorted((config.DATA_DIR / "knowledge").glob("**/*")):
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        text = p.read_text(encoding="utf-8")
        title = text.strip().splitlines()[0].lstrip("# ").strip() if text.strip() else p.stem
        for c in chunk_text(text):
            body = c if c.lstrip().startswith("#") else f"{title}\n{c}"
            items.append({"source": p.name, "text": body})
    if not items:
        raise RuntimeError(f"No notes found in {config.DATA_DIR / 'knowledge'}")
    return Index(items)


@lru_cache(maxsize=1)
def style_index() -> Index:
    path = config.DATA_DIR / "style_examples.jsonl"
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return Index(items, text_key="question", doc_prefix="query: ")


def warmup():
    """Load the embedding model and build both indexes now, so the first question is fast."""
    notes_index()
    style_index()
    _encode_query("warmup")
