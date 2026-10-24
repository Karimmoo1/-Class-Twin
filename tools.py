"""The four @tool functions + call_tool(), which runs a tool and logs name, JSON args and time."""
import json
import re
import threading
import time
from difflib import get_close_matches
from functools import lru_cache

from langchain_core.tools import tool

import config
import textutils as tu


@tool
def search_my_notes(query: str, k: int = 3) -> list[dict]:
    """Search my own notes (project write-ups, course notes) and return the k most relevant chunks."""
    from retrieval import notes_index
    return [{"source": it["source"], "text": it["text"], "score": round(s, 3)}
            for it, s in notes_index().search(query, k)]


@tool
def get_style_examples(question: str, language: str = "en", k: int = 4) -> list[dict]:
    """Return k of my past Q&A pairs most similar to the question, preferring the same language
    ('ar', 'en' or 'mixed'). Used to copy my tone, not my facts."""
    from retrieval import style_index
    pref = {"ar": ["ar", "mixed"], "mixed": ["mixed", "ar"], "en": ["en"]}.get(language, ["en"])
    hits = style_index().search(question, 12)
    ordered = [h for h in hits if h[0]["language"] in pref] + [h for h in hits if h[0]["language"] not in pref]
    return [{"question": it["question"], "answer": it["answer"],
             "language": it["language"], "score": round(s, 3)} for it, s in ordered[:k]]


@tool
def get_profile() -> dict:
    """Return my profile: name, background, how I speak, and the rules I follow."""
    return config.load_profile()


@lru_cache(maxsize=1)
def _glossary() -> dict:
    g = {}
    path = config.DATA_DIR / "knowledge" / "ml_glossary.md"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^- ([^:]+):\s*(.+)$", line.strip())
            if m:
                g[m.group(1).strip().lower()] = m.group(2).strip()
    return g


@tool
def lookup_glossary(term: str) -> dict:
    """Look up a short definition of an ML/GenAI term in my own course glossary."""
    g = _glossary()
    t = term.lower().strip()
    key = t if t in g else None
    if not key:   # token overlap, e.g. 'pruning decision tree' -> 'decision tree pruning'
        tt = set(tu.words(t))
        best, best_s = None, 0.5
        for k in g:
            kt = set(tu.words(k))
            s = len(tt & kt) / max(len(kt), 1)
            if s > best_s:
                best, best_s = k, s
        key = best
    if not key:
        close = get_close_matches(t, list(g), n=1, cutoff=0.6)
        key = close[0] if close else None
    return {"found": True, "term": key, "definition": g[key]} if key else {"found": False, "term": term}


_TOOLS = {t.name: t for t in [search_my_notes, get_style_examples, get_profile, lookup_glossary]}
CALL_LOG: list[dict] = []
_log_lock = threading.Lock()


def clear_call_log():
    with _log_lock:
        CALL_LOG.clear()


def call_tool(name: str, **args):
    t0 = time.perf_counter()
    out = _TOOLS[name].invoke(args)
    dt = time.perf_counter() - t0
    with _log_lock:
        CALL_LOG.append({"tool": name, "args": args, "seconds": round(dt, 3),
                         "result_preview": json.dumps(out, ensure_ascii=False)[:300]})
    return out
