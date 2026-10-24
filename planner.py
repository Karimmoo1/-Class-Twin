"""Reply = dependency graph run in graphlib order, with parallel branches."""
import time
from concurrent.futures import ThreadPoolExecutor
from graphlib import TopologicalSorter

import textutils as tu
from config import load_profile
from llm import llm_json
from schemas import TwinReply
from tools import CALL_LOG, call_tool

# node -> the nodes it depends on. The four tool nodes are independent (run in parallel).
GRAPH = {
    "get_profile": set(),
    "search_my_notes": set(),
    "get_style_examples": set(),
    "lookup_glossary": set(),
    "build_prompt": {"get_profile", "search_my_notes", "get_style_examples", "lookup_glossary"},
    "llm_reply": {"build_prompt"},
}

_STOP = {"what", "does", "do", "is", "the", "a", "an", "of", "in", "and", "to", "how", "why", "can",
         "you", "your", "me", "mean", "means", "difference", "between", "explain", "tell", "us",
         "that", "this", "about", "are", "it", "for", "karim"}


def run_graph(graph: dict, funcs: dict, max_workers: int = 4):
    """Run nodes in topological order; every wave of ready nodes runs in parallel.
    Raises graphlib.CycleError if the plan has a cycle."""
    ts = TopologicalSorter(graph)
    ts.prepare()
    results, waves, node_secs = {}, [], {}

    def timed(name):
        t0 = time.perf_counter()
        out = funcs[name](results)
        node_secs[name] = round(time.perf_counter() - t0, 3)
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while ts.is_active():
            ready = list(ts.get_ready())
            waves.append(ready)
            futs = {n: ex.submit(timed, n) for n in ready}
            for n, fut in futs.items():
                results[n] = fut.result()
                ts.done(n)
    return results, waves, node_secs


def _glossary_term(question: str) -> str:
    latin = [w for w in tu.words(question) if w.isascii() and w not in _STOP and w not in tu.name_tokens()]
    return " ".join(latin) or question


def _lang_rule(lang: str) -> str:
    return {"ar": "Egyptian Arabic (عامية مصرية), NOT formal Arabic. Technical terms stay in English letters.",
            "mixed": "Egyptian Arabic mixed with English technical terms in Latin letters, the way I talk.",
            "en": "English."}[lang]


def _build_prompt(lang, question, memory, r):
    prof = r["get_profile"]
    style = prof["speaking_style"]
    notes = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in r["search_my_notes"]) or "(no notes found)"
    gl = r["lookup_glossary"]
    glossary = f"{gl['term']}: {gl['definition']}" if gl.get("found") else "(no glossary hit)"
    shots = "\n".join(f"Q: {e['question']}\nA: {e['answer']}" for e in r["get_style_examples"])
    system = f"""You are Karim's AI twin in a university class meeting. Answer exactly as Karim would: first person, {style['tone']}.
LANGUAGE: {_lang_rule(lang)}
LENGTH: {style['typical_answer_length']}.
STRUCTURE: {style['explains_with']}.
RULES:
- Use ONLY facts found in NOTES, GLOSSARY, MEETING MEMORY or PROFILE. If a detail is missing, say you are not sure. Never invent numbers, results or experience. NO PERSONAL STORIES: do not say things like 'I tried', 'in my project I noticed' or 'once I saw' unless that exact experience is written in NOTES. Explain the concept in general terms instead. If asked about an experience that is not in NOTES, say honestly that I have not worked on it yet.
- STYLE EXAMPLES show my voice (tone, fillers, terms kept in English). Copy the voice, NOT their facts, unless NOTES support them.
- Keep these terms in English: {', '.join(style['keep_in_english'])}.
- {' '.join(prof['honesty_rules'])}
- Return JSON only: {{"reply_text": str, "confidence": number 0-1, "sources": [note file names you used]}}"""
    user = f"""QUESTION: {question}

MEETING MEMORY:
{memory.context()}

PROFILE BACKGROUND:
{chr(10).join('- ' + b for b in prof['background'])}

NOTES:
{notes}

GLOSSARY: {glossary}

STYLE EXAMPLES:
{shots}"""
    return system, user


def _canned(action: str, lang: str) -> str:
    safe = load_profile()["safe_replies"]
    if action == "stay_silent":
        return ""
    ar = lang in ("ar", "mixed")
    if action == "ask_to_repeat":
        return safe["ask_to_repeat_ar" if ar else "ask_to_repeat_en"]
    return safe["defer_ar" if ar else "defer_en"]


def reply(decision: dict, tr: dict, memory) -> tuple[TwinReply, dict]:
    action, lang = decision["selected"], decision["language"]
    plan = {"waves": [], "node_seconds": {}, "timings": {}}
    base = dict(addressed_to_me=decision["addressed_to_me"], action=action, language=lang)

    if action != "answer":
        return TwinReply(**base, reply_text=_canned(action, lang), confidence=1.0 if action != "defer" else 0.9,
                         tools_used=[c["tool"] for c in CALL_LOG], sources=[]), plan

    q = tr["text"]
    query = tu.make_query(q, memory.last_said())
    funcs = {
        "get_profile": lambda r: call_tool("get_profile"),
        "search_my_notes": lambda r: call_tool("search_my_notes", query=query, k=3),
        "get_style_examples": lambda r: call_tool("get_style_examples", question=q, language=lang, k=4),
        "lookup_glossary": lambda r: call_tool("lookup_glossary", term=_glossary_term(q)),
        "build_prompt": lambda r: _build_prompt(lang, q, memory, r),
        "llm_reply": lambda r: llm_json(*r["build_prompt"], temperature=0.3),
    }
    t0 = time.perf_counter()
    results, waves, node_secs = run_graph(GRAPH, funcs)
    plan["waves"], plan["node_seconds"] = waves, node_secs
    plan["timings"] = {"tools_parallel": round(max(node_secs[n] for n in waves[0]), 3),
                       "llm": node_secs["llm_reply"], "plan_total": round(time.perf_counter() - t0, 3)}

    out = results["llm_reply"] or {}
    text = str(out.get("reply_text", "")).strip()
    if not text:   # LLM failed -> never guess, fall back to the safe reply
        return TwinReply(**{**base, "action": "ask_to_repeat"}, reply_text=_canned("ask_to_repeat", lang),
                         confidence=0.0, tools_used=list(dict.fromkeys(c["tool"] for c in CALL_LOG)),
                         sources=[]), plan
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    available = {h["source"] for h in results["search_my_notes"]}
    sources = [s for s in out.get("sources", []) if s in available] or sorted(available)[:1]
    return TwinReply(**base, reply_text=text, confidence=round(min(conf, decision["answer_score"] or conf), 2),
                     tools_used=list(dict.fromkeys(c["tool"] for c in CALL_LOG)), sources=sources), plan
