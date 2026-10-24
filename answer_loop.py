"""Inner loop: Groq tool-calling over the Phase 1 tools, with feedback, a repeat-blocker and a goal-check.

answer(tr, decision, memory) -> (TwinReply, trace)
Everything that touches the network (the LLM, the tools) is injectable, so the forced tests can
run offline with stubs: think_fn=..., runner=...
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import config
import grounding
import live_config as lc
import textutils as tu
from agent_loop import run_agent_loop
from schemas import TwinReply

GLOSSARY_FILE = "ml_glossary.md"


# ---------------------------------------------------------------- goal-check (deterministic, no LLM)
_LANG_OK = {"en": {"en"}, "ar": {"ar", "mixed"}, "mixed": {"ar", "mixed"}}


def goal_check(draft: dict, q_lang: str, seen_sources: set) -> dict:
    """Run on every draft before anything is spoken."""
    reply = parse_draft(draft.get("content", ""))
    text = reply["reply_text"]
    n = len(tu.words(text))
    r_lang = tu.detect_language(text) if text else None
    checks = {
        "not_empty": bool(text),
        "language_matches": r_lang in _LANG_OK[q_lang],
        "length_in_range": lc.LEN_MIN_WORDS <= n <= lc.LEN_MAX_WORDS,
        "cites_source": any(s in seen_sources for s in reply["sources"]),
    }
    unsupported = grounding.find_unsupported(text) if text else []
    checks["claims_grounded"] = not unsupported
    why = {
        "not_empty": "the reply is empty",
        "language_matches": f"the question is '{q_lang}' but the reply looks '{r_lang}'; "
                            f"reply in {'English' if q_lang == 'en' else 'Egyptian Arabic (English tech terms allowed)'}",
        "length_in_range": f"the reply has {n} words, it must have {lc.LEN_MIN_WORDS} to {lc.LEN_MAX_WORDS}",
        "cites_source": ("sources must list at least one note file returned by a tool, e.g. "
                         + (", ".join(sorted(seen_sources)[:3]) if seen_sources else
                            "(no source retrieved yet: call search_my_notes with a different query, "
                            "or say you are not sure)")),
        "claims_grounded": ("these sentences claim project work that is not in my notes: " + " | ".join(
            f'"{u["sentence"][:100]}" (not in notes: {", ".join(u["unsupported"])})' for u in unsupported[:2])
            + ". Delete them and explain the concept in general terms; say what I did only if a note says it"),
    }
    failed = [why[k] for k, v in checks.items() if not v]
    return {"ok": not failed, "checks": checks, "failed": failed, "words": n, "reply_language": r_lang,
            "unsupported_claims": unsupported}


def parse_draft(content: str) -> dict:
    """The model is asked for JSON; if it answers in plain text we still keep the text."""
    content = (content or "").strip()
    out = None
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                out = json.loads(m.group(0))
            except json.JSONDecodeError:
                out = None
    if not isinstance(out, dict):
        return {"reply_text": content, "confidence": 0.5, "sources": []}
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    srcs = out.get("sources", [])
    return {"reply_text": str(out.get("reply_text", "")).strip(), "confidence": conf,
            "sources": [s for s in srcs if isinstance(s, str)] if isinstance(srcs, list) else []}


# ---------------------------------------------------------------- tools: run, cache, block repeats
def _key(name, args):
    return name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def compact(name, result):
    """Shorter tool results = fewer tokens = faster and under the Groq free-tier limit."""
    cap = lc.TOOL_RESULT_CHARS
    if name == "search_my_notes":
        return [{"source": h["source"], "text": h["text"][:cap]} for h in result]
    if name == "get_style_examples":
        return [{"question": h["question"], "answer": h["answer"][:300]} for h in result[:3]]
    if name == "get_profile":
        return {k: result[k] for k in ("background", "honesty_rules") if k in result}
    return result


def sources_of(name, result) -> set:
    if name == "search_my_notes" and isinstance(result, list):
        return {h["source"] for h in result if isinstance(h, dict) and "source" in h}
    if name == "lookup_glossary" and isinstance(result, dict) and result.get("found"):
        return {GLOSSARY_FILE}
    return set()


class ToolRunner:
    """Runs the model's tool calls (in parallel), caches repeats and blocks exact repeats of failures.
    `runner(name, **args)` defaults to tools.call_tool (the Phase 1 logger)."""

    CACHEABLE = {"search_my_notes", "get_style_examples", "lookup_glossary", "get_profile"}
    _cache: dict = {}            # shared across questions: embeddings + lookups that repeat

    def __init__(self, runner=None):
        if runner is None:
            from tools import call_tool as runner
        self.runner = runner
        self.failed: set = set()
        self.seen_sources: set = set()

    def _one(self, call) -> dict:
        name, args = call["name"], call.get("args") or {}
        k = _key(name, args)
        obs = {"tool_call_id": call["id"], "name": name, "args": args, "ok": False, "blocked": False,
               "cached": False, "seconds": 0.0}
        if k in self.failed:
            obs.update(blocked=True, content=json.dumps({
                "error": "BLOCKED: this exact call already failed.",
                "hint": "change the arguments, use another tool, or say you are not sure"}))
            obs["preview"] = "BLOCKED (exact repeat of a failed call)"
            return obs
        t0 = time.perf_counter()
        try:
            if k in ToolRunner._cache:
                result, obs["cached"] = ToolRunner._cache[k], True
            else:
                result = self.runner(name, **args)
                if name in self.CACHEABLE:
                    ToolRunner._cache[k] = result
            self.seen_sources |= sources_of(name, result)
            obs.update(ok=True, content=json.dumps(compact(name, result), ensure_ascii=False, default=str))
            obs["preview"] = obs["content"][:200]
        except Exception as e:      # a failure is just another observation
            self.failed.add(k)
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            obs.update(content=json.dumps({"error": msg, "hint": "do not repeat this exact call; "
                                           "change the query, use another tool, or say you are not sure"}))
            obs["preview"] = "ERROR " + msg
        obs["seconds"] = round(time.perf_counter() - t0, 3)
        return obs

    def act(self, calls, context=None) -> list:
        if len(calls) == 1:
            return [self._one(calls[0])]
        with ThreadPoolExecutor(max_workers=4) as ex:
            return list(ex.map(self._one, calls))


# ---------------------------------------------------------------- the real LLM step (Groq tool-calling)
def _lang_rule(lang: str) -> str:
    return {"ar": "Egyptian Arabic (عامية مصرية), NOT formal Arabic. Technical terms stay in English letters.",
            "mixed": "Egyptian Arabic mixed with English technical terms in Latin letters, the way I talk.",
            "en": "English."}[lang]


def system_prompt(lang: str) -> str:
    prof = config.load_profile()
    st = prof["speaking_style"]
    return f"""You are Karim's AI twin in a university class meeting. Answer exactly as Karim would: first person, {st['tone']}.
LANGUAGE: {_lang_rule(lang)}
LENGTH: {st['typical_answer_length']}. STRUCTURE: {st['explains_with']}.
HOW TO WORK:
1. Call search_my_notes for facts, and lookup_glossary for any ML term. Call get_style_examples once to copy my voice (never my facts). You can call several tools at once.
2. If a tool returns an error or BLOCKED, do NOT repeat that exact call. Change the query, use another tool, or say you are not sure.
3. When you have enough, STOP calling tools and reply with JSON only: {{"reply_text": str, "confidence": number 0-1, "sources": [note file names you used, e.g. "ml_glossary.md"]}}
RULES: use ONLY facts from tool results, the background below or the meeting memory. Never invent numbers, results or experience. Do NOT add a project example to a concept question: explain the concept only. Say what I did in a project only if a note states exactly that (same project, same technique), otherwise leave it out. No personal stories. Keep these terms in English: {', '.join(st['keep_in_english'])}.
{' '.join(prof['honesty_rules'])}
BACKGROUND: {' '.join(prof['background'])}"""


def initial_context(tr, lang, memory) -> list:
    return [{"role": "system", "content": system_prompt(lang)},
            {"role": "user", "content": f"QUESTION: {tr['text']}\n\nMEETING MEMORY:\n{memory.context()}"}]


def make_groq_think(tool_objs=None):
    """think_fn for the real model. Needs GROQ_API_KEY."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_groq import ChatGroq
    if tool_objs is None:
        from tools import _TOOLS
        tool_objs = list(_TOOLS.values())
    try:
        base = ChatGroq(model=config.LLM_MODEL, temperature=0.3, reasoning_effort=config.LLM_REASONING)
    except Exception:
        base = ChatGroq(model=config.LLM_MODEL, temperature=0.3)
    llm = base.bind_tools(tool_objs)

    def to_lc(ctx):
        out = []
        for m in ctx:
            r = m["role"]
            if r == "system":
                out.append(SystemMessage(content=m["content"]))
            elif r == "user":
                out.append(HumanMessage(content=m["content"]))
            elif r == "assistant":
                out.append(AIMessage(content=m["content"], tool_calls=[
                    {"id": c["id"], "name": c["name"], "args": c["args"]} for c in m.get("tool_calls", [])]))
            else:
                out.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))
        return out

    def think(goal, ctx):
        msgs = to_lc(ctx)
        for attempt in range(5):      # Groq free tier: tokens-per-minute limit -> wait and retry
            try:
                resp = llm.invoke(msgs)
                break
            except Exception as e:
                if ("429" in str(e) or "rate_limit" in str(e)) and attempt < 4:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        calls = [{"id": c.get("id") or f"call_{i}", "name": c["name"], "args": c.get("args") or {}}
                 for i, c in enumerate(resp.tool_calls)]
        return {"content": content, "tool_calls": calls}
    return think


# ----------------------------------------------------------------- the public entry point
def _safe(kind: str, lang: str) -> str:
    if kind == "fallback":
        return lc.FALLBACKS["en" if lang == "en" else "ar"]
    safe = config.load_profile()["safe_replies"]
    return safe["defer_en" if lang == "en" else "defer_ar"]


def _trace(res, runner, reply_dict) -> dict:
    steps = []
    for s in res.steps:
        steps.append({
            "iteration": s["iteration"], "think_s": s["think_s"],
            "assistant_text": s["assistant"]["content"][:300],
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in s["assistant"]["tool_calls"]],
            "tool_results": [{k: o[k] for k in ("name", "args", "ok", "blocked", "cached", "seconds", "preview")}
                             for o in s.get("tool_results", [])],
            **({"goal_check": s["goal_check"]} if "goal_check" in s else {}),
        })
    return {"termination_reason": res.reason, "iterations": res.iterations, "retries": res.retries,
            "goal_check": res.goal_check, "seconds": res.seconds, "steps": steps,
            "failed_calls": sorted(runner.failed), "seen_sources": sorted(runner.seen_sources)}


def answer(tr: dict, decision: dict, memory, think_fn=None, runner=None,
           max_iterations: int = None) -> tuple[TwinReply, dict]:
    lang = decision["language"]
    max_iterations = max_iterations or lc.MAX_ITERATIONS
    think_fn = think_fn or make_groq_think()
    tools = ToolRunner(runner)
    res = run_agent_loop(
        goal=tr["text"], think_fn=think_fn, act_fn=tools.act, max_iterations=max_iterations,
        goal_check_fn=lambda d: goal_check(d, lang, tools.seen_sources),
        context=initial_context(tr, lang, memory))
    used = list(dict.fromkeys(o["name"] for s in res.steps for o in s.get("tool_results", []) if o["ok"]))
    base = dict(addressed_to_me=decision.get("addressed_to_me", True), language=lang, tools_used=used)

    if res.reason == "model_stopped":
        d = parse_draft(res.draft["content"])
        cap = decision.get("answer_score") or d["confidence"]
        reply = TwinReply(**base, action="answer", reply_text=d["reply_text"],
                          confidence=round(min(d["confidence"], cap), 2),
                          sources=[s for s in d["sources"] if s in tools.seen_sources])
    elif res.reason == "max_iterations":      # cap reached: speak the safe fallback, never a half draft
        reply = TwinReply(**base, action="defer", reply_text=_safe("fallback", lang), confidence=0.0, sources=[])
    else:                                     # goal-check failed twice: defer, the bad draft is never spoken
        reply = TwinReply(**base, action="defer", reply_text=_safe("defer", lang), confidence=0.0, sources=[])
    return reply, _trace(res, tools, reply.model_dump())
