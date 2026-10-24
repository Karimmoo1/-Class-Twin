"""Response policy as a decision tree.

root: should I speak?
 |- ask_to_repeat  (cheap check: audio quality / too short)
 |- stay_silent    (cheap check: called someone else, or not a question)  + LLM score
 |- defer          (LLM score: it is for me but outside my notes)
 `- answer         (LLM score: it is for me and my notes/memory cover it)

Cheap checks run first. If one fires strongly, the LLM is never called and the LLM branches
are pruned. Otherwise the LLM scores only what is left. prune_and_select() drops every branch
below its threshold and picks the best survivor. If nothing survives -> safe reply.
"""
import config
import textutils as tu
from llm import llm_json
from schemas import Branch
from tools import call_tool

PRIORITY = ["ask_to_repeat", "stay_silent", "defer", "answer"]   # tie-break: safest first
CHEAP_FIRE = 0.9


def prune_and_select(branches: list[Branch]):
    for b in branches:
        b.status = "kept" if (b.score is not None and b.score >= b.threshold) else "pruned"
    kept = [b for b in branches if b.status == "kept"]
    if not kept:
        return None
    best = max(kept, key=lambda b: (b.score, -PRIORITY.index(b.action)))
    best.status = "selected"
    return best


def _features(tr: dict, memory) -> dict:
    text = tr["text"]
    name = tu.name_match(text)
    voc = tu.find_vocative(text)
    other = bool(voc) and tu.name_match(voc) is None
    q = tu.is_question(text)
    fu = tu.is_followup(text)
    continuing = fu and memory.last_speaker == "me"
    addressed = bool(name) or continuing
    open_q = (not addressed) and (not other) and (tu.is_open_question(text) or q)
    return {"text": text, "quality": tr["quality"], "language": tr["language"],
            "word_count": len(tu.words(text)), "name_in_text": name, "vocative": voc,
            "addressed_to_other": other, "is_question": q, "is_followup": fu,
            "addressed_to_me": addressed, "open_question": open_q}


def _llm_scores(f: dict, memory) -> tuple[dict, list]:
    query = tu.make_query(f["text"], memory.last_said())
    hits = call_tool("search_my_notes", query=query, k=3)
    notes = "\n".join(f"[{h['source']}] {h['text'][:800]}" for h in hits) or "(none)"
    system = ("You are the decision module of Karim's AI twin in a university class meeting. "
              "Output JSON only, with keys: answer, defer, stay_silent (numbers 0 to 1) and reason (short).")
    user = f"""Transcript of what was just said: "{f['text']}"
Addressed to me (Karim): {f['addressed_to_me']}
Open question to the whole class: {f['open_question']}
Follow-up of something I said: {f['is_followup']}

MEETING MEMORY:
{memory.context()}

MY NOTES (top matches):
{notes}

Score each action from 0 to 1:
- answer: I can answer correctly using ONLY my notes, my memory of this meeting, or my profile.
- defer: it is for me, but the notes/memory do not cover it, or it asks about work I did not do. I must not guess.
- stay_silent: it is not for me, needs no reply, or it is an open question to the class that my notes do not clearly cover (I never speak unprompted without good reason).
Be strict: a topic that only looks similar is NOT covered. If a good answer would need a personal story or experience that is not written in my notes, the answer score must be below 0.5."""
    out = llm_json(system, user) or {}

    def num(k):
        try:
            return max(0.0, min(1.0, float(out.get(k, 0.0))))
        except (TypeError, ValueError):
            return 0.0
    return {"answer": num("answer"), "defer": num("defer"), "stay_silent": num("stay_silent"),
            "reason": str(out.get("reason", ""))}, hits


def decide(tr: dict, memory) -> dict:
    f = _features(tr, memory)
    th = config.DECISION_THRESHOLD
    ans_th = config.OPEN_Q_ANSWER_MIN if f["open_question"] else th
    B = {a: Branch(action=a, kind=k, threshold=(ans_th if a == "answer" else th))
         for a, k in [("ask_to_repeat", "check"), ("stay_silent", "check+llm"),
                      ("defer", "llm"), ("answer", "llm")]}

    # 1) cheap checks
    bad_audio = f["quality"] < config.QUALITY_MIN or f["word_count"] < 2
    B["ask_to_repeat"].score = 0.95 if bad_audio else 0.0
    B["ask_to_repeat"].reason = (f"quality {f['quality']:.2f} (min {config.QUALITY_MIN}), "
                                 f"{f['word_count']} words")
    cheap_silent = 0.0
    if f["addressed_to_other"]:
        cheap_silent, why = 0.9, f"called someone else: '{f['vocative']}'"
    elif not f["addressed_to_me"] and not f["is_question"]:
        cheap_silent, why = 0.9, "no question and my name was not said"
    else:
        why = "no cheap reason to stay silent"
    B["stay_silent"].score = cheap_silent
    B["stay_silent"].reason = why

    # 2) LLM only for what is left
    hits, llm_reason = [], ""
    if max(B["ask_to_repeat"].score, cheap_silent) >= CHEAP_FIRE:
        for a in ("defer", "answer"):
            B[a].score = None
            B[a].reason = "not scored: a cheap check already decided"
    else:
        s, hits = _llm_scores(f, memory)
        llm_reason = s["reason"]
        B["answer"].score, B["defer"].score = s["answer"], s["defer"]
        B["stay_silent"].score = max(cheap_silent, s["stay_silent"])
        B["answer"].reason = f"LLM score (threshold {ans_th:.2f}" + (", open question)" if f["open_question"] else ")")
        B["defer"].reason = "LLM score"
        B["stay_silent"].reason += f"; LLM {s['stay_silent']:.2f}"

    # 3) prune and select
    branches = [B[a] for a in PRIORITY]
    chosen = prune_and_select(branches)
    fallback = chosen is None
    if fallback:
        # Safe default: if the question was for me, ask to repeat; if it was an open question
        # to the class, do not interrupt - stay silent.
        chosen = B["stay_silent"] if f["open_question"] else B["ask_to_repeat"]
        chosen.status = "selected"
        chosen.reason += " | FALLBACK: nothing cleared the threshold -> safe default, no guessing"

    return {"features": f, "selected": chosen.action, "fallback": fallback,
            "language": f["language"], "addressed_to_me": f["addressed_to_me"],
            "answer_score": B["answer"].score or 0.0, "llm_reason": llm_reason,
            "notes_preview": [{"source": h["source"], "score": h["score"]} for h in hits],
            "branches": [b.model_dump() for b in branches]}


def render_tree(d: dict) -> str:
    f = d["features"]
    mark = {"selected": "★ SELECTED", "kept": "✓ kept", "pruned": "✂ pruned", "pending": "?"}
    lines = [f"root: should I speak?   (threshold {config.DECISION_THRESHOLD})",
             f"  name heard: {f['name_in_text']} | vocative: {f['vocative']} | question: {f['is_question']} "
             f"| open: {f['open_question']} | follow-up: {f['is_followup']} | quality: {f['quality']}"]
    for i, b in enumerate(d["branches"]):
        last = i == len(d["branches"]) - 1
        score = "n/a" if b["score"] is None else f"{b['score']:.2f}"
        lines.append(f"{'└─' if last else '├─'} {b['action']:<13} [{b['kind']}] score {score} "
                     f"(min {b['threshold']:.2f})  {mark[b['status']]}")
        lines.append(f"{'   ' if last else '│  '}   why: {b['reason']}")
    if d["llm_reason"]:
        lines.append(f"LLM reason: {d['llm_reason']}")
    return "\n".join(lines)
