"""Tests for the live layer: Egyptian Arabic / mixed Arabic-English / English greetings, conversation window,
repeat / rephrase / continue requests, who-is-talking-to-whom, error recovery. Offline (no Groq, no audio).
    python test_live_layer.py"""
import json
import sys
import time
from types import SimpleNamespace

import answer_loop
import live_answer as la
import textutils as tu
from live_answer import (REPLIES, WINDOW_S, _other_named, conversation_active, live_answer, live_canned, live_decide,
                         recover, reply_for, small_talk_kind)

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print("OK  " if cond else "FAIL", name)


# ---------------------------------------------------------------- 1. errors from the first live run
class FakeBadRequest(Exception):
    pass


msg1 = ("Error code: 400 - {'error': {'message': \"Tool call validation failed: attempted to call tool 'JSON' "
        "which was not in request.tools\", 'type': 'invalid_request_error', 'code': 'tool_use_failed', "
        "'failed_generation': '{\"name\": \"JSON\", \"arguments\": {\"reply_text\":\"I\\'m doing well, thanks! "
        "Just gearing up for the next class.\", \"confidence\":1.0, \"sources\":[]}}'}}")
rec = recover(FakeBadRequest(msg1), {"search_my_notes"})
check("recover: fake 'JSON' tool call -> final reply", rec and not rec["tool_calls"]
      and json.loads(rec["content"])["reply_text"].startswith("I'm doing well"))
e2 = FakeBadRequest("x")
e2.body = {"error": {"failed_generation": '{"name": "search_my_notes", "arguments": {"query": "rag"}}'}}
rec = recover(e2, {"search_my_notes"})
check("recover: real tool call with bad format -> tool call", rec and rec["tool_calls"][0]["name"] == "search_my_notes")
check("recover: unrelated error -> None", recover(FakeBadRequest("Error code: 500 - {'error': {}}"), set()) is None)

# ---------------------------------------------------------------- 2. what kind of sentence is it?
CASES = [
    # English
    ("How are you today, Kareem?", "how"), ("Hey Kareem, how are you today?", "how"), ("Hai Kerim", "hello"),
    ("Hallo", "hello"), ("Hi everyone", "hello"), ("Hola, Kari", "hello"), ("Good morning Karim", "hello"),
    ("Hi Karim, can you hear me?", "hear"), ("Thanks Karim!", "thanks"), ("I'm good, thanks", "thanks"),
    ("Karim?", "called"), ("okay", "ack"), ("Can you repeat that?", "repeat"), ("I didn't understand", "rephrase"),
    ("go on", "continue"), ("Hi Karim, what is RAG?", None), ("What is overfitting?", None),
    ("Karim, is that good?", None), ("Karim, how are you using RAG in AgriVision?", None),
    # Egyptian Arabic: greetings, how are you, can you hear me, thanks, acknowledgements, being called
    ("ازيك يا كريم", "how"), ("إزيك يا كريم عامل إيه؟", "how"), ("عامل إيه يا باشا", "how"),
    ("صباح الخير يا جماعة", "hello"), ("صباح الفل يا كريم", "hello"), ("السلام عليكم", "hello"),
    ("السلام عليكم يا كريم", "hello"), ("أهلاً يا كريم", "hello"), ("مساء النور", "hello"),
    ("كريم سامعني؟", "hear"), ("الصوت واضح؟", "hear"), ("سامعني كويس يا كريم", "hear"),
    ("شكراً يا كريم", "thanks"), ("شكراً جداً", "thanks"), ("ميرسي", "thanks"), ("تسلم يا كريم", "thanks"),
    ("تمام", "ack"), ("ماشي يا كريم", "ack"), ("طيب فهمت", "ack"), ("الحمد لله", "ack"),
    ("كريم؟", "called"), ("يا كريم", "called"),
    # Egyptian Arabic: requests about the conversation itself
    ("ممكن تعيد؟", "repeat"), ("معلش مسمعتش", "repeat"), ("تعيد تاني يا كريم", "repeat"),
    ("مش فاهم", "rephrase"), ("مفهمتش يا كريم", "rephrase"), ("وضحها تاني", "rephrase"),
    ("كمل", "continue"), ("كمل يا كريم", "continue"), ("وبعدين؟", "continue"),
    # real questions must NOT be taken for small talk (Egyptian + mixed)
    ("يا كريم، ممكن تشرحلنا الـ overfitting يعني إيه؟", None), ("كريم، إيه الفرق بين الـ RAG والـ fine-tuning؟", None),
    ("إزاي الـ MobileNetV2 بيساعد في AgriVision؟", None), ("عامل إيه في مشروع التخرج يا كريم؟", None),
    ("كمل شرح الـ RAG", None), ("ممكن تعيد الجزء بتاع الـ embedding؟", None),
    ("السلام عليكم يا كريم ممكن تشرح الـ RAG", None), ("يا كريم إزاي بتعمل الـ pruning في الـ decision tree؟", None),
]
for text, want in CASES:
    check(f"kind {text!r} -> {want}", small_talk_kind(text) == want)

# ---------------------------------------------------------------- 3. is the sentence for someone else?
for text, want in [("سارة، ممكن تشاركي الشاشة؟", True), ("يا سارة ممكن تشاركي الشاشة", True),
                   ("Sara, can you share your screen?", True), ("Hello, Sara, how are you?", True),
                   ("أحمد، تعال هنا", True), ("يا أحمد ركز معايا", True),
                   ("طيب يا جماعة، كريم ممكن تشرح؟", False), ("طيب، ممكن تشرح؟", False), ("يا جماعة ركزوا", False),
                   ("تمام، كمل", False), ("بص يا كريم", False), ("Hi everyone", False), ("Hola, Kari", False),
                   ("Ay, KariM", False), ("كريم، ممكن تشرح؟", False)]:
    check(f"other named {text!r} -> {want}", _other_named(text) == want)

# ---------------------------------------------------------------- 4. replies
check("replies: time of day / salam / dialect",
      reply_for("hello", "en", "Good morning Karim").startswith("Good morning")
      and reply_for("hello", "ar", "صباح الخير يا كريم") in REPLIES["ar"]["morning"]
      and reply_for("hello", "ar", "السلام عليكم") in REPLIES["ar"]["salam"]
      and reply_for("hello", "ar", "مساء الخير") in REPLIES["ar"]["evening"]
      and all("التوأم" in r for r in REPLIES["ar"]["how"]))
check("replies: every variant exists in both languages", set(REPLIES["en"]) == set(REPLIES["ar"]))


# ---------------------------------------------------------------- 5. the decision policy (stub for the Phase 1 tree)
def stub_decide(selected="answer"):
    calls = []

    def f(trn, mem):
        calls.append(trn["text"])
        name = tu.name_match(trn["text"])
        sel = selected if name else "stay_silent"           # like the real tree: no name heard -> not for me
        if trn.get("quality", 1) < 0.45:
            sel = "ask_to_repeat"
        br = [{"action": a, "kind": "x", "score": 0.0, "threshold": 0.6, "status": "pruned", "reason": "stub"}
              for a in ("ask_to_repeat", "stay_silent", "defer", "answer")]
        return {"selected": sel, "language": trn.get("language", "en"), "addressed_to_me": bool(name),
                "answer_score": 0.9, "branches": br, "features": {"text": trn["text"], "is_followup": False},
                "llm_reason": "stub", "fallback": False}
    f.calls = calls
    return f


def T(text, q=0.8, lang=None):
    return {"text": text, "quality": q, "language": lang or tu.detect_language(text)}


def memory(last=None, in_window=False):
    return SimpleNamespace(last_said_at=time.monotonic() if in_window else None, last_said=lambda: last or "")


ANSWER = "الـ RAG بيجيب الـ notes المناسبة الأول وبيحطها في الـ prompt عشان الإجابة تبقى مبنية على مصادر."
mem, sd = memory(), stub_decide()
for text, q in [("Hallo", 0.33), ("Hola, Kari", 0.41), ("Hi everyone", 0.8), ("How are you today, Kareem?", 0.9),
                ("Hai Kerim", 0.58), ("Karim?", 0.6), ("صباح الخير", 0.5), ("ازيك", 0.45), ("السلام عليكم يا جماعة", 0.7),
                ("كريم سامعني؟", 0.8)]:
    d = live_decide(T(text, q), mem, sd)
    check(f"greeting {text!r} q={q} -> a friendly reply (no 'could you repeat')", d["selected"] == "small_talk")
check("greetings never reached the decision tree", sd.calls == [])
for text in ["Hello, Sara, how are you?", "Sara, can you share your screen?", "سارة، ممكن تشاركي الشاشة؟",
             "يا سارة ممكن تشاركي الشاشة", "Thank you.", "شكراً", "اشتركوا في القناة"]:
    check(f"stays silent: {text!r}", live_decide(T(text, 0.5), mem, sd)["selected"] == "stay_silent")
check("noise without my name -> silent, not 'could you repeat'",
      live_decide(T("It's a thing that's a good thing, you understand?", 0.27), mem, sd)["selected"] == "stay_silent")
check("unclear audio WITH my name -> asks to repeat", live_decide(T("Ay, KariM", 0.41), mem, sd)["selected"] == "ask_to_repeat")
for text in ["Karim, what is overfitting?", "يا كريم، ممكن تشرحلنا الـ overfitting يعني إيه؟",
             "كريم، إيه الفرق بين الـ RAG والـ fine-tuning؟", "طيب يا جماعة، كريم ممكن تشرح الـ pruning؟"]:
    check(f"real question with my name -> answer: {text!r}", live_decide(T(text, 0.85), mem, sd)["selected"] == "answer")
check("question without my name, no conversation -> silent", live_decide(T("What is overfitting?"), mem, sd)["selected"] == "stay_silent")
check("'ممكن تعيد؟' with nothing said yet and no name -> silent", live_decide(T("ممكن تعيد؟"), mem, sd)["selected"] == "stay_silent")
d = live_decide(T("كريم ممكن تعيد؟"), mem, sd)
check("'كريم ممكن تعيد؟' with nothing said yet -> polite small-talk reply", d["selected"] == "small_talk" and d["small_talk"] == "repeat_none")

# --- inside a conversation (the twin has just given a real answer)
mem = memory(ANSWER, True)
sd = stub_decide()
d = live_decide(T("طب وإيه الفرق بين كده والـ fine-tuning؟"), mem, sd)
check("conversation: Egyptian follow-up WITHOUT my name -> answer", d["selected"] == "answer")
check("conversation: decision tree saw my name; the log keeps the real text",
      sd.calls[-1].startswith("Karim, ") and d["features"]["text"].startswith("طب وإيه") and d["features"]["conversation"])
for text in ["and why not fine-tuning?", "I want to understand how the pipeline handles new documents",
             "عايز أفهم الـ pipeline بتتعامل مع documents جديدة إزاي"]:
    mem.last_said_at = time.monotonic()
    check(f"conversation: reply/request -> answer: {text!r}", live_decide(T(text), mem, sd)["selected"] == "answer")
for text in ["So today we will cover transformers and attention in a lot of detail for the next hour of class",
             "النهارده هنتكلم عن الـ transformers والـ attention بالتفصيل وهناخد ساعة كاملة في الموضوع ده"]:
    mem.last_said_at = time.monotonic()
    d = live_decide(T(text), mem, sd)
    check(f"conversation: a long lecture sentence is NOT for me: {text[:30]!r}",
          d["selected"] == "stay_silent" and mem.last_said_at is None)
for text in ["okay", "تمام", "ماشي", "طيب فهمت"]:
    mem.last_said_at = time.monotonic()
    d = live_decide(T(text), mem, sd)
    check(f"conversation: {text!r} ends it quietly", d["selected"] == "stay_silent" and mem.last_said_at is None)
for text in ["Thanks!", "شكراً يا كريم", "تمام شكراً"]:
    mem.last_said_at = time.monotonic()
    d = live_decide(T(text), mem, sd)
    check(f"conversation: {text!r} -> you're welcome", d["selected"] == "small_talk" and d["small_talk"] == "thanks")

mem = memory(ANSWER, True)
d = live_decide(T("ممكن تعيد؟"), mem, sd)
reply, _ = live_canned(d, T("ممكن تعيد؟"), mem)
check("'ممكن تعيد؟' -> repeats my last answer word for word", d["small_talk"] == "repeat" and reply.reply_text == ANSWER)
for text, flag in [("مش فاهم", "rephrase"), ("وضحها تاني", "rephrase"), ("I didn't understand", "rephrase"),
                   ("كمل", "continue"), ("وبعدين؟", "continue"), ("go on", "continue")]:
    mem.last_said_at = time.monotonic()
    d = live_decide(T(text), mem, sd)
    check(f"conversation: {text!r} -> answer with the '{flag}' instruction",
          d["selected"] == "answer" and d["features"].get(flag) is True and d["answer_score"] > 0)
mem = memory("Hi! I'm Karim's AI twin. How can I help you?", True)
d = live_decide(T("ممكن تعيد؟"), mem, sd)
check("after a greeting there is nothing to repeat -> polite reply, not a made-up answer",
      d["selected"] == "small_talk" and d["small_talk"] == "repeat_none")
mem = memory(ANSWER, True)
mem.last_said_at = time.monotonic() - WINDOW_S - 1
check("the conversation window expires", live_decide(T("طب وإيه الفرق؟"), mem, sd)["selected"] == "stay_silent")

# ---------------------------------------------------------------- 6. the answer step: hints, and crashes never mean silence
seen = {}
answer_loop.make_groq_think = lambda: (lambda goal, ctx: {"content": "", "tool_calls": []})
real_answer = answer_loop.answer
answer_loop.answer = lambda tr_, d_, m_, think_fn=None: (seen.update(text=tr_["text"]) or (None, {}))
la._think = None
sys.modules.setdefault("tools", SimpleNamespace(_TOOLS={}))
last = memory(ANSWER, True)
for feats, needle in [({"rephrase": True}, "simpler words"), ({"continue": True}, "ONE more useful detail"),
                      ({"conversation": True}, "do NOT repeat my previous"), ({"is_followup": True}, "do NOT repeat my previous")]:
    live_answer(T("x"), {"language": "ar", "features": feats}, last)
    check(f"answer hint for {list(feats)[0]}", needle in seen["text"])
live_answer(T("what is RAG?"), {"language": "en", "features": {}}, last)
check("no hint for a fresh question", "(" not in seen["text"])
live_answer(T("x"), {"language": "ar", "features": {"conversation": True}}, memory("أهلاً بيك! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟"))
check("no hint right after a greeting", "do NOT repeat" not in seen["text"])


def boom(*a, **k):
    raise RuntimeError("groq is down")


answer_loop.answer = boom
reply, trace = live_answer(T("يا كريم، اشرح الـ overfitting"), {"language": "ar", "features": {}}, last)
check("any crash -> spoken fallback, never silence", trace["termination_reason"] == "error" and reply.action == "defer"
      and reply.reply_text == la.lc.FALLBACKS["ar"])
answer_loop.answer = real_answer

print("\nALL GOOD" if ok else "\nPROBLEM")
sys.exit(0 if ok else 1)
