"""Live-meeting layer on top of the Phase 2 answer path (decision policy, small talk, conversation, safety net).

WHAT IT DOES
 - Greetings ("hi", "hello", "how are you", "can you hear me", "good morning", "السلام عليكم", "ازيك") get a short
   friendly reply from the AI twin, even without my name and even if Whisper was unsure. No LLM, so it is instant.
 - Conversation: after the twin speaks, the next sentence (a question, a request or a short reply) is for the twin
   even without my name, for WINDOW_S seconds. Pure acknowledgements ("okay", "تمام") end the conversation quietly.
 - Egyptian Arabic and mixed Arabic/English are first-class: letters are normalized (أ/ا ة/ه ى/ي), greetings, thanks,
   requests ("ممكن تعيد؟", "مش فاهم", "كمل") and vocatives without يا ("سارة، ممكن...") are understood.
 - Interjections ("Hola, Kari", "Ay, KariM", "Hey, Karim") are no longer mistaken for another person's name.
 - The twin never says "could you repeat?" to noise nobody addressed to it, and ignores Whisper's invented
   "Thank you." on silence.
 - Follow-ups are told to build on the previous answer instead of repeating it.
 - Problems found in the first live run (still handled here):

 1. gpt-oss on Groq sometimes finishes by "calling a tool" named JSON/json to hand in its final reply.
    Groq rejects that with a 400 (tool_use_failed), but the intended reply is inside `failed_generation`.
    -> recover() reads it from there, so the reply is not lost.
 2. Small talk ("how are you today?", "hello", "can you hear me?", "thanks") has no note and no source,
    so the twin deferred or crashed. -> a short fixed reply (it also says it is the AI twin).
 3. Any other crash in the answer path used to leave the instructor in silence.
    -> the twin now speaks the safe fallback ("Let me check that and get back to you").

run_live.py passes live_decide / live_answer / live_canned to the meeting loop. Tests:  python test_live_layer.py
"""
import ast
import json
import random
import re
import time

import answer_loop
import live_config as lc
import textutils as tu
from schemas import TwinReply


# ------------------------------------------------------------------ 1. recover the reply from a 400 error
def _failed_generation(e):
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("failed_generation"):
            return err["failed_generation"]
    try:
        d = ast.literal_eval(str(e).split(" - ", 1)[1])
        return d["error"].get("failed_generation")
    except Exception:
        return None


def recover(e, tool_names):
    """-> an assistant message {"content", "tool_calls"} rebuilt from failed_generation, or None."""
    gen = _failed_generation(e)
    if not gen:
        return None
    try:
        obj = json.loads(gen)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", gen, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return None
    name, args = obj.get("name"), obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    if name in tool_names and isinstance(args, dict):                 # a real tool call with a format problem
        return {"content": "", "tool_calls": [{"id": "recovered_0", "name": name, "args": args}]}
    if isinstance(args, dict) and "reply_text" in args:               # the final reply wrapped as a fake tool call
        return {"content": json.dumps(args, ensure_ascii=False), "tool_calls": []}
    if "reply_text" in obj:
        return {"content": json.dumps(obj, ensure_ascii=False), "tool_calls": []}
    return None


def robust_think():
    from tools import _TOOLS
    names = set(_TOOLS)
    base = answer_loop.make_groq_think()

    def think(goal, ctx):
        try:
            return base(goal, ctx)
        except Exception as e:
            rec = recover(e, names)
            if rec is None:
                raise
            return rec
    return think




# ------------------------------------------------------------------ settings
GREET_WITHOUT_NAME = True     # reply to "hi" / "hello everyone" even if my name was not said
GREET_MIN_Q = 0.25            # greetings are short: accept lower Whisper confidence than for questions
WINDOW_S = 45.0               # after the twin speaks, the next sentence is for the twin for this long


# ------------------------------------------------------------------ text helpers (Egyptian Arabic + English)
def _n(text: str) -> str:
    """Normalize for matching: lower case, no tashkeel/tatweel, أإآ->ا, ى->ي, ة->ه, punctuation -> space."""
    t = tu.norm(text.replace("’", "'"))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t)).strip()


def _alt(*phrases: str) -> str:
    return "|".join(sorted((_n(p) for p in phrases), key=len, reverse=True))


_HEAR = _alt("can you hear me", "can you hear us", "do you hear me", "are you there", "you there",
             "سامعني", "سامعنا", "سامعين", "سامعني كويس", "الصوت واضح", "الصوت واصل", "صوتي واصل", "صوتي واضح",
             "انت معانا", "انتو معانا", "معانا", "موجود")
_REPEAT = _alt("can you repeat", "repeat that", "repeat please", "say that again", "say it again", "come again",
               "pardon", "sorry what", "didn't catch that", "once more", "ممكن تعيد", "تعيد تاني", "اعد", "اعيد",
               "كرر", "كررها", "قول تاني", "قولها تاني", "قولنا تاني", "مسمعتش", "ما سمعتش", "معلش تاني")
_REPHRASE = _alt("explain again", "rephrase", "i don't understand", "i didn't understand", "i did not understand",
                 "simpler", "مفهمتش", "ما فهمتش", "مش فاهم", "مش فاهمين", "مش واضح", "وضحها تاني", "اشرحها تاني",
                 "اشرح تاني", "بسطها", "ابسط", "بسط")
_CONTINUE = _alt("go on", "continue", "keep going", "go ahead", "tell me more", "more details", "elaborate", "and then",
                 "كمل", "كمل كلامك", "كملي", "وبعدين", "قول اكتر", "وضح اكتر", "زود", "اتكلم اكتر")
_HOW = _alt("how are you", "how's it going", "how are things", "how do you do", "what's up", "whats up",
            "how have you been", "how you doing", "ازيك", "ازيكم", "عامل ايه", "عاملين ايه", "عامله ايه", "اخبارك",
            "اخباركم", "ايه الاخبار", "ايه اخبارك", "الدنيا عامله ايه", "ايه الدنيا")
_THANKS = _alt("thank you", "thanks", "thx", "thank u", "شكرا", "شكرا جزيلا", "مشكور", "متشكر", "ميرسي", "تسلم",
               "تسلم ايدك", "كتر خيرك", "الله يكرمك", "ربنا يكرمك", "ربنا يخليك", "جزاك الله خير", "يسلمو")
_HELLO = _alt("hello", "hallo", "hi", "hai", "hey", "hola", "allo", "good morning", "good afternoon", "good evening",
              "salam", "marhaba", "مرحبا", "اهلا", "اهلين", "اهلا وسهلا", "يا هلا", "هاي", "هلا", "السلام عليكم",
              "سلام عليكم", "صباح الخير", "صباح الفل", "صباح النور", "مساء الخير", "مساء الفل", "مساء النور",
              "نورتنا", "نورت")
_PATTERNS = (("hear", _HEAR), ("repeat", _REPEAT), ("rephrase", _REPHRASE), ("continue", _CONTINUE),
             ("how", _HOW), ("thanks", _THANKS), ("hello", _HELLO))

_NAMES = {_n(x) for x in ("karim", "kareem", "kerim", "kari", "carim", "kareim", "kariim", "كريم", "كاريم")}
_FILLER = {_n(x) for x in ("today", "there", "everyone", "everybody", "all", "so", "well", "please", "um", "uh",
                           "guys", "you", "again", "too", "also", "class", "that", "this", "يا", "النهاردة", "كلكم", "تاني", "كمان",
                           "و", "جماعة", "شباب", "هندسة", "باشا", "بطل", "لو", "سمحت", "فضلك", "من", "معلش", "بقى",
                           "كده", "طيب", "طب", "بص", "اسمع", "شوف", "يعني", "يلا", "استاذ", "دكتور", "باشمهندس",
                           "مهندس", "فندم", "يافندم", "ياباشا", "جدا", "اوي", "خالص")}
_ACK_WORDS = {_n(x) for x in ("ok", "okay", "fine", "good", "great", "nice", "alright", "sure", "yes", "yeah", "yep",
                              "no", "nope", "cool", "right", "got", "it", "i", "see", "understood", "that's", "thats",
                              "all", "nothing", "i'm", "im", "well", "alhamdulillah", "تمام", "ماشي", "حلو", "كويس",
                              "اه", "أه", "ايوه", "أيوه", "ايوا", "لا", "مفيش", "خلاص", "فاهم", "فهمت", "مفهوم",
                              "واضح", "الحمد", "لله", "الحمدلله", "طيب", "اوكي", "أوكي", "جميل", "عظيم", "ممتاز",
                              "برافو", "صح", "بالظبط", "اكيد", "زي", "الفل", "كده")}
_GROUP = {_n(x) for x in ("everyone", "everybody", "all", "guys", "class", "جماعة", "شباب", "بنات", "كلكم",
                          "هندسة", "باشا", "بطل", "حبايبي", "جميعا")}
_INTERJ = _FILLER | {_n(x) for x in ("ay", "oh", "ah", "hmm", "hola", "hello", "hi", "hey", "hai", "hallo", "allo",
                                     "okay", "ok", "اه", "ايوه", "ايوا", "اوكي", "تمام", "ماشي", "خلينا", "اولا")}
_STARTERS = {_n(x) for x in ("طيب", "طب", "بص", "اسمع", "شوف", "يعني", "تمام", "ماشي", "اه", "ايوه", "ايوا", "خلاص",
                             "كده", "دلوقتي", "النهارده", "اولا", "ثانيا", "تاني", "بقى", "اوكي", "هاي", "هلا",
                             "اهلا", "مرحبا", "السلام", "صباح", "مساء", "طبعا", "فعلا", "مثلا", "معلش", "بس", "لو",
                             "هو", "هي", "انا", "احنا", "انت", "انتو", "نعم", "لا", "اكيد", "عموما", "اساسا",
                             "ببساطة", "يلا", "الحمد", "شكرا", "ممكن", "لازم", "حاليا", "تعالوا", "اتفضل", "فيه",
                             "ده", "دي", "دول", "بالنسبة", "نبدا", "نكمل", "خلينا", "استاذ", "دكتور", "مهندس",
                             "باشمهندس", "تمام")}
_REQ_EXACT = ("ممكن", "الفرق", "فرق", "عايز", "عاوز", "نفسي", "هل", "كيف", "لماذا", "ماذا", "ايه", "ليه", "ازاي", "مين",
              "فين", "امتي", "كام", "ينفع", "تقدر", "تعرف", "وايه", "والفرق")
_REQ_STEM = ("اشرح", "وضح", "فهمنا", "عرفنا", "قول", "ورين", "قارن", "بتعمل", "بيعمل", "بتستخدم", "استخدمت")  # +suffix: اشرحلنا
_REQUEST = re.compile(
    r"\b(explain|tell me|show me|can you|could you|please|what|why|how|when|which|who|describe|give me|compare|"
    r"difference)\b|(?<!\w)(?:" + "|".join(_n(x) for x in _REQ_EXACT) + r")(?!\w)|(?<!\w)(?:"
    + "|".join(_n(x) for x in _REQ_STEM) + r")\w*", re.I)
_HALLUC = {"thank you", "thanks", "thank you so much", "thanks for watching", "thank you for watching", "bye",
           "bye bye", "you", "the end", "okay", "ok", "شكرا", "شكرا لكم", "شكرا لمشاهدتكم"}


def _lang_key(lang: str) -> str:
    return "en" if lang == "en" else "ar"


def _is_name(w: str) -> bool:
    return w in _NAMES or (len(w) >= 3 and tu.name_match(w) is not None)


# ------------------------------------------------------------------ fixed replies (a few variants, so it does not sound like a robot)
REPLIES = {
    "en": {"hello": ["Hi! I'm Karim's AI twin. How can I help you?", "Hello! Karim's AI twin here. What can I do for you?"],
           "morning": ["Good morning! I'm Karim's AI twin. How can I help you?"],
           "evening": ["Good evening! I'm Karim's AI twin. How can I help you?"],
           "salam": ["Peace be upon you too! I'm Karim's AI twin. How can I help you?"],
           "how": ["I'm doing well, thank you! I'm Karim's AI twin. How can I help?",
                   "Doing great, thanks for asking! I'm Karim's AI twin, go ahead."],
           "hear": ["Yes, I can hear you clearly. Go ahead.", "Yes, loud and clear. Go ahead."],
           "thanks": ["You're welcome! Anything else?", "Happy to help! Anything else?"],
           "called": ["Yes, I'm here. Go ahead.", "Yes, I'm listening."],
           "repeat_none": ["I haven't said anything yet. Go ahead with your question."]},
    "ar": {"hello": ["أهلاً بيك! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟",
                     "أهلاً وسهلاً! معاك التوأم الذكي بتاع كريم. تحب أساعدك في إيه؟"],
           "morning": ["صباح النور! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟", "صباح الفل! معاك التوأم الذكي بتاع كريم. اتفضل."],
           "evening": ["مساء النور! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟", "مساء الفل! معاك التوأم الذكي بتاع كريم. اتفضل."],
           "salam": ["وعليكم السلام ورحمة الله! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟"],
           "how": ["الحمد لله تمام، الله يسلمك! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟",
                   "تمام الحمد لله، شكراً! أنا التوأم الذكي بتاع كريم. اتفضل."],
           "hear": ["أيوه، سامعك كويس. اتفضل.", "أيوه سامعك، الصوت واضح. اتفضل."],
           "thanks": ["العفو! في حاجة تانية؟", "ولا يهمك! أي خدمة تانية؟", "العفو، تحت أمرك."],
           "called": ["أيوه، معاك. اتفضل.", "أيوه يا فندم، سامعك."],
           "repeat_none": ["لسه ماقولتش حاجة. اتفضل بسؤالك."]},
}
_SMALL_TALK_TEXTS = {v for table in REPLIES.values() for variants in table.values() for v in variants}


def reply_for(kind: str, lang: str, text: str = "") -> str:
    t = _n(text)
    if kind == "hello":
        if "السلام عليكم" in t or "سلام عليكم" in t:
            kind = "salam"
        elif "morning" in t or "صباح" in t:
            kind = "morning"
        elif "evening" in t or "مساء" in t:
            kind = "evening"
    return random.choice(REPLIES[_lang_key(lang)][kind])


def small_talk_kind(text: str):
    """'hear' | 'repeat' | 'rephrase' | 'continue' | 'how' | 'thanks' | 'hello' | 'called' (just my name) |
    'ack' (okay / تمام) | None. Only when nothing but that phrase, a name and filler words are left, so
    'Hi Karim, what is RAG?' and 'يا كريم، ممكن تشرح الـ RAG؟' stay real questions."""
    t = _n(text)
    kind, rest = None, t
    for k, pat in _PATTERNS:
        p = rf"(?<!\w)(?:{pat})(?!\w)"
        if re.search(p, t):
            kind = kind or k
            rest = re.sub(p, " ", rest)
    words = [w for w in rest.split() if not _is_name(w) and w not in _FILLER]
    if kind:
        return None if [w for w in words if w not in _ACK_WORDS] else kind
    if not words:
        return "called" if any(_is_name(w) for w in t.split()) else None
    return "ack" if all(w in _ACK_WORDS for w in words) else None


# ------------------------------------------------------------------ who is talking to whom, and the conversation
def _strip_interjections(text: str) -> str:
    """Drop leading interjections: 'Hola, Kari' -> 'Kari', 'طيب يا جماعة، كريم...' -> 'كريم...'.
    'يا' is kept in front of a name, so 'يا كريم' / 'يا سارة' stay vocatives."""
    toks = list(re.finditer(r"\S+", text))
    i = 0
    while i < len(toks):
        w = _n(toks[i].group())
        if w == "يا":
            nxt = _n(toks[i + 1].group()) if i + 1 < len(toks) else ""
            if nxt in _GROUP or nxt in _INTERJ:
                i += 2
                continue
            break
        if w in _INTERJ or not w:
            i += 1
            continue
        break
    return text[toks[i].start():].strip() if i < len(toks) else ""


def _clean(text: str) -> str:
    return _strip_interjections(text)


def _other_named(text: str) -> bool:
    """Is the sentence addressed to someone else by name? 'Sara, ...' / 'يا سارة ...' / 'سارة، ممكن ...'"""
    if tu.name_match(text):
        return False                                   # my name is in it: it is for me
    rest = _strip_interjections(text)
    m = re.match(r"يا\s+([\u0621-\u064A]+)", rest)
    if m:
        v = _n(m.group(1))
        return v not in _GROUP and v not in _INTERJ and tu.name_match(v) is None
    m = re.match(r"([A-Za-z]{2,})\s*[,:]", rest)
    if m:
        v = m.group(1).lower()
        return v not in _GROUP and v not in _INTERJ and tu.name_match(v) is None
    m = re.match(r"([\u0621-\u064A]{3,})\s*[،,:]", rest)    # Egyptian instructors: 'سارة، ممكن تشاركي الشاشة؟'
    if m:
        v = _n(m.group(1))
        return v not in _GROUP and v not in _STARTERS and v not in _INTERJ and tu.name_match(v) is None
    return False


def conversation_active(memory) -> bool:
    t = getattr(memory, "last_said_at", None)
    return t is not None and time.monotonic() - t <= WINDOW_S


def _close_conversation(memory):
    if hasattr(memory, "last_said_at"):
        memory.last_said_at = None


def _last_answer(memory) -> str:
    """What I last said, if it was a real answer (a greeting is not something to repeat or build on)."""
    last = getattr(memory, "last_said", lambda: "")() or ""
    return "" if last in _SMALL_TALK_TEXTS else last


def _looks_like_reply(text: str) -> bool:
    """Inside a conversation: a question, a request or a short reply is for me. A long statement is not."""
    return bool(tu.is_question(text) or _REQUEST.search(_n(text)) or len(tu.words(text)) <= 6)


def _synthetic(tr, selected, kind=None, reason="", lang=None, addressed=True):
    """A decision made by a rule (no LLM). Same shape as Phase 1 decide() so the meeting loop and the log work."""
    lang = lang or tr.get("language") or "en"
    branches = [{"action": a, "kind": "rule", "score": 0.0, "threshold": 0.6, "status": "pruned",
                 "reason": "not scored: a rule decided"} for a in ("ask_to_repeat", "stay_silent", "defer", "answer")]
    for b in branches:
        if b["action"] == selected:
            b.update(score=1.0, status="selected", reason=reason)
    d = {"features": {"text": tr["text"], "quality": tr.get("quality"), "language": lang, "rule": reason},
         "selected": selected, "fallback": False, "language": lang, "addressed_to_me": addressed,
         "answer_score": 1.0 if selected == "small_talk" else 0.0, "llm_reason": reason, "notes_preview": [],
         "branches": branches}
    if kind:
        d["small_talk"] = kind
    return d


def _to_silent(d, why):
    d["selected"] = "stay_silent"
    d["llm_reason"] = (d.get("llm_reason") or "") + f" | {why}"
    for b in d["branches"]:
        b["status"] = "selected" if b["action"] == "stay_silent" else "pruned"
        if b["action"] == "stay_silent":
            b["score"], b["reason"] = 0.9, why
    return d


def live_decide(tr, memory, decide_fn=None):
    """The live decision policy. Rules first (instant), then the Phase 1 decision tree."""
    text = tr["text"].strip()
    q = tr.get("quality") or 0.0
    lang = tr.get("language") or tu.detect_language(text)
    name = tu.name_match(text)
    other = _other_named(text)
    in_win = conversation_active(memory)
    kind = small_talk_kind(text)
    addressed = bool(name) or (in_win and not other and _looks_like_reply(text))
    last = _last_answer(memory)

    # 1) greetings and the like: a fixed friendly reply, even without my name, even if Whisper was unsure
    if kind in ("hello", "how", "hear") and not other and q >= GREET_MIN_Q and (name or in_win or GREET_WITHOUT_NAME):
        return _synthetic(tr, "small_talk", kind, f"greeting ({kind}): fixed reply, no notes needed", lang)
    if kind == "thanks" and not other and (name or in_win):
        return _synthetic(tr, "small_talk", kind, "thanks: fixed reply", lang)
    if kind == "called" and name and q >= GREET_MIN_Q:
        return _synthetic(tr, "small_talk", kind, "my name was called: I am here", lang)
    if kind == "ack" and in_win:                       # "okay", "تمام": the conversation is over, stay quiet
        _close_conversation(memory)
        return _synthetic(tr, "stay_silent", None, "acknowledgement: nothing to add", lang, addressed=False)

    # 2) conversation requests: "can you repeat?" / "مش فاهم" / "كمل"
    if kind in ("repeat", "rephrase", "continue") and not other and q >= GREET_MIN_Q:
        if in_win and last:
            if kind == "repeat":
                return _synthetic(tr, "small_talk", "repeat", "repeat my last answer", lang)
            d = _synthetic(tr, "answer", None, f"{kind} request: build on my last answer", lang)
            d["answer_score"] = 0.9
            d["features"].update(conversation=True, is_followup=True, **{kind: True})
            return d
        if in_win or name:                             # nothing real to repeat or build on yet
            return _synthetic(tr, "small_talk", "repeat_none" if kind == "repeat" else "called",
                              "request, but I have not given an answer yet", lang)

    # 3) Whisper invents "Thank you." / "شكرا" / "Bye" on silence and noise
    if _n(text) in _HALLUC and not name and not in_win and q < 0.75:
        return _synthetic(tr, "stay_silent", None, "probably a Whisper hallucination on noise", lang, addressed=False)

    # 4) the Phase 1 decision tree. Inside a conversation the next sentence is for me even without my name.
    if decide_fn is None:
        from decide import decide as decide_fn
    cleaned = _clean(text) or text
    tr_dec = dict(tr, text=("Karim, " + cleaned) if (addressed and not name) else cleaned)
    d = decide_fn(tr_dec, memory)
    d["features"]["text"] = text
    d["features"]["conversation"] = bool(in_win and addressed and not name)

    # 5) never interrupt for noise: "could you repeat?" only if somebody was talking to me
    if d["selected"] == "ask_to_repeat" and not addressed:
        _to_silent(d, "unclear audio and nobody called me: stay quiet")
    if d["selected"] == "stay_silent" and in_win:
        _close_conversation(memory)                    # the twin stayed quiet: the conversation is over
    return d


def live_canned(decision, tr, memory):
    """Replies that need no LLM: small talk, repeating my last answer, and the Phase 1 safe replies."""
    kind = decision.get("small_talk")
    if kind:
        lang = decision["language"]
        text = (_last_answer(memory) or reply_for("repeat_none", lang)) if kind == "repeat" \
            else reply_for(kind, lang, tr["text"])
        reply = TwinReply(addressed_to_me=True, action="answer", language=lang, reply_text=text,
                          confidence=0.9, tools_used=[], sources=[])
        return reply, {}
    from planner import reply as planner_reply
    return planner_reply(decision, tr, memory)


# ------------------------------------------------------------------ the real answer, never silence
_think = None
_HINTS = {
    "continue": ("\n(The instructor asked me to continue: add ONE more useful detail from my notes about the same "
                 "topic. Do NOT repeat what I already said.)"),
    "rephrase": ("\n(The instructor did not understand: explain my previous answer again in simpler words with one "
                 "tiny practical example. Keep it short.)"),
    "conversation": ("\n(We are in the middle of a conversation: build on what I just said, do NOT repeat my "
                     "previous answer, answer only what is new.)"),
}


def live_answer(tr, decision, memory):
    global _think
    lang = decision["language"]
    feats = decision.get("features") or {}
    hint = _HINTS["continue"] if feats.get("continue") else _HINTS["rephrase"] if feats.get("rephrase") else None
    if hint is None and (feats.get("is_followup") or (feats.get("conversation") and _last_answer(memory))):
        hint = _HINTS["conversation"]
    if hint:
        tr = dict(tr, text=tr["text"] + hint)
    try:
        if _think is None:
            _think = robust_think()
        return answer_loop.answer(tr, decision, memory, think_fn=_think)
    except AssertionError:
        raise
    except Exception as e:                              # model server down, bad request, anything
        reply = TwinReply(addressed_to_me=True, action="defer", language=lang,
                          reply_text=lc.FALLBACKS["en" if lang == "en" else "ar"], confidence=0.0,
                          tools_used=[], sources=[])
        return reply, {"termination_reason": "error", "error": f"{type(e).__name__}: {str(e)[:300]}",
                       "iterations": 0, "retries": 0, "goal_check": None, "seconds": 0.0, "steps": [],
                       "failed_calls": [], "seen_sources": []}
