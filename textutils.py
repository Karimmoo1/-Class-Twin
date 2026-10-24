"""Cheap text checks (no LLM): language, my name, vocatives, questions."""
import difflib
import re
from functools import lru_cache

import config

_DIAC = re.compile("[\u064B-\u0652\u0640]")           # tashkeel + tatweel
_WORD = re.compile(r"[\w\u0621-\u064A]+(?:-\w+)*")     # keeps fine-tuning as one word
_AR = re.compile(r"[\u0621-\u064A]")
_LAT = re.compile(r"[a-z]")

_NOT_NAMES = {"okay", "ok", "so", "well", "now", "alright", "right", "yes", "no", "hello",
              "hi", "hey", "also", "and", "but", "actually", "anyway", "first", "next",
              "today", "everyone", "anyone", "guys", "class", "جماعه", "شباب", "بنات",
              "دكتور", "استاذ", "ريس"}

_OPEN = ["anyone", "anybody", "any one", "everyone", "everybody", "who can", "who knows",
         "who wants", "guys", "حد ", "اي حد", "حد منكم", "حد يعرف", "يا جماعه", "يا شباب",
         "مين يعرف", "مين يقدر"]

_Q_EN = re.compile(r"\b(what|why|how|when|where|which|who|can you|could you|would you|do you|"
                   r"does|is it|are you|explain|tell us|difference)\b", re.I)
_Q_AR = {"ايه", "ليه", "ازاي", "ممكن", "هل", "مين", "فين", "امتي", "كام", "اشرح", "فرق", "الفرق"}

_FOLLOW_EN = re.compile(r"\b(that|earlier|before|you said|previous|mentioned|as well)\b", re.I)
_FOLLOW_AR = ["قبل كده", "قلته", "قلت", "اللي فات", "اللي قلته"]


def norm(text: str) -> str:
    t = _DIAC.sub("", text.lower())
    t = re.sub("[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return t


def words(text: str) -> list[str]:
    return _WORD.findall(norm(text))


@lru_cache(maxsize=1)
def name_tokens() -> frozenset:
    p = config.load_profile()
    toks = set()
    for n in p.get("name_spellings", []) + [p.get("first_name", "")]:
        for w in words(n):
            if len(w) >= 3:
                toks.add(w)
    return frozenset(toks)


def name_match(text: str):
    """Return the transcript word that matches my name (fuzzy), else None."""
    for w in words(text):
        if len(w) < 3:
            continue
        for n in name_tokens():
            if w == n or difflib.SequenceMatcher(None, w, n).ratio() >= config.NAME_MATCH_RATIO:
                return w
    return None


def detect_language(text: str) -> str:
    """'ar' / 'en' / 'mixed' from the script of the words (more reliable than Whisper's label
    for code-switched Egyptian Arabic). My own name is ignored so 'Karim' in an Arabic sentence
    does not count as English."""
    names = name_tokens()
    ws = [w for w in words(text) if w not in names]
    ar = [w for w in ws if _AR.search(w)]
    la = [w for w in ws if _LAT.search(w)]
    if not ar:
        return "en"
    if not la:
        return "ar"
    share = len(la) / len(ws)
    if len(la) >= config.MIXED_MIN_LATIN_WORDS or share >= config.MIXED_MIN_LATIN_SHARE:
        return "mixed"
    return "ar"


def find_vocative(text: str):
    """The person being called at the start of the sentence ('Sara, ...' / 'يا سارة ...')."""
    t = text.strip()
    m = re.match(r"^(?:(?:hey|hi|hello|ok|okay)\s+)?([A-Za-z]{2,})\s*[,،:]", t, re.I)
    v = m.group(1) if m else None
    if not v:
        m = re.match(r"^\s*يا\s+([\u0621-\u064A]+)", t)
        v = m.group(1) if m else None
    if v and norm(v) in _NOT_NAMES:
        return None
    return v


def is_open_question(text: str) -> bool:
    t = norm(text) + " "
    return any(m in t for m in _OPEN)


def is_question(text: str) -> bool:
    if "?" in text or "؟" in text or _Q_EN.search(text):
        return True
    ws = set(words(text))
    return bool(ws & _Q_AR) or "يعني ايه" in norm(text)


def is_followup(text: str) -> bool:
    t = norm(text)
    return bool(_FOLLOW_EN.search(text)) or any(m in t for m in _FOLLOW_AR)


def make_query(question: str, last_said: str) -> str:
    """Follow-ups ('how is that different from what you said earlier?') carry no topic,
    so we add what I said last to the search query."""
    if last_said and is_followup(question):
        return f"{question} {last_said[:200]}"
    return question
