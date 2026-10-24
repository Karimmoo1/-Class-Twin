"""Harakat (تشكيل) for the VOICE only. Whisper writes Arabic without short vowels, and the voice (XTTS) pronounces
from the text it is given, so it has to guess the vowels. Here the reply gets harakat just before it is spoken.
The text shown on screen, in the log and in the memory stays clean.

How: one small Groq call per reply, asked for light, Egyptian-friendly harakat (no formal case endings, English
words untouched). Safety: the answer is accepted ONLY if, after removing the harakat, it is exactly the same text
(no word added, removed or changed). Otherwise the plain text is spoken. Results are cached on disk, so a sentence
is diacritized once.

Switch:  TWIN_TASHKEEL=llm  (or  python run_live.py --tashkeel llm).  Default: off, until you listened to the A/B test:
         python tashkeel_ab_test.py
Honest limit: harakat fix vowels. They cannot turn ج into the Egyptian "g" or ق into the glottal stop, because
the XTTS Arabic model speaks standard Arabic consonants.
"""
import json
import os
import re
import threading

import live_config as lc

MODE = os.getenv("TWIN_TASHKEEL", "off")          # "off" | "llm"
_AR = re.compile(r"[\u0621-\u064A]")
_MARKS = re.compile(r"[\u064B-\u0652\u0670\u0640]")   # fatha, damma, kasra, tanween, shadda, sukun, dagger alef, tatweel
_FILE = lc.LIVE_DIR / "tashkeel_cache.json"
_lock = threading.Lock()
_cache = None

SYSTEM = (
    "You add Arabic diacritics (harakat) to Egyptian Arabic text so that a text-to-speech engine pronounces it "
    "correctly the way an Egyptian speaker talks. Rules: (1) put short vowels (fatha, damma, kasra), sukun and "
    "shadda on every Arabic word; (2) NO formal case endings (no final i'rab vowels), except fixed forms like "
    "شكراً / أهلاً; (3) vowel dialect words as Egyptians say them (بتاع → بِتَاع، ده → دَه، إيه → إِيه، "
    "ازاي → إِزَّاي) and standard words as in standard Arabic (الذكي → الذَّكِي); (4) keep English/Latin words, "
    "numbers and punctuation exactly as they are; (5) do NOT add, remove, reorder or change any word or letter, only "
    "add diacritics. Output JSON only: {\"text\": \"...\"}."
)


def strip_marks(s: str) -> str:
    return re.sub(r"\s+", " ", _MARKS.sub("", s)).strip()


def has_arabic(s: str) -> bool:
    return bool(_AR.search(s))


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def _save():
    try:
        _FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception:
        pass


def _default_llm(system: str, user: str):
    from llm import llm_json
    return llm_json(system, user)


def _ask(text: str, llm) -> str | None:
    out = llm(SYSTEM, text) or {}
    cand = str(out.get("text", "")).strip()
    if cand and strip_marks(cand) == strip_marks(text) and _MARKS.search(cand):
        return cand
    return None


def diacritize(text: str, llm=None) -> str:
    """Return the text with harakat, or the original text if anything is off."""
    if MODE == "off" or not has_arabic(text):
        return text
    llm = llm or _default_llm
    with _lock:
        hit = _load().get(text)
    if hit:
        return hit
    try:
        res = _ask(text, llm)
        if res is None and len(text) > 60:                     # whole reply failed the check: do it sentence by sentence
            parts, ok = re.split(r"(?<=[.!?؟،؛])\s+", text.strip()), []
            for p in parts:
                ok.append(diacritize(p, llm) if has_arabic(p) else p)
            res = " ".join(ok) if any(_MARKS.search(x) for x in ok) else None
    except Exception:
        res = None                                             # Groq down, rate limit, anything: speak it plain
    if res:
        with _lock:
            _load()[text] = res
            _save()
        return res
    return text


def for_speech(text: str, lang: str) -> str:
    return text if lang == "en" else diacritize(text)
