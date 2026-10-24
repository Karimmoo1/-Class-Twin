"""Offline tests for the harakat step (no Groq, no voice).   python test_tashkeel.py"""
import sys
from types import SimpleNamespace

import tashkeel as tk
from speaker import Speaker

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print("OK  " if cond else "FAIL", name)


calls = []


def good_llm(system, user):                      # a model that adds harakat correctly
    calls.append(user)
    return {"text": user.replace("إيه", "إِيه").replace("أنا", "أَنَا").replace("بتاع", "بِتَاع")}


def bad_llm(system, user):                       # a model that CHANGES a word: must be rejected
    return {"text": user.replace("كريم", "كَرِيمة")}


def none_llm(system, user):
    return None


def boom_llm(system, user):
    raise RuntimeError("groq down")


tk._cache = {}                                   # no disk cache in tests
tk._save = lambda: None
T = "أنا التوأم الذكي بتاع كريم، أقدر أساعدك بإيه؟"

tk.MODE = "off"
check("off: text unchanged, no model call", tk.diacritize(T, good_llm) == T and not calls)
tk.MODE = "llm"
r = tk.diacritize(T, good_llm)
check("on: harakat added, words unchanged", r != T and tk.strip_marks(r) == tk.strip_marks(T) and "أَنَا" in r)
n = len(calls)
check("cached: second call does not ask the model", tk.diacritize(T, good_llm) == r and len(calls) == n)
check("model changed a word -> rejected, plain text spoken", tk.diacritize("يا كريم", bad_llm) == "يا كريم")
check("model returned nothing -> plain", tk.diacritize("ازيك يا كريم", none_llm) == "ازيك يا كريم")
check("model crashed -> plain, no exception", tk.diacritize("شكراً جداً", boom_llm) == "شكراً جداً")
check("English only -> untouched, no model call", tk.diacritize("Hello there", good_llm) == "Hello there" and len(calls) == n)
mixed = "الـ RAG بيجيب الـ notes إيه"
r = tk.diacritize(mixed, good_llm)
check("mixed text: English words preserved", "RAG" in r and "notes" in r and tk.strip_marks(r) == tk.strip_marks(mixed))
check("for_speech: English reply never goes to the model", tk.for_speech("Hello", "en") == "Hello")

# the speaker gives the voice the harakat text, but history and screen keep the clean text
seen = {}


class FakeVoice:
    def stream(self, text, lang, cancel=None):
        seen["text"] = text
        return iter([])


class FakePlayer:
    def play(self, chunks, on_first_audio=None):
        list(chunks)
        return {"played_s": 1.0, "stopped": False, "seconds": 0.1}


controls = SimpleNamespace(muted=False, stop_event=SimpleNamespace(clear=lambda: None), set_state=lambda s: None)
tk._default_llm = good_llm
tk._cache.clear()
sp = Speaker(FakeVoice(), FakePlayer(), controls, printer=lambda *_: None)
sp.say("أنا مهندس كريم", "ar")
check("speaker: voice got harakat, history keeps the clean text",
      "أَنَا" in seen["text"] and sp.history[-1]["text"] == "أنا مهندس كريم")
tk.MODE = "off"
sp.say("أنا مهندس كريم", "ar")
check("speaker: with harakat off the voice gets the text as is", seen["text"] == "أنا مهندس كريم")

print("\nALL GOOD" if ok else "\nPROBLEM")
sys.exit(0 if ok else 1)
