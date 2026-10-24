"""Meeting memory that does not grow forever: last ~10 turns word for word, older turns summarized."""
import threading
import time

import live_config as lc
from memory import Memory


def _extractive(prev: str, old: list) -> str:
    """Fallback when the LLM summary fails: keep the start of every old turn."""
    bits = " | ".join(f"{'me' if w == 'me' else 'class'}: {t[:80]}" for w, t in old)
    return ((prev + " | " if prev else "") + bits)[-700:]


def llm_summarizer(prev: str, old: list) -> str:
    from llm import llm_json
    lines = "\n".join(f"{'ME (twin)' if w == 'me' else 'CLASS'}: {t}" for w, t in old)
    out = llm_json(
        "You summarize a class meeting for a note-taker. Output JSON only: {\"summary\": str}. "
        "At most 70 words. Keep topics, names and what Karim (ME) already said. Same language as the text.",
        f"EARLIER SUMMARY: {prev or '(none)'}\n\nNEW LINES:\n{lines}") or {}
    s = str(out.get("summary", "")).strip()
    if not s:
        raise ValueError("empty summary")
    return s


class LiveMemory(Memory):
    """Same interface as Phase 1 Memory (decide() and the answer loop use it unchanged)."""

    def __init__(self, verbatim=lc.VERBATIM_TURNS, batch=lc.SUMMARIZE_BATCH, summarizer=llm_summarizer,
                 background=True):
        self._lock = threading.RLock()
        self.verbatim, self.batch, self.summarizer, self.background = verbatim, batch, summarizer, background
        self._thread = None
        super().__init__(max_turns=10 ** 6)      # we trim ourselves

    def reset(self):
        super().reset()
        self.summary = ""
        self._inflight = []
        self.last_said_at = None          # when the twin last spoke (the conversation window starts here)

    def _add(self, who, text):
        with self._lock:
            super()._add(who, text)
            if who == "me":
                self.last_said_at = time.monotonic()
        self._maybe_summarize()

    def _maybe_summarize(self):
        with self._lock:
            n = len(self.turns) - self.verbatim
            if self._inflight or n < self.batch:
                return
            self._inflight, self.turns = self.turns[:n], self.turns[n:]
            old, prev = list(self._inflight), self.summary

        def work():
            try:
                s = self.summarizer(prev, old)
            except Exception:
                s = _extractive(prev, old)
            with self._lock:
                self.summary, self._inflight = s, []

        if self.background:
            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
        else:
            work()

    def flush(self):
        if self._thread:
            self._thread.join()

    def context(self) -> str:
        with self._lock:
            parts = []
            if self.summary:
                parts.append(f"EARLIER (summary): {self.summary}")
            if self._inflight:
                parts.append("EARLIER: " + " / ".join(t[:120] for _, t in self._inflight))
            parts += [f"{'ME (twin)' if w == 'me' else 'CLASS'}: {t}" for w, t in self.turns]
            return "\n".join(parts) if parts else "(nothing yet)"
