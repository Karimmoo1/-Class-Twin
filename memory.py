"""Rolling meeting memory: what was heard + what the twin already said."""
from config import MEMORY_TURNS


class Memory:
    def __init__(self, max_turns: int = MEMORY_TURNS):
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        self.turns: list[tuple[str, str]] = []   # ("heard" | "me", text)
        self.last_speaker = None

    def add_heard(self, text: str):
        if text.strip():
            self._add("heard", text.strip())

    def add_said(self, text: str):
        if text.strip():
            self._add("me", text.strip())

    def _add(self, who, text):
        self.turns.append((who, text))
        self.turns = self.turns[-self.max_turns * 2:]
        self.last_speaker = who

    def last_said(self) -> str:
        for who, t in reversed(self.turns):
            if who == "me":
                return t
        return ""

    def context(self) -> str:
        if not self.turns:
            return "(nothing yet)"
        return "\n".join(f"{'ME (twin)' if w == 'me' else 'CLASS'}: {t}" for w, t in self.turns)
