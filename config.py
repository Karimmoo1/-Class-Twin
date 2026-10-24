"""Central settings. Everything you may want to tune lives here."""
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("TWIN_DATA", ROOT / "twin_data"))
OUT_DIR = Path(os.getenv("TWIN_OUT", ROOT / "outputs"))
(OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tests").mkdir(parents=True, exist_ok=True)

# models
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_REASONING = "low"
EMBED_MODEL = "intfloat/multilingual-e5-small"
# hints Whisper about names and terms (helps spelling of Karim / كريم and technical words)
WHISPER_PROMPT = "كريم, Karim, RAG, fine-tuning, embedding, decision tree, LLM, FAISS, AgriVision"

# decision policy
QUALITY_MIN = 0.45          # below this the audio is "unclear" -> ask_to_repeat
DECISION_THRESHOLD = 0.60   # a branch must score at least this to survive pruning
OPEN_Q_ANSWER_MIN = 0.80    # stricter bar for answering an open question to the class

# language detection (script based, see textutils.detect_language)
MIXED_MIN_LATIN_WORDS = 3
MIXED_MIN_LATIN_SHARE = 0.25
NAME_MATCH_RATIO = 0.80     # fuzzy match for my name in the transcript

MEMORY_TURNS = 8

# edge-tts voices (Karim is male, so Shakir for Arabic)
VOICES = {
    "ar": "ar-EG-ShakirNeural",
    "mixed": "ar-EG-ShakirNeural",
    "en": "en-US-AndrewNeural",
}


@lru_cache(maxsize=1)
def load_profile() -> dict:
    return json.loads((DATA_DIR / "profile.json").read_text(encoding="utf-8"))
