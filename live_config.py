"""Phase 2 settings. Phase 1 settings stay in config.py."""
import os

import config

# ---- meeting loop (outer) ----
MAX_MEETING_MIN = 60              # hard stop
IDLE_MIN = 5                      # stop after this many minutes with no speech
MODE_DEFAULT = "approve"          # "approve" (you press Enter) or "auto"
APPROVE_TIMEOUT_S = 25            # no key pressed -> the draft is dropped (safe default)
FILLER_IN_APPROVE_MODE = False    # in approve mode a filler would be spoken before you approve

# ---- audio + VAD ----
SAMPLE_RATE = 16000
CHUNK = 512                       # silero-vad needs exactly 512 samples at 16 kHz (32 ms)
VAD_THRESHOLD = 0.5
SILENCE_MS = 700                  # an utterance ends after this much silence
MIN_SPEECH_MS = 150               # speech must last this long to open an utterance
PRE_ROLL_MS = 500                 # audio kept from before the speech started
TAIL_KEEP_MS = 250                # silence kept at the end of the saved utterance
MIN_UTTERANCE_S = 0.6             # shorter blips are ignored
MAX_UTTERANCE_S = 30.0
DEAF_TAIL_S = 0.4                 # half-duplex: keep ignoring audio this long after the twin stops
BARGE_IN_MS = 400                 # bonus: the instructor talks this long -> the twin stops

# ---- answer loop (inner) ----
MAX_ITERATIONS = 4
LEN_MIN_WORDS = 8                 # goal-check length range (my style is ~30-60 words)
LEN_MAX_WORDS = 110
TOOL_RESULT_CHARS = 600           # tool results are shortened before they go to the LLM
                                  # (Groq free tier: 8000 tokens per minute)

# ---- context management ----
VERBATIM_TURNS = 10               # last turns kept word for word
SUMMARIZE_BATCH = 4               # summarize older turns this many at a time

# ---- voice (XTTS-v2) ----
XTTS_MODE = os.getenv("XTTS_MODE", "remote")     # "remote" (Colab) | "local" | "off"
XTTS_URL = os.getenv("XTTS_URL", "")             # the gradio share link printed by the Colab server
XTTS_USER = "twin"
XTTS_PASS = os.getenv("TWIN_TTS_PASS", "")       # share links are public: always set a password
VOICE_DIR = config.DATA_DIR / "voice"            # private, never push to GitHub
SPEAKER_FILES = {"en": ["english_sample.wav"], "ar": ["arabic_sample.wav"]}

FILLERS = {
    "en": ["Good question...", "Okay, let me think...", "Right, so..."],
    "ar": ["سؤال حلو...", "طيب، بص...", "تمام، خليني أفكر..."],
}
FALLBACKS = {   # spoken when the answer loop hits the iteration cap
    "en": "Let me check that and get back to you.",
    "ar": "خليني أراجعها وأرد على حضرتك.",
}

LIVE_DIR = config.OUT_DIR / "live"
(LIVE_DIR / "utt").mkdir(parents=True, exist_ok=True)
