"""Your cloned voice (XTTS-v2), sentence by sentence.

  remote : XTTS runs in a small Gradio app on Colab (xtts_server_colab.py); we call it with gradio_client
  local  : XTTS runs on this machine (needs a GPU): pip install coqui-tts
The first sentence is played while the next ones are still being generated.
"""
import hashlib
import queue
import random
import re
import threading

import numpy as np

import live_config as lc

_SPLIT = re.compile(r"(?<=[.!?؟…])\s+|\n+")


def split_sentences(text: str, max_chars: int = 150, min_chars: int = 15) -> list:
    """XTTS works best on short pieces (Arabic has a ~166 character limit): split on . ! ? ؟ and
    long pieces on commas; glue tiny pieces to the next one so the prosody stays natural."""
    parts = []
    for p in _SPLIT.split(text.strip()):
        p = p.strip()
        while len(p) > max_chars:
            cut = max(p.rfind(c, 0, max_chars) for c in (",", "،", ";", " "))
            cut = cut if cut > 20 else max_chars
            parts.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            parts.append(p)
    out, carry = [], ""
    for p in parts:
        p = (carry + " " + p).strip()
        if len(p) < min_chars:
            carry = p
        else:
            out.append(p)
            carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out


def _xl(lang: str) -> str:
    return "en" if lang == "en" else "ar"       # XTTS has no 'mixed': Arabic model reads Latin words too


class XTTSVoice:
    def __init__(self, mode=None, url=None, password=None, cache_dir=None):
        self.mode = mode or lc.XTTS_MODE
        self.url = url or lc.XTTS_URL
        self.password = password if password is not None else lc.XTTS_PASS
        self.cache_dir = cache_dir or (lc.LIVE_DIR / "fillers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._tts = self._latents = None
        self._fillers: dict = {}
        self._lock = threading.Lock()

    # ---- one sentence -> (float32 mono array, sample rate)
    def synth(self, text: str, lang: str):
        return self._synth_remote(text, lang) if self.mode == "remote" else self._synth_local(text, lang)

    def _synth_remote(self, text, lang):
        import soundfile as sf
        if self._client is None:
            from gradio_client import Client
            auth = (lc.XTTS_USER, self.password) if self.password else None
            self._client = Client(self.url, auth=auth, verbose=False)
        path = self._client.predict(text, _xl(lang), api_name="/synthesize")
        data, sr = sf.read(path, dtype="float32")
        return (data.mean(axis=1) if data.ndim > 1 else data), sr

    def _synth_local(self, text, lang):
        import os
        os.environ.setdefault("COQUI_TOS_AGREED", "1")      # XTTS-v2 license: non-commercial (CPML)
        with self._lock:
            if self._tts is None:
                import torch
                from TTS.api import TTS
                self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                    "cuda" if torch.cuda.is_available() else "cpu")
        wavs = [str(lc.VOICE_DIR / f) for f in lc.SPEAKER_FILES[_xl(lang)]]
        out = self._tts.tts(text=text, speaker_wav=wavs, language=_xl(lang))
        return np.asarray(out, dtype=np.float32), int(self._tts.synthesizer.output_sample_rate)

    # ---- fillers: synthesized once at startup, then played instantly
    def prepare_fillers(self, fillers: dict = None):
        import soundfile as sf
        for lang, phrases in (fillers or lc.FILLERS).items():
            self._fillers[lang] = []
            for ph in phrases:
                f = self.cache_dir / f"{lang}_{hashlib.md5((self.mode + ph).encode()).hexdigest()[:10]}.wav"
                if f.exists():
                    data, sr = sf.read(f, dtype="float32")
                else:
                    data, sr = self.synth(ph, lang)
                    sf.write(f, data, sr)
                self._fillers[lang].append((data, sr))

    def filler(self, lang: str):
        pool = self._fillers.get("en" if lang == "en" else "ar") or []
        return random.choice(pool) if pool else None

    # ---- a whole reply: sentences are generated ahead while the player plays
    def stream(self, text: str, lang: str, cancel: threading.Event = None):
        """Generator of (audio, sr). Yields None while the next sentence is not ready, so the player
        can keep checking the kill switch."""
        q: queue.Queue = queue.Queue(maxsize=2)
        done = object()

        def put(item):
            while not (cancel and cancel.is_set()):
                try:
                    q.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def produce():
            try:
                for s in split_sentences(text):
                    if cancel and cancel.is_set():
                        break
                    try:
                        put(self.synth(s, lang))
                    except Exception as e:
                        put(e)
                        break
            finally:
                put(done)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                if cancel and cancel.is_set():
                    return
                yield None
                continue
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item
