"""transcribe(audio_path) -> text, language, quality score (Groq Whisper, verbose_json)."""
import math
import os

import config
import textutils as tu


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    v = getattr(obj, key, None)
    if v is None and getattr(obj, "model_extra", None):
        v = obj.model_extra.get(key)
    return default if v is None else v


def quality_score(segments, text: str) -> float:
    """0..1. exp(avg_logprob) is a per-segment confidence; (1 - no_speech_prob) penalises
    segments that are probably silence/noise. Segments are weighted by duration."""
    if not text.strip():
        return 0.0
    if not segments:
        return 0.5
    num = den = 0.0
    for s in segments:
        dur = max(float(_get(s, "end", 0)) - float(_get(s, "start", 0)), 0.05)
        conf = min(1.0, math.exp(float(_get(s, "avg_logprob", -1.0))))
        speech = 1.0 - float(_get(s, "no_speech_prob", 0.0))
        num += conf * speech * dur
        den += dur
    return round(num / den, 3)


def transcribe(audio_path: str) -> dict:
    from groq import Groq
    client = Groq()   # reads GROQ_API_KEY
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model=config.STT_MODEL,
            response_format="verbose_json",
            temperature=0.0,
            prompt=config.WHISPER_PROMPT,
        )
    text = (_get(resp, "text", "") or "").strip()
    segs = _get(resp, "segments", []) or []
    return {
        "text": text,
        "language": tu.detect_language(text),
        "whisper_language": _get(resp, "language"),
        "quality": quality_score(segs, text),
        "duration": _get(resp, "duration"),
        "segments": [{"start": _get(s, "start"), "end": _get(s, "end"),
                      "avg_logprob": _get(s, "avg_logprob"),
                      "no_speech_prob": _get(s, "no_speech_prob")} for s in segs],
    }


def transcript_from_text(text: str, quality: float | None = None) -> dict:
    """Dev mode: skip speech-to-text and type the question."""
    return {"text": text.strip(), "language": tu.detect_language(text), "whisper_language": None,
            "quality": 0.9 if quality is None else quality, "duration": None, "segments": []}
