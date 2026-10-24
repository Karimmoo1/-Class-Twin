"""Cut the live audio into utterances. silero-vad gives a speech probability for every 32 ms chunk;
the state machine below opens an utterance on speech and closes it after SILENCE_MS of silence.
prob_fn is injectable, so the logic is testable without torch."""
import time
from collections import deque

import numpy as np

import live_config as lc


def silero_prob_fn():
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad()

    def fn(chunk: np.ndarray) -> float:
        with torch.no_grad():
            return float(model(torch.from_numpy(chunk), lc.SAMPLE_RATE).item())
    fn.reset = model.reset_states
    return fn


class UtteranceDetector:
    def __init__(self, prob_fn=None, sr=lc.SAMPLE_RATE, chunk=lc.CHUNK, threshold=lc.VAD_THRESHOLD,
                 silence_ms=lc.SILENCE_MS, min_speech_ms=lc.MIN_SPEECH_MS, pre_roll_ms=lc.PRE_ROLL_MS,
                 tail_keep_ms=lc.TAIL_KEEP_MS, min_s=lc.MIN_UTTERANCE_S, max_s=lc.MAX_UTTERANCE_S):
        self.prob_fn = prob_fn or silero_prob_fn()
        self.sr, self.chunk, self.threshold = sr, chunk, threshold
        self.ms = 1000.0 * chunk / sr
        self.silence_chunks = int(np.ceil(silence_ms / self.ms))
        self.min_speech_chunks = max(1, int(np.ceil(min_speech_ms / self.ms)))
        self.pre_chunks = max(1, int(pre_roll_ms / self.ms))
        self.tail_keep = int(tail_keep_ms / 1000 * sr)
        self.min_s, self.max_s = min_s, max_s
        self.reset()

    def reset(self):
        self.active = False
        self.buf, self._sil, self._run = [], 0, 0
        self.pre = deque(maxlen=self.pre_chunks)
        self.t_last_speech = None
        getattr(self.prob_fn, "reset", lambda: None)()

    @property
    def speech_s(self) -> float:
        """How long the current utterance has been going (used for barge-in)."""
        return len(self.buf) * self.chunk / self.sr if self.active else 0.0

    def feed(self, chunk: np.ndarray, t: float = None):
        """Feed one 512-sample float32 chunk. Returns an utterance dict when one just ended, else None."""
        t = time.monotonic() if t is None else t
        speech = self.prob_fn(chunk) >= self.threshold
        if not self.active:
            self.pre.append(chunk)
            self._run = self._run + 1 if speech else 0
            if self._run >= self.min_speech_chunks:
                self.active, self.buf, self._sil = True, list(self.pre), 0
                self.t_last_speech = t
            return None
        self.buf.append(chunk)
        if speech:
            self._sil, self.t_last_speech = 0, t
        else:
            self._sil += 1
        too_long = len(self.buf) * self.chunk / self.sr >= self.max_s
        if self._sil >= self.silence_chunks or too_long:
            audio = np.concatenate(self.buf)
            cut = max(0, self._sil * self.chunk - self.tail_keep)       # trim most of the trailing silence
            audio = audio[:len(audio) - cut] if cut else audio
            t_end, t_emit = self.t_last_speech, t
            self.reset()
            if len(audio) / self.sr < self.min_s:
                return None
            return {"audio": audio, "duration_s": round(len(audio) / self.sr, 2),
                    "t_end": t_end, "t_emit": t_emit}
        return None
