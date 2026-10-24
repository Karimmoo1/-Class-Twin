"""Outer loop: Observe -> Think -> Act, until the stop key, the max length or too long without speech.

  ears thread : capture -> VAD -> finished utterances in a queue (half-duplex: deaf while the twin speaks)
  OBSERVE     : next utterance + its transcript (speech to text) + the meeting memory
  THINK       : Phase 1 decide() (the decision tree)
  ACT         : answer (inner agent loop) / ask_to_repeat / defer  -> speak;  stay_silent -> nothing
"""
import queue
import threading
import time

import numpy as np

import live_config as lc


class MeetingLoop:
    def __init__(self, capture, detector, speaker, controls, memory, logger, stt_fn=None, decide_fn=None,
                 answer_fn=None, canned_fn=None, barge_in=False, run_id=None, printer=print,
                 max_minutes=lc.MAX_MEETING_MIN, idle_minutes=lc.IDLE_MIN):
        if stt_fn is None:
            from stt import transcribe as stt_fn
        if decide_fn is None:
            from decide import decide as decide_fn
        if answer_fn is None:
            from answer_loop import answer as answer_fn
        if canned_fn is None:
            from planner import reply as canned_fn
        self.capture, self.detector, self.speaker, self.controls = capture, detector, speaker, controls
        self.memory, self.logger = memory, logger
        self.stt_fn, self.decide_fn, self.answer_fn, self.canned_fn = stt_fn, decide_fn, answer_fn, canned_fn
        self.barge_in, self.printer = barge_in, printer
        self.max_s, self.idle_s = max_minutes * 60, idle_minutes * 60
        self.run_id = run_id or time.strftime("%H%M%S")
        self.utts: queue.Queue = queue.Queue()
        self.n = 0
        self.handled = 0
        self._ears_stop = threading.Event()
        self._deaf_until = 0.0
        self._was_deaf = False

    # ------------------------------------------------------------ ears (runs in its own thread)
    def _ears(self):
        while not self._ears_stop.is_set():
            item = self.capture.get(timeout=0.2)
            if item is None:
                continue
            t, chunk = item
            speaking = self.controls.state == "speaking"
            if speaking:
                self._deaf_until = t + lc.DEAF_TAIL_S
            if not self.barge_in and (speaking or t < self._deaf_until):
                if not self._was_deaf:           # half-duplex: ignore incoming audio while the twin speaks
                    self.detector.reset()
                    self._was_deaf = True
                continue
            self._was_deaf = False
            utt = self.detector.feed(chunk, t)
            if self.barge_in and speaking and self.detector.speech_s >= lc.BARGE_IN_MS / 1000:
                self.controls.stop_event.set()   # bonus: the instructor started talking -> stop and listen
            if utt:
                self.utts.put(utt)

    # ------------------------------------------------------------ the loop
    def stop_reason(self, t_start, last_activity):
        now = time.monotonic()
        if self.controls.quit_event.is_set():
            return "stop_key"
        if now - t_start >= self.max_s:
            return "max_length"
        if now - last_activity >= self.idle_s:
            return "idle_timeout"
        return None

    def run(self) -> str:
        threading.Thread(target=self._ears, daemon=True).start()
        t_start = last = time.monotonic()
        self.controls.set_state("listening")
        reason = None
        while not reason:
            reason = self.stop_reason(t_start, last)
            if reason:
                break
            try:
                utt = self.utts.get(timeout=0.2)             # OBSERVE
            except queue.Empty:
                continue
            last = time.monotonic()
            try:
                self.handle(utt)
            except Exception as e:                           # one bad utterance must not end the meeting
                self.printer(f"[error] {type(e).__name__}: {e}")
                self.logger.write({"utterance": self.n, "error": f"{type(e).__name__}: {e}"})
            self.handled += 1
            last = time.monotonic()
            self.controls.set_state("listening")
        self._ears_stop.set()
        self.capture.stop()
        self.printer(f"[meeting loop ended: {reason}]")
        return reason

    # ------------------------------------------------------------ one utterance
    def handle(self, utt: dict) -> dict:
        import soundfile as sf
        self.n += 1
        t_end = utt["t_end"]
        rec = {"utterance": self.n, "time": time.strftime("%H:%M:%S"), "mode": self.controls.mode,
               "duration_s": utt["duration_s"]}
        tm = {"vad_s": round(utt["t_emit"] - t_end, 3)}
        self.controls.set_state("thinking")

        path = lc.LIVE_DIR / "utt" / f"{self.run_id}_{self.n:03d}.wav"
        sf.write(path, utt["audio"], lc.SAMPLE_RATE, subtype="PCM_16")
        t0 = time.perf_counter()
        tr = self.stt_fn(str(path))                                              # OBSERVE: transcript
        tm["stt_s"] = round(time.perf_counter() - t0, 3)
        rec["transcript"] = {k: tr.get(k) for k in ("text", "language", "quality")}
        if not tr["text"].strip():
            rec.update(decision=None, action="empty_transcript", timings=tm)
            self.logger.write(rec)
            return rec
        self.printer(f"\n  HEARD [{tr['language']}, q={tr['quality']}]: {tr['text']}")

        t0 = time.perf_counter()
        decision = self.decide_fn(tr, self.memory)                               # THINK
        tm["decision_s"] = round(time.perf_counter() - t0, 3)
        action = decision["selected"]
        rec["decision"] = {"selected": action, "fallback": decision.get("fallback"),
                           "llm_reason": decision.get("llm_reason"),
                           "features": decision.get("features"),
                           "branches": [{k: b[k] for k in ("action", "kind", "score", "threshold", "status", "reason")}
                                        for b in decision["branches"]]}
        self.printer(f"  DECISION: {action}")

        first = {"t": None}

        def on_first():
            if first["t"] is None:
                first["t"] = time.monotonic()

        reply, trace, approval, spoke, filler_t = None, None, "n/a", False, None
        if action == "stay_silent":
            self.memory.add_heard(tr["text"])
            rec.update(action=action, reply=None, timings=tm)
            self.logger.write(rec)
            return rec

        if action == "answer":                                                   # ACT
            filler = None
            if self.controls.mode == "auto" or lc.FILLER_IN_APPROVE_MODE:
                filler = threading.Thread(target=self.speaker.say_filler, args=(decision["language"], on_first))
                filler.start()                                                   # instant, cached
            t0 = time.perf_counter()
            reply, trace = self.answer_fn(tr, decision, self.memory)
            tm["answer_loop_s"] = round(time.perf_counter() - t0, 3)
            if filler:
                filler.join()
        else:
            reply, _ = self.canned_fn(decision, tr, self.memory)

        approve_wait = 0.0
        if self.controls.muted:
            approval = "muted"
        elif self.controls.mode == "approve":
            t0 = time.perf_counter()
            approval = self.controls.approve(reply.reply_text, reply.language, reply.sources, lc.APPROVE_TIMEOUT_S)
            approve_wait = round(time.perf_counter() - t0, 3)
        else:
            approval = "auto"
        tm["approval_wait_s"] = approve_wait

        self.memory.add_heard(tr["text"])
        say = {"spoken": False}
        if approval in ("auto", "speak"):
            t0 = time.perf_counter()
            say = self.speaker.say(reply.reply_text, reply.language, on_first)
            tm["speak_s"] = round(time.perf_counter() - t0, 3)
            if say.get("spoken"):
                self.memory.add_said(reply.reply_text)
        elif approval == "drop":
            self.printer("  (draft dropped)")

        if first["t"] is not None:
            tm["first_audio_s"] = round(first["t"] - t_end, 3)               # question end -> first sound
            tm["first_audio_excl_approval_s"] = round(first["t"] - t_end - approve_wait, 3)
        tm["think_total_s"] = round(sum(tm.get(k, 0) for k in ("vad_s", "stt_s", "decision_s", "answer_loop_s")), 3)
        rec.update(action=action, reply=reply.model_dump(), answer_loop=trace, approval=approval,
                   spoken=say, memory={"verbatim_turns": len(self.memory.turns),
                                       "summary_chars": len(getattr(self.memory, "summary", ""))},
                   timings=tm)
        self.logger.write(rec)
        return rec
