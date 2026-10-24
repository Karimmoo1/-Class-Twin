"""Speaker = voice + player + controls. Everything the twin says goes through say()."""
import live_config as lc
import tashkeel


class Speaker:
    def __init__(self, voice, player, controls, printer=print):
        self.voice, self.player, self.controls, self.printer = voice, player, controls, printer
        self.history: list = []          # what was really sent to the speakers (the tests read this)

    def say(self, text: str, lang: str, on_first_audio=None) -> dict:
        if self.controls.muted:
            return {"spoken": False, "reason": "muted"}
        if not text.strip():
            return {"spoken": False, "reason": "empty"}
        self.controls.stop_event.clear()
        self.controls.set_state("speaking")
        try:
            if self.voice is None:       # --dry: no audio, print what would be said
                self.printer(f"  [SAY {lang}] {text}")
                self.history.append({"text": text, "lang": lang, "played_s": 0.0, "stopped": False})
                return {"spoken": True, "dry": True, "played_s": 0.0, "stopped": False}
            spoken = tashkeel.for_speech(text, lang)          # harakat for the voice only (off by default)
            res = self.player.play(self.voice.stream(spoken, lang, cancel=self.controls.stop_event), on_first_audio)
            self.history.append({"text": text, "lang": lang, **res})
            return {"spoken": res["played_s"] > 0, "reason": "killed" if res["stopped"] else "done", **res}
        except Exception as e:           # voice server down etc.: never crash the meeting
            self.printer(f"  [voice error] {type(e).__name__}: {str(e)[:120]}\n  [TEXT ONLY] {text}")
            return {"spoken": False, "reason": "voice_error", "error": str(e)[:200]}
        finally:
            self.controls.set_state("listening")

    def say_filler(self, lang: str, on_first_audio=None) -> dict:
        if self.controls.muted or self.voice is None:
            return {"spoken": False, "reason": "muted_or_dry"}
        f = self.voice.filler(lang)
        if f is None:
            return {"spoken": False, "reason": "no_filler"}
        self.controls.stop_event.clear()
        self.controls.set_state("speaking")
        try:
            res = self.player.play(iter([f]), on_first_audio)
            self.history.append({"text": "<filler>", "lang": lang, **res})
            return {"spoken": res["played_s"] > 0, **res}
        finally:
            self.controls.set_state("thinking")
