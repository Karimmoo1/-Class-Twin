"""Human in the loop: state on screen, approve/auto mode, kill switch.
Keys (terminal must be focused):   k = KILL (stop speaking now + mute)   u = unmute
                                   a = switch approve/auto              q = quit the meeting loop
                                   Enter = let the twin speak           Esc = drop the draft"""
import queue
import threading
import time

ICON = {"listening": "🟢 LISTENING", "thinking": "🟡 THINKING", "speaking": "🔵 SPEAKING", "muted": "🔴 MUTED"}


class Controls:
    def __init__(self, mode="approve", printer=print):
        self.mode, self.printer = mode, printer
        self.muted = False
        self._state = "listening"
        self.stop_event = threading.Event()       # the player stops when this is set
        self.quit_event = threading.Event()
        self.killed_at = None
        self._keys = queue.Queue()
        self._waiting = False
        self._lock = threading.Lock()

    # ---- state on screen
    @property
    def state(self) -> str:
        return "muted" if self.muted else self._state

    def set_state(self, s: str):
        with self._lock:
            changed = self.state != ("muted" if self.muted else s)
            self._state = s
        if changed:
            self.printer(f"[{ICON['muted' if self.muted else s]}]")

    # ---- the kill switch
    def kill(self):
        self.killed_at = time.monotonic()
        self.muted = True
        self.stop_event.set()                      # speech stops within one audio block
        if self._waiting:                          # a draft was waiting for approval: drop it
            self._keys.put("drop")
        self.printer(f"[{ICON['muted']}]  (press u to unmute)")

    def unmute(self):
        self.muted = False
        self.stop_event.clear()
        self.printer(f"[{ICON['listening']}]")

    def toggle_mode(self):
        self.mode = "auto" if self.mode == "approve" else "approve"
        self.printer(f"[mode: {self.mode.upper()}]")

    # ---- keyboard
    def handle_key(self, k: str):
        k = k.lower() if len(k) == 1 else k
        if k == "k":
            self.kill()
        elif k == "u":
            self.unmute()
        elif k == "a":
            self.toggle_mode()
        elif k == "q":
            self.quit_event.set()
            self.stop_event.set()
            if self._waiting:
                self._keys.put("drop")
        elif self._waiting and k in ("\r", "\n", "\x1b"):
            self._keys.put("speak" if k in ("\r", "\n") else "drop")

    def start_keyboard(self):
        import readchar          # pip install readchar (works on Windows, macOS, Linux terminals)

        def loop():
            while not self.quit_event.is_set():
                try:
                    self.handle_key(readchar.readkey())
                except Exception:
                    time.sleep(0.2)
        threading.Thread(target=loop, daemon=True).start()

    # ---- approve mode
    def approve(self, text: str, lang: str, sources: list, timeout: float) -> str:
        """Show the draft. Enter -> 'speak', Esc -> 'drop'. No key before the timeout -> 'drop' (safe)."""
        self.printer(f"\n  DRAFT [{lang}] sources={sources}\n  {text}\n  Enter = speak   Esc = drop")
        while not self._keys.empty():
            self._keys.get_nowait()
        self._waiting = True
        try:
            return self._keys.get(timeout=timeout)
        except queue.Empty:
            return "drop"
        finally:
            self._waiting = False
