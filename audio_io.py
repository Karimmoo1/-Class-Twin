"""Audio in and out.  python audio_io.py --list   shows your devices.

Capture sources (what the twin hears):
  loopback        Windows (or Linux): record what your speakers play = the meeting audio   [soundcard]
  device:NAME     any input device by name part, e.g. device:BlackHole 16ch (macOS)         [sounddevice]
  mic             the default microphone (Plan B: classmate asks in the room)               [sounddevice]
Playback: --out-device NAME, e.g. "CABLE Input" (Windows VB-CABLE) or "BlackHole 2ch" (macOS).
"""
import queue
import threading
import time
from math import gcd

import numpy as np

import live_config as lc


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    g = gcd(int(sr_in), int(sr_out))
    try:
        from scipy.signal import resample_poly
        return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)
    except ImportError:
        n = int(len(x) * sr_out / sr_in)
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def find_device(name, kind: str):
    """name: None (default device), an int index, or part of a device name."""
    import sounddevice as sd
    if name is None or name == "":
        return None
    if isinstance(name, int) or str(name).isdigit():
        return int(name)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if str(name).lower() in d["name"].lower() and d[key] > 0:
            return i
    raise RuntimeError(f"no {kind} device matching '{name}'. Run: python audio_io.py --list")


def list_devices():
    import sounddevice as sd
    print("== sounddevice (playback and macOS/BlackHole capture) ==")
    print(sd.query_devices())
    try:
        import soundcard as sc
        print("\n== soundcard (Windows loopback) ==")
        for s in sc.all_speakers():
            print("speaker:", s.name)
    except Exception as e:
        print("\nsoundcard not available:", e)


# ---------------------------------------------------------------- playback
class SoundDeviceSink:
    """Writes float32 mono blocks to an output device (resampled, copied to 2 channels if needed)."""

    def __init__(self, device=None):
        import sounddevice as sd
        self.sd = sd
        self.device = find_device(device, "output")
        info = sd.query_devices(self.device if self.device is not None else sd.default.device[1])
        self.sr = int(info["default_samplerate"])
        self.ch = 2 if info["max_output_channels"] >= 2 else 1
        self.stream = None

    def write(self, block: np.ndarray, sr: int):
        if self.stream is None:
            self.stream = self.sd.OutputStream(device=self.device, samplerate=self.sr, channels=self.ch,
                                               dtype="float32")
            self.stream.start()
        x = resample(block, sr, self.sr)
        x = np.stack([x, x], axis=1) if self.ch == 2 else x[:, None]
        self.stream.write(np.ascontiguousarray(x, dtype=np.float32))

    def abort(self):                       # kill switch: drop what is still buffered, right now
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
            finally:
                self.stream = None

    def close(self):                       # normal end: let the buffer drain
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None


class Player:
    """Plays chunks block by block (0.1 s) and checks stop_event between blocks, so the kill switch
    stops the audio within about one block. `chunks` may yield None (= nothing ready yet)."""

    def __init__(self, sink, stop_event: threading.Event, block_s: float = 0.1):
        self.sink, self.stop, self.block_s = sink, stop_event, block_s

    def play(self, chunks, on_first_audio=None) -> dict:
        played, first, stopped = 0.0, False, False
        t0 = time.perf_counter()
        try:
            for item in chunks:
                if self.stop.is_set():
                    stopped = True
                    break
                if item is None:
                    continue
                arr, sr = item
                step = max(1, int(sr * self.block_s))
                for i in range(0, len(arr), step):
                    if self.stop.is_set():
                        stopped = True
                        break
                    if not first:
                        first = True
                        if on_first_audio:
                            on_first_audio()
                    self.sink.write(arr[i:i + step], sr)
                    played += min(step, len(arr) - i) / sr
                if stopped:
                    break
        finally:
            (self.sink.abort if stopped else self.sink.close)()
        return {"played_s": round(played, 2), "stopped": stopped, "seconds": round(time.perf_counter() - t0, 3)}


# ---------------------------------------------------------------- capture
class _Chunker:
    """Any block size / sample rate in -> 512-sample chunks at 16 kHz out."""

    def __init__(self, sr_in):
        self.sr_in, self.buf = sr_in, np.zeros(0, np.float32)

    def push(self, x: np.ndarray) -> list:
        x = resample(x.astype(np.float32), self.sr_in, lc.SAMPLE_RATE)
        self.buf = np.concatenate([self.buf, x])
        out = []
        while len(self.buf) >= lc.CHUNK:
            out.append(self.buf[:lc.CHUNK].copy())
            self.buf = self.buf[lc.CHUNK:]
        return out


class Capture:
    """Background thread that fills a queue with (monotonic_time, 512-sample float32 chunk)."""

    def __init__(self, source="loopback", device=None):
        self.source, self.device = source, device
        self.q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        target = self._run_loopback if self.source == "loopback" else self._run_sounddevice
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()

    def get(self, timeout=0.2):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run_loopback(self):
        import soundcard as sc          # pip install -U soundcard   (Windows WASAPI loopback)
        spk = sc.get_speaker(self.device) if self.device else sc.default_speaker()
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        print(f"[capture] loopback of: {spk.name}")
        for ch in (2, 1):
            try:
                with mic.recorder(samplerate=lc.SAMPLE_RATE, channels=ch) as rec:
                    while not self._stop.is_set():
                        data = rec.record(numframes=lc.CHUNK)
                        self.q.put((time.monotonic(), data.mean(axis=1).astype(np.float32)))
                return
            except Exception as e:
                print(f"[capture] loopback with {ch} channel(s) failed: {e}")

    def _run_sounddevice(self):
        import sounddevice as sd
        dev = find_device(self.device, "input")
        name = self.device or "default microphone"
        try:                              # most devices accept 16 kHz directly
            sr, chunker = lc.SAMPLE_RATE, _Chunker(lc.SAMPLE_RATE)
            stream = sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32", blocksize=lc.CHUNK,
                                    callback=lambda i, f, t, s: self._push(chunker, i))
        except Exception:                 # otherwise record at the device rate and resample
            info = sd.query_devices(dev if dev is not None else sd.default.device[0])
            sr = int(info["default_samplerate"])
            chunker = _Chunker(sr)
            stream = sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32",
                                    callback=lambda i, f, t, s: self._push(chunker, i))
        print(f"[capture] input: {name} at {sr} Hz")
        with stream:
            while not self._stop.is_set():
                time.sleep(0.1)

    def _push(self, chunker, indata):
        t = time.monotonic()
        for c in chunker.push(indata[:, 0]):
            self.q.put((t, c))


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        list_devices()
    else:
        print(__doc__)
