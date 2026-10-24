"""Start the twin in a live meeting.   python run_live.py --help

Typical runs
  Windows (Zoom/Meet/Teams):  python run_live.py --source loopback --out-device "CABLE Input"
  macOS:                      python run_live.py --source "device:BlackHole 16ch" --out-device "BlackHole 2ch"
  Plan B (mic + speakers):    python run_live.py --source mic
  No audio at all (debug):    python run_live.py --dry --source mic
In the meeting app: microphone = "CABLE Output" (Windows) or "BlackHole 2ch" (macOS),
and the display name must end with "(AI twin)".
"""
import argparse
import threading
import time

import live_config as lc

BANNER = """
  Class Twin - live.  Responsible use:
   - only YOUR OWN voice, with your consent recording made
   - only in meetings where everyone knows the twin is running (name ends with "(AI twin)")
   - never for attendance, quizzes/exams, or to attend in your place
  Keys:  k = KILL (stop + mute)   u = unmute   a = approve/auto   q = quit   Enter = speak   Esc = drop
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["approve", "auto"], default=lc.MODE_DEFAULT)
    ap.add_argument("--source", default="loopback", help="loopback | mic | device:NAME")
    ap.add_argument("--out-device", default=None, help='e.g. "CABLE Input" or "BlackHole 2ch"')
    ap.add_argument("--xtts", choices=["remote", "local", "off"], default=lc.XTTS_MODE)
    ap.add_argument("--url", default=lc.XTTS_URL, help="Colab gradio share link (remote mode)")
    ap.add_argument("--minutes", type=float, default=lc.MAX_MEETING_MIN)
    ap.add_argument("--idle", type=float, default=lc.IDLE_MIN, help="stop after N minutes without speech")
    ap.add_argument("--barge-in", action="store_true", help="bonus (experimental)")
    ap.add_argument("--tashkeel", choices=["off", "llm"], default=None, help="harakat for the voice (try the A/B test first)")
    ap.add_argument("--dry", action="store_true", help="no audio output: print what the twin would say")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()

    if a.list_devices:
        from audio_io import list_devices
        return list_devices()

    if a.tashkeel:
        import tashkeel
        tashkeel.MODE = a.tashkeel
    print(BANNER)
    from audio_io import Capture, Player, SoundDeviceSink
    from context_mgr import LiveMemory
    from controls import Controls
    from live_answer import live_answer, live_canned, live_decide
    from livelog import JsonlLogger
    from meeting_loop import MeetingLoop
    from speaker import Speaker
    from vad import UtteranceDetector

    controls = Controls(mode=a.mode)
    voice = sink = None
    if not a.dry:
        from tts_xtts import XTTSVoice
        if a.xtts != "off":
            voice = XTTSVoice(mode=a.xtts, url=a.url)
            print("preparing fillers in your voice ...")
            voice.prepare_fillers()
        sink = SoundDeviceSink(a.out_device)
    speaker = Speaker(voice, Player(sink, controls.stop_event) if sink else None, controls)

    print("loading models (VAD + embeddings) ...")
    detector = UtteranceDetector()
    try:
        import retrieval
        retrieval.warmup()
    except Exception as e:
        print("warmup skipped:", e)

    source, _, device = a.source.partition(":")
    if source == "mic":      # speakers + microphone: the room echo lasts longer, so stay deaf a bit longer
        lc.DEAF_TAIL_S = max(lc.DEAF_TAIL_S, 1.0)
    capture = Capture(source=source, device=device or None)
    logger = JsonlLogger()
    loop = MeetingLoop(capture, detector, speaker, controls, LiveMemory(), logger, barge_in=a.barge_in,
                       max_minutes=a.minutes, idle_minutes=a.idle,
                       decide_fn=live_decide, answer_fn=live_answer, canned_fn=live_canned)
    capture.start()
    controls.start_keyboard()
    print(f"listening ... log: {logger.path}\n")
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C)")
    finally:
        capture.stop()
        controls.quit_event.set()
    print("log saved:", logger.path)
    try:
        import latency_report as lr
        rows = lr.load([logger.path])
        if rows:
            print(lr.table(rows))
    except Exception:
        pass


if __name__ == "__main__":
    main()
