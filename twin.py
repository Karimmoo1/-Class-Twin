"""One full turn: transcribe -> decide -> reply -> speak, with per-stage timings.
Phase 2 can call these same four functions inside a live loop."""
import csv
import time
from contextlib import contextmanager

import config
from decide import decide, render_tree
from memory import Memory
from planner import reply
from speak import speak
from stt import transcribe, transcript_from_text
from tools import CALL_LOG, clear_call_log

MEMORY = Memory()
_turn = {"n": 0}


@contextmanager
def stage(timings: dict, name: str):
    t0 = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - t0, 3)


def run_turn(audio_path=None, text=None, fake_quality=None, memory=MEMORY, muted=False) -> dict:
    _turn["n"] += 1
    timings = {}
    clear_call_log()
    t_all = time.perf_counter()

    with stage(timings, "speech_to_text"):
        tr = transcript_from_text(text, fake_quality) if text else transcribe(audio_path)
    with stage(timings, "decision"):
        decision = decide(tr, memory)
    with stage(timings, "reply_total"):
        twin_reply, plan = reply(decision, tr, memory)
    timings["tools_parallel"] = plan["timings"].get("tools_parallel", 0.0)
    timings["llm"] = plan["timings"].get("llm", 0.0)

    memory.add_heard(tr["text"])
    if twin_reply.action in ("answer", "defer", "ask_to_repeat") and twin_reply.reply_text:
        memory.add_said(twin_reply.reply_text)

    with stage(timings, "speech"):
        audio = speak(twin_reply, muted=muted)
    timings["total"] = round(time.perf_counter() - t_all, 3)

    log = config.OUT_DIR / "timings.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["turn", "action", *timings.keys()])
        w.writerow([_turn["n"], twin_reply.action, *timings.values()])

    return {"transcript": tr, "decision": decision, "tree": render_tree(decision),
            "tool_calls": list(CALL_LOG), "plan": plan, "reply": twin_reply.model_dump(),
            "audio": audio, "timings": timings}
