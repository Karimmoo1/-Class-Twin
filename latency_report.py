"""Median and worst time per stage, from the JSONL meeting logs.
python latency_report.py [path-to-log.jsonl ...]     (default: every log in outputs/live/)"""
import json
import statistics
import sys

import live_config as lc

STAGES = [("vad_s", "VAD (end of speech -> utterance)"), ("stt_s", "Speech to text"),
          ("decision_s", "Decision tree"), ("answer_loop_s", "Answer loop"),
          ("first_audio_s", "First audio (question end -> sound)"),
          ("first_audio_excl_approval_s", "First audio, without your approval time")]


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def table(rows) -> str:
    out = ["| Stage | n | Median (s) | Worst (s) |", "|---|---|---|---|"]
    for key, label in STAGES:
        vals = [r["timings"][key] for r in rows if r.get("timings", {}).get(key) is not None]
        if vals:
            out.append(f"| {label} | {len(vals)} | {statistics.median(vals):.2f} | {max(vals):.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(str(p) for p in lc.LIVE_DIR.glob("meeting_*.jsonl"))
    if not paths:
        sys.exit("no logs found in " + str(lc.LIVE_DIR))
    t = table(load(paths))
    print(t)
    (lc.LIVE_DIR / "latency_table.md").write_text(t + "\n", encoding="utf-8")
    print("\nsaved to", lc.LIVE_DIR / "latency_table.md")
