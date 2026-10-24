"""Run the 8 tests in order (memory is shared, so test 6 can use what the twin said earlier).
  python run_tests.py --mode text    # uses the "text" field (fast, no audio needed)
  python run_tests.py --mode audio   # uses the "audio" files in tests/audio/
Writes outputs/results.md and outputs/tests/test_N.json"""
import argparse
import json
from pathlib import Path

import config
from twin import MEMORY, run_turn

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["text", "audio"], default="text")
args = ap.parse_args()

tests = json.loads(Path(__file__).with_name("tests").joinpath("tests.json").read_text(encoding="utf-8"))
MEMORY.reset()
rows = []
for t in tests:
    if args.mode == "audio":
        out = run_turn(audio_path=str(Path(__file__).parent / t["audio"]))
    else:
        out = run_turn(text=t["text"], fake_quality=t.get("fake_quality"))
    exp = t["expected"] if isinstance(t["expected"], list) else [t["expected"]]
    actual = out["reply"]["action"]
    (config.OUT_DIR / "tests" / f"test_{t['id']}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows.append((t["id"], t["name"], "/".join(exp), actual, "✅" if actual in exp else "❌",
                 out["reply"]["language"], out["transcript"]["quality"], out["timings"]["total"]))
    print(f"[{t['id']}] expected {exp} -> {actual}\n{out['tree']}\nreply: {out['reply']['reply_text']}\n")

md = ["| # | Test | Expected | Actual | Match | Lang | Quality | Total s |", "|---|---|---|---|---|---|---|---|"]
md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
(config.OUT_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
