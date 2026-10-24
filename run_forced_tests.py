"""Phase 2 forced tests, with real asserts. Runs offline (stub model, fake tools, fake speakers).
    python run_forced_tests.py                 the 5 required tests + 3 extra checks
    python run_forced_tests.py --live-llm      tests 1 and 3 again with the real Groq model + real tools
    python run_forced_tests.py --audio-dir tests/audio   take the English/Arabic questions from your recordings
The report is saved to outputs/live/forced_tests_output.txt (a Phase 2 deliverable)."""
import argparse
import io
import json
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import redirect_stdout

import numpy as np

import config
import live_config as lc
from answer_loop import ToolRunner, answer
from controls import Controls
from memory import Memory
from audio_io import Player
from speaker import Speaker

EN_Q = "Karim, what's the difference between RAG and fine-tuning?"
AR_Q = "يا كريم، إيه الفرق بين الـ RAG والـ fine-tuning؟"
EN_REPLY = ("RAG retrieves the relevant notes at question time and adds them to the prompt, while fine-tuning "
            "changes the model weights. In AgriVision I used RAG because the disease info has to show its sources.")
AR_REPLY = ("RAG هو إنك تجيب الـ notes المناسبة وقت السؤال وتضيفها للـ prompt، أما fine-tuning فهو تعديل وزن "
            "الـ model نفسه. في AgriVision استخدمت RAG عشان معلومات الأمراض لازم تبان مع مصدرها.")


# ------------------------------------------------------------------ fakes
class FakeTools:
    """Stands in for tools.call_tool. `fail` = tool names that raise."""

    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), Counter()

    def __call__(self, name, **args):
        self.calls[name] += 1
        if name in self.fail:
            raise RuntimeError("index offline")
        return {
            "search_my_notes": [{"source": "disease_rag_pipeline.md", "text": "RAG pipeline: FAISS + embeddings + Llama.", "score": 0.8}],
            "get_style_examples": [{"question": "what is RAG?", "answer": "Basically it retrieves first.", "language": "en", "score": 0.7}],
            "lookup_glossary": {"found": True, "term": "rag", "definition": "retrieval-augmented generation"},
            "get_profile": {"background": ["AI/ML on AgriVision"], "honesty_rules": ["never invent"]},
        }[name]


def msg(content="", *calls):
    return {"content": content, "tool_calls": [{"id": f"c{i}", "name": n, "args": a} for i, (n, a) in enumerate(calls)]}


def final(text, sources, conf=0.8):
    return msg(json.dumps({"reply_text": text, "confidence": conf, "sources": sources}, ensure_ascii=False))


def scripted(*steps):
    it, n = iter(steps), {"calls": 0}

    def think(goal, ctx):
        n["calls"] += 1
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("think_fn called more often than scripted")
    think.n = n
    return think


def decision(lang):
    return {"language": lang, "addressed_to_me": True, "answer_score": 0.9, "selected": "answer"}


def tr(text, lang):
    return {"text": text, "language": lang, "quality": 0.9}


class FakeSink:
    """A 'speaker' that takes real time to play, like a sound card."""

    def __init__(self, realtime=True):
        self.realtime, self.written_s, self.aborted, self.closed = realtime, 0.0, False, False

    def write(self, block, sr):
        if self.realtime:
            time.sleep(len(block) / sr)
        self.written_s += len(block) / sr

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


class FakeVoice:
    def __init__(self, n=3, seconds_each=2.0, sr=24000):
        self.n, self.sec, self.sr = n, seconds_each, sr

    def stream(self, text, lang, cancel=None):
        t = np.linspace(0, self.sec, int(self.sec * self.sr), dtype=np.float32)
        for _ in range(self.n):
            yield (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), self.sr

    def filler(self, lang):
        return np.zeros(2400, np.float32), 24000


def make_speaker(realtime=False, n=1, sec=0.3):
    c = Controls(mode="auto", printer=lambda *_: None)
    sink = FakeSink(realtime)
    return Speaker(FakeVoice(n, sec), Player(sink, c.stop_event, 0.1), c, printer=lambda *_: None), c, sink


# ------------------------------------------------------------------ the five forced tests
def test1_normal_ending():
    ToolRunner._cache.clear()
    tools = FakeTools()
    think = scripted(msg("", ("search_my_notes", {"query": EN_Q, "k": 3}), ("get_style_examples", {"question": EN_Q})),
                     final(EN_REPLY, ["disease_rag_pipeline.md"]))
    reply, trace = answer(tr(EN_Q, "en"), decision("en"), Memory(), think_fn=think, runner=tools)
    assert trace["termination_reason"] == "model_stopped", trace["termination_reason"]
    assert trace["iterations"] == 2 < lc.MAX_ITERATIONS
    assert trace["goal_check"]["ok"] and trace["retries"] == 0
    assert reply.action == "answer" and reply.sources == ["disease_rag_pipeline.md"]
    assert set(reply.tools_used) == {"search_my_notes", "get_style_examples"}
    print(f"reason={trace['termination_reason']} iterations={trace['iterations']}/{lc.MAX_ITERATIONS} "
          f"tools={reply.tools_used} sources={reply.sources}")


def test2_never_terminates():
    for lang, q, fb in (("en", EN_Q, lc.FALLBACKS["en"]), ("ar", AR_Q, lc.FALLBACKS["ar"])):
        ToolRunner._cache.clear()
        n = {"i": 0}

        def always_tool(goal, ctx):
            n["i"] += 1
            return msg("", ("search_my_notes", {"query": f"attempt {n['i']}", "k": 3}))
        reply, trace = answer(tr(q, lang), decision(lang), Memory(), think_fn=always_tool, runner=FakeTools())
        assert trace["termination_reason"] == "max_iterations"
        assert trace["iterations"] == lc.MAX_ITERATIONS == n["i"]
        assert reply.action == "defer" and reply.reply_text == fb
        speaker, c, sink = make_speaker()
        speaker.say(reply.reply_text, reply.language)
        assert [h["text"] for h in speaker.history] == [fb]
        print(f"[{lang}] stopped at iteration {trace['iterations']} (cap {lc.MAX_ITERATIONS}); spoke the fallback: {fb!r}")


def test3_tool_failure():
    # 3a: the model reads the error and changes plan (another tool)
    ToolRunner._cache.clear()
    tools = FakeTools(fail={"search_my_notes"})

    def replan(goal, ctx):
        last = ctx[-1]
        if last["role"] == "user":
            return msg("", ("search_my_notes", {"query": EN_Q, "k": 3}))
        if last["role"] == "tool" and "error" in last["content"]:
            return msg("", ("lookup_glossary", {"term": "RAG"}))
        return final(EN_REPLY, ["ml_glossary.md"])
    reply, trace = answer(tr(EN_Q, "en"), decision("en"), Memory(), think_fn=replan, runner=tools)
    s = trace["steps"]
    assert s[0]["tool_results"][0]["ok"] is False
    assert s[1]["tool_calls"][0]["name"] == "lookup_glossary", "the plan did not change after the error"
    assert tools.calls["search_my_notes"] == 1 and tools.calls["lookup_glossary"] == 1
    assert reply.action == "answer" and reply.sources == ["ml_glossary.md"]
    print(f"3a plan changed after the error: search_my_notes (failed) -> lookup_glossary -> answer {reply.sources}")

    # 3b: a stubborn model repeats the same failing call: the code blocks it
    ToolRunner._cache.clear()
    tools = FakeTools(fail={"search_my_notes"})
    stubborn = lambda goal, ctx: msg("", ("search_my_notes", {"query": EN_Q, "k": 3}))
    reply, trace = answer(tr(EN_Q, "en"), decision("en"), Memory(), think_fn=stubborn, runner=tools)
    results = [r for st in trace["steps"] for r in st["tool_results"]]
    assert tools.calls["search_my_notes"] == 1, "an identical failed call was executed again"
    assert results[0]["ok"] is False and not results[0]["blocked"]
    assert all(r["blocked"] for r in results[1:]) and len(results) == lc.MAX_ITERATIONS
    assert reply.action == "defer" and reply.reply_text == lc.FALLBACKS["en"]
    print(f"3b repeat blocked {len(results) - 1}x (tool really ran once); no crash; twin deferred")

    # 3c: every tool is down
    ToolRunner._cache.clear()
    tools = FakeTools(fail={"search_my_notes", "lookup_glossary", "get_style_examples", "get_profile"})
    think = scripted(msg("", ("search_my_notes", {"query": EN_Q}), ("lookup_glossary", {"term": "RAG"})),
                     final(EN_REPLY, ["disease_rag_pipeline.md"]), final(EN_REPLY, ["disease_rag_pipeline.md"]))
    reply, trace = answer(tr(EN_Q, "en"), decision("en"), Memory(), think_fn=think, runner=tools)
    assert reply.action == "defer" and trace["termination_reason"] == "goal_check_failed"
    print("3c all tools down: the draft had no valid source -> goal-check failed twice -> defer")


def test4_goal_check():
    # 4a: an English reply to an Arabic question, every time -> never spoken, twin defers
    ToolRunner._cache.clear()
    think = scripted(msg("", ("search_my_notes", {"query": AR_Q})),
                     final(EN_REPLY, ["disease_rag_pipeline.md"]), final(EN_REPLY, ["disease_rag_pipeline.md"]))
    reply, trace = answer(tr(AR_Q, "ar"), decision("ar"), Memory(), think_fn=think, runner=FakeTools())
    gc = [st["goal_check"] for st in trace["steps"] if "goal_check" in st]
    assert not gc[0]["ok"] and gc[0]["checks"]["language_matches"] is False
    assert trace["termination_reason"] == "goal_check_failed" and trace["retries"] == 1
    assert reply.action == "defer" and reply.reply_text == config.load_profile()["safe_replies"]["defer_ar"]
    speaker, c, sink = make_speaker()
    speaker.say(reply.reply_text, reply.language)
    spoken = " ".join(h["text"] for h in speaker.history)
    assert EN_REPLY not in spoken and "RAG retrieves" not in spoken
    print(f"4a check caught: {gc[0]['failed'][0][:70]}...; retried once; wrong English reply NEVER spoken; spoke: {spoken!r}")

    # 4b: wrong first, fixed on the retry (the failure was put in the context)
    ToolRunner._cache.clear()
    think = scripted(msg("", ("search_my_notes", {"query": AR_Q})),
                     final(EN_REPLY, ["disease_rag_pipeline.md"]), final(AR_REPLY, ["disease_rag_pipeline.md"]))
    reply, trace = answer(tr(AR_Q, "ar"), decision("ar"), Memory(), think_fn=think, runner=FakeTools())
    assert trace["termination_reason"] == "model_stopped" and trace["retries"] == 1
    assert reply.action == "answer" and "RAG هو" in reply.reply_text
    speaker, c, sink = make_speaker()
    speaker.say(reply.reply_text, reply.language)
    assert [h["text"] for h in speaker.history] == [AR_REPLY]
    print("4b first draft failed the language check, the retry was Arabic and passed; only the Arabic reply was spoken")


def test5_kill_switch():
    speaker, c, sink = make_speaker(realtime=True, n=3, sec=2.0)       # a 6 second reply
    out = {}
    th = threading.Thread(target=lambda: out.update(speaker.say("one. two. three.", "en")))
    th.start()
    time.sleep(1.0)
    t_kill = time.monotonic()
    c.kill()
    th.join(timeout=5)
    back = time.monotonic() - t_kill
    assert not th.is_alive(), "speech did not stop"
    assert back < 1.0, f"audio took {back:.2f}s to stop"
    assert out["reason"] == "killed" and sink.aborted and sink.written_s < 2.5
    assert c.muted and c.state == "muted"
    again = speaker.say("hello again", "en")
    assert again["spoken"] is False and again["reason"] == "muted"
    print(f"kill pressed at 1.0s of a 6.0s reply; audio stopped {back:.2f}s later (limit 1.0s); "
          f"{sink.written_s:.1f}s was played; muted; a new reply was refused")


# ------------------------------------------------------------------ extra checks (not required, but they protect the demo)
def extra_context_manager():
    from context_mgr import LiveMemory
    m = LiveMemory(summarizer=lambda prev, old: f"summary of {len(old)} older turns", background=False)
    for i in range(30):
        (m.add_heard if i % 2 == 0 else m.add_said)(f"turn number {i}")
    ctx = m.context()
    assert lc.VERBATIM_TURNS <= len(m.turns) < lc.VERBATIM_TURNS + lc.SUMMARIZE_BATCH
    assert "EARLIER (summary)" in ctx and "turn number 29" in ctx and "turn number 0" not in ctx
    assert m.last_said() == "turn number 29" and m.last_speaker == "me"
    print(f"30 turns -> {len(m.turns)} verbatim + a summary; the oldest turn is only in the summary")


def extra_cache_and_approve():
    ToolRunner._cache.clear()
    tools = FakeTools()
    r = ToolRunner(tools)
    r.act([{"id": "1", "name": "search_my_notes", "args": {"query": "x", "k": 3}}])
    o = r.act([{"id": "2", "name": "search_my_notes", "args": {"query": "x", "k": 3}}])[0]
    assert o["cached"] and tools.calls["search_my_notes"] == 1
    c = Controls(printer=lambda *_: None)
    for key, want in (("\r", "speak"), ("\x1b", "drop")):
        threading.Timer(0.1, c.handle_key, args=(key,)).start()
        assert c.approve("draft", "en", [], timeout=2) == want
    assert c.approve("draft", "en", [], timeout=0.2) == "drop"      # no key -> safe default
    threading.Timer(0.1, c.handle_key, args=("k",)).start()
    assert c.approve("draft", "en", [], timeout=2) == "drop" and c.muted
    print("repeat lookups hit the cache; Enter=speak, Esc=drop, no key=drop, kill while waiting=drop")


def extra_pipeline_smoke():
    """VAD -> loop -> log with fake audio, fake STT, fake decision, fake voice."""
    import tempfile
    from pathlib import Path
    from livelog import JsonlLogger
    from meeting_loop import MeetingLoop
    from schemas import TwinReply
    from vad import UtteranceDetector
    from context_mgr import LiveMemory

    rng = np.random.default_rng(0)
    speech = lambda s: [(0.3 * rng.standard_normal(lc.CHUNK)).astype(np.float32) for _ in range(int(s * 31.25))]
    silence = lambda s: [np.zeros(lc.CHUNK, np.float32) for _ in range(int(s * 31.25))]
    scenes = [speech(1.5) + silence(1.0), speech(1.2) + silence(1.0), speech(1.2) + silence(1.0)]

    holder = {}

    class FakeCapture:
        def __init__(self):
            self.i, self.k, self.t = 0, 0, time.monotonic()

        def get(self, timeout=0.2):
            if self.k >= len(scenes):
                time.sleep(timeout)
                return None
            if self.i == 0 and holder["loop"].handled < self.k:      # wait until the last utterance is done
                time.sleep(0.05)
                return None
            ch = scenes[self.k][self.i]
            self.i += 1
            if self.i >= len(scenes[self.k]):
                self.i, self.k = 0, self.k + 1
            self.t += lc.CHUNK / lc.SAMPLE_RATE
            return self.t, ch

        def stop(self):
            pass

    texts = [("Karim, what is overfitting?", "en"), ("Sara, share your screen please", "en"),
             ("Karim, how did you use Kubernetes?", "en")]
    plan = ["answer", "stay_silent", "defer"]
    cnt = {"i": 0}

    def stt(path):
        t, l = texts[cnt["i"]]
        return {"text": t, "language": l, "quality": 0.9}

    def decide(trn, mem):
        a = plan[cnt["i"]]
        cnt["i"] += 1
        br = [{"action": x, "kind": "llm", "score": 1.0 if x == a else 0.0, "threshold": 0.6,
               "status": "selected" if x == a else "pruned", "reason": "stub"}
              for x in ("ask_to_repeat", "stay_silent", "defer", "answer")]
        return {"selected": a, "language": "en", "addressed_to_me": a != "stay_silent", "answer_score": 0.9,
                "branches": br, "features": {}, "llm_reason": "stub", "fallback": False}

    def ans(trn, dec, mem):
        r = TwinReply(addressed_to_me=True, action="answer", language="en", reply_text="Overfitting is when a model memorizes the training set.",
                      confidence=0.8, tools_used=["search_my_notes"], sources=["ml_glossary.md"])
        return r, {"termination_reason": "model_stopped", "iterations": 2, "steps": []}

    def canned(dec, trn, mem):
        return TwinReply(addressed_to_me=True, action="defer", language="en", reply_text="Let me check that and get back to you.",
                         confidence=0.9), {}

    speaker, controls, sink = make_speaker(realtime=False)
    controls.mode = "auto"
    log = Path(tempfile.mkdtemp()) / "m.jsonl"
    memory = LiveMemory(summarizer=lambda p, o: "s", background=False)
    loop = MeetingLoop(FakeCapture(), UtteranceDetector(prob_fn=lambda ch: float(np.abs(ch).mean() > 0.05)),
                       speaker, controls, memory, JsonlLogger(log), stt_fn=stt, decide_fn=decide, answer_fn=ans,
                       canned_fn=canned, printer=lambda *_: None, max_minutes=1, idle_minutes=0.05)
    holder["loop"] = loop
    reason = loop.run()
    rows = [json.loads(l) for l in open(log, encoding="utf-8")]
    assert reason == "idle_timeout", reason
    assert [r["action"] for r in rows] == plan
    assert all(set(("vad_s", "stt_s", "decision_s")) <= set(r["timings"]) for r in rows)
    assert abs(rows[0]["timings"]["vad_s"] - 0.7) < 0.12, rows[0]["timings"]
    assert rows[0]["answer_loop"]["termination_reason"] == "model_stopped" and "answer_loop_s" in rows[0]["timings"]
    assert "first_audio_s" in rows[0]["timings"] and rows[1]["reply"] is None
    assert [h["text"] for h in speaker.history if h["text"] != "<filler>"] == [
        "Overfitting is when a model memorizes the training set.", "Let me check that and get back to you."]
    assert any(h["text"] == "<filler>" for h in speaker.history)       # auto mode: filler at the 'answer' decision
    assert len(memory.turns) == 5 and memory.turns[-1][0] == "me"      # 3 heard + 2 said
    import latency_report as lr
    assert "| Speech to text |" in lr.table(rows)
    print(f"3 fake utterances -> actions {[r['action'] for r in rows]}; vad_s={rows[0]['timings']['vad_s']}; "
          f"loop ended by {reason}; JSONL ok; filler played; stay_silent stayed silent")


# ------------------------------------------------------------------ real model (optional)
def live_llm_tests(audio_dir):
    from tools import call_tool
    for name, wrap in (("test1 (real model, real tools)", lambda n, **a: call_tool(n, **a)),
                       ("test3 (real model, search_my_notes raises)",
                        lambda n, **a: (_ for _ in ()).throw(RuntimeError("index offline")) if n == "search_my_notes" else call_tool(n, **a))):
        ToolRunner._cache.clear()
        calls = Counter()

        def runner(n, **a):
            calls[n] += 1
            return wrap(n, **a)
        reply, trace = answer(tr(EN_Q, "en"), decision("en"), Memory(), runner=runner)
        print(f"-- {name}: reason={trace['termination_reason']} iterations={trace['iterations']} "
              f"action={reply.action} calls={dict(calls)}")
        print("   reply:", reply.reply_text)
        if "test1" in name:
            assert trace["termination_reason"] in ("model_stopped", "goal_check_failed")
        else:
            assert calls["search_my_notes"] == 1 and reply.action in ("answer", "defer")


TESTS = [("1. Normal ending", test1_normal_ending), ("2. Never terminates", test2_never_terminates),
         ("3. Tool failure", test3_tool_failure), ("4. Goal-check", test4_goal_check),
         ("5. Kill switch", test5_kill_switch), ("extra. Context manager", extra_context_manager),
         ("extra. Cache + approve keys", extra_cache_and_approve), ("extra. Pipeline smoke test", extra_pipeline_smoke)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-llm", action="store_true")
    ap.add_argument("--audio-dir", default=None)
    a = ap.parse_args()
    buf, fails = io.StringIO(), 0

    def say(s=""):
        print(s)
        buf.write(s + "\n")

    for name, fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
            say(f"PASS  {name}")
            for line in out.getvalue().strip().splitlines():
                say("      " + line)
        except Exception:
            fails += 1
            say(f"FAIL  {name}")
            say("      " + traceback.format_exc().replace("\n", "\n      "))
    if a.live_llm:
        say("\n== live model ==")
        try:
            with redirect_stdout(io.StringIO()) as o:
                live_llm_tests(a.audio_dir)
            say(o.getvalue())
        except Exception:
            fails += 1
            say("FAIL  live model\n" + traceback.format_exc())
    say(f"\n{len(TESTS) - fails if not a.live_llm else len(TESTS)} checks run, {fails} failed")
    (lc.LIVE_DIR / "forced_tests_output.txt").write_text(buf.getvalue(), encoding="utf-8")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
