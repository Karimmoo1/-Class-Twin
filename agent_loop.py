"""Session 4 pattern: Observe -> Think -> Act, with a hard cap and a deterministic goal-check.

run_agent_loop(goal, think_fn, act_fn, max_iterations) is the lab pattern. Two small additions:
  - goal_check_fn: runs on every draft BEFORE anything is spoken (retry once, then give up)
  - the full context (every think and every observation) is returned for logging
Three ways the loop ends (result.reason):
  "model_stopped"      the model stopped calling tools and its draft passed the goal-check
  "max_iterations"     the cap was reached (the caller speaks a safe fallback)
  "goal_check_failed"  the draft failed the goal-check twice (the caller defers)
"""
import time
from dataclasses import dataclass, field


@dataclass
class LoopResult:
    reason: str
    iterations: int
    draft: dict | None = None
    steps: list = field(default_factory=list)
    context: list = field(default_factory=list)
    goal_check: dict | None = None
    retries: int = 0
    seconds: float = 0.0


def run_agent_loop(goal, think_fn, act_fn, max_iterations=4, goal_check_fn=None,
                   max_check_retries=1, context=None) -> LoopResult:
    """think_fn(goal, context) -> assistant message {"content": str, "tool_calls": [{"id","name","args"}]}
    act_fn(tool_calls, context) -> list of observations {"tool_call_id","name","content",...}
    Every observation is appended to the context, so the next think step sees it."""
    t0 = time.perf_counter()
    ctx = list(context) if context is not None else [{"role": "user", "content": goal}]
    steps, retries, draft, last_check = [], 0, None, None

    def result(reason, i):
        return LoopResult(reason, i, draft, steps, ctx, last_check, retries, round(time.perf_counter() - t0, 3))

    for i in range(1, max_iterations + 1):
        t_think = time.perf_counter()
        msg = think_fn(goal, ctx)                                   # THINK
        msg = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []}
        ctx.append(msg)
        step = {"iteration": i, "think_s": round(time.perf_counter() - t_think, 3), "assistant": msg}
        steps.append(step)

        if msg["tool_calls"]:                                       # ACT
            obs = act_fn(msg["tool_calls"], ctx)
            step["tool_results"] = obs
            for o in obs:                                           # tool errors are observations too
                ctx.append({"role": "tool", "tool_call_id": o["tool_call_id"], "name": o["name"],
                            "content": o["content"]})
            continue

        draft = msg                                                 # the model stopped calling tools
        if goal_check_fn is None:
            return result("model_stopped", i)
        last_check = goal_check_fn(draft)                           # deterministic check, my code
        step["goal_check"] = last_check
        if last_check["ok"]:
            return result("model_stopped", i)
        if retries >= max_check_retries:
            return result("goal_check_failed", i)
        retries += 1
        ctx.append({"role": "user", "content":
                    "GOAL CHECK FAILED: " + "; ".join(last_check["failed"]) +
                    ". Fix exactly this and reply with the JSON only."})
    return result("max_iterations", max_iterations)
