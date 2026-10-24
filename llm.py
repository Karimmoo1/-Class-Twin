import json
import re
import time

from functools import lru_cache

import config


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0):
    from langchain_groq import ChatGroq
    return ChatGroq(model=config.LLM_MODEL, temperature=temperature,
                    model_kwargs={"response_format": {"type": "json_object"}})


def _invoke_with_retry(messages, temperature, tries=5):
    """Groq's free tier has a tokens-per-minute limit. On a 429, wait and retry."""
    for attempt in range(tries):
        try:
            return get_llm(temperature).invoke(messages)
        except Exception as e:
            limited = "429" in str(e) or "rate_limit" in str(e)
            if limited and attempt < tries - 1:
                wait = 3 * (attempt + 1)
                print(f"[llm] rate limit, waiting {wait}s and retrying")
                time.sleep(wait)
            else:
                raise


def llm_json(system: str, user: str, temperature: float = 0.0):
    """Call the LLM in JSON mode. Returns a dict, or None if the output is not valid JSON."""
    msg = _invoke_with_retry([("system", system), ("user", user)], temperature)
    txt = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None
