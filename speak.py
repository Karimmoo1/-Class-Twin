import asyncio
import threading
import time

import config
from schemas import TwinReply


def _run(coro):
    """Works inside Colab/Jupyter too (where an event loop is already running)."""
    err = []

    def target():
        try:
            asyncio.run(coro)
        except Exception as e:   # noqa
            err.append(e)
    t = threading.Thread(target=target)
    t.start()
    t.join()
    if err:
        raise err[0]


def speak(reply: TwinReply, muted: bool = False):
    """Save the reply as an mp3 with edge-tts. Voice depends on reply.language.
    Returns the file path, or None when there is nothing to say (stay_silent) or the kill switch is on."""
    if muted or reply.action == "stay_silent" or not reply.reply_text.strip():
        return None
    path = config.OUT_DIR / "audio" / f"reply_{int(time.time() * 1000)}.mp3"
    try:
        import edge_tts
        _run(edge_tts.Communicate(reply.reply_text, config.VOICES[reply.language]).save(str(path)))
        return str(path)
    except Exception as e:
        print(f"[speak] edge-tts failed ({type(e).__name__}: {str(e)[:80]}), trying gTTS")
    try:
        from gtts import gTTS
        gTTS(reply.reply_text, lang="en" if reply.language == "en" else "ar").save(str(path))
        return str(path)
    except Exception as e:
        print(f"[speak] gTTS failed too ({type(e).__name__}), continuing without audio")
        return None
