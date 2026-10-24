"""One JSON line per utterance."""
import json
import threading
import time

import live_config as lc


class JsonlLogger:
    def __init__(self, path=None):
        self.path = path or (lc.LIVE_DIR / f"meeting_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self._lock = threading.Lock()

    def write(self, record: dict):
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
