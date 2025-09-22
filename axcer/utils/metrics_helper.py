from functools import wraps
from datetime import datetime
import time
from typing import Any
from pathlib import Path

import json
import tiktoken
from axcer.utils.custom_logger import logger


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        print("getting encoder")
        encoder = tiktoken.encoding_for_model(model)
        print("finished encoder")
        logger.info(f"Successfully loaded encoding {encoder.name}")
    except KeyError:
        logger.warning(f"Unknown model '{model}', falling back to cl100k_base.")
        encoder = tiktoken.get_encoding("cl100k_base")

    tokens = encoder.encode(text)
    return len(tokens)


def get_next_id(data: dict[str, Any]) -> str:
    """Get the next ID from json key data"""
    if not data:
        return "0"
    return str(max(int(key) for key in data) + 1)


def load_existing_data(json_file: Path) -> dict[str, Any]:
    json_file.parent.mkdir(parents=True, exist_ok=True)

    if not json_file.exists():
        json_file.write_text("{}")
        return {}

    with open(json_file) as f:
        return json.load(f)


def track_runtime(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "enable_metrics_logging", True):
            return func(self, *args, **kwargs)

        start_time = time.perf_counter()
        runtime_set_early = False

        original_set_runtime = getattr(self, "_set_runtime_now", None)

        def set_runtime_now():
            nonlocal runtime_set_early
            current_time = time.perf_counter()
            self.total_runtime = float(f"{current_time - start_time:.3f}")
            self.runtime_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            runtime_set_early = True  # Mark that runtime was set early

        self._set_runtime_now = set_runtime_now

        try:
            results = func(self, *args, **kwargs)
        finally:
            if original_set_runtime:
                self._set_runtime_now = original_set_runtime
            else:
                delattr(self, "_set_runtime_now")

        if not runtime_set_early:
            end_time = time.time()
            self.total_runtime = float(f"{end_time - start_time:.2f}")
            self.runtime_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        return results

    return wrapper
