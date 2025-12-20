import json
import time
import tracemalloc
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
import tiktoken
from axcer.utils.custom_logger import logger


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        print("getting encoder")
        encoder = tiktoken.encoding_for_model(model)
        # print("finished encoder")
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


def preprocess_text(text):
    words = []
    for word in text.split():
        cleaned_word = word.strip("?!,.")
        if cleaned_word:
            words.append(cleaned_word)
    return words


def track_runtime_and_memory(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "enable_metrics_logging", True):
            return func(self, *args, **kwargs)

        start_time = time.perf_counter()
        runtime_set_early = False
        memory_set_early = False

        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()

        original_set_runtime = getattr(self, "_set_runtime_now", None)
        original_set_memory = getattr(self, "_set_memory_now", None)

        def _set_runtime_now():
            nonlocal runtime_set_early
            current_time = time.perf_counter()
            self.total_runtime = float(f"{current_time - start_time:.6f}")
            self.runtime_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            runtime_set_early = True

        def _set_memory_now():
            nonlocal memory_set_early
            end_snapshot = tracemalloc.take_snapshot()
            # Calculate memory allocated in bytes
            stats = end_snapshot.compare_to(start_snapshot, "lineno")
            total_alloc = sum(stat.size_diff for stat in stats)
            self.total_memory_kb = float(f"{total_alloc / 1024:.3f}")
            memory_set_early = True

        # Attach hooks to instance
        self._set_runtime_now = _set_runtime_now
        self._set_memory_now = _set_memory_now

        try:
            results = func(self, *args, **kwargs)
        finally:
            if original_set_runtime:
                self._set_runtime_now = original_set_runtime
            else:
                delattr(self, "_set_runtime_now")

            if original_set_memory:
                self._set_memory_now = original_set_memory
            else:
                delattr(self, "_set_memory_now")

        # --- Set metrics if hooks not called ---
        if not runtime_set_early:
            end_time = time.perf_counter()
            self.total_runtime = float(f"{end_time - start_time:.6f}")
            self.runtime_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        if not memory_set_early:
            end_snapshot = tracemalloc.take_snapshot()
            stats = end_snapshot.compare_to(start_snapshot, "lineno")
            total_alloc = sum(stat.size_diff for stat in stats)
            self.total_memory_kb = float(f"{total_alloc / 1024:.3f}")

        tracemalloc.stop()

        return results

    return wrapper


# def track_runtime(func):
#     @wraps(func)
#     def wrapper(self, *args, **kwargs):
#         if not getattr(self, "enable_metrics_logging", True):
#             return func(self, *args, **kwargs)
#
#         start_time = time.perf_counter()
#         runtime_set_early = False
#
#         original_set_runtime = getattr(self, "_set_runtime_now", None)
#
#         def set_runtime_now():
#             nonlocal runtime_set_early
#             current_time = time.perf_counter()
#             self.total_runtime = float(f"{current_time - start_time:.3f}")
#             self.runtime_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
#             runtime_set_early = True  # Mark that runtime was set early
#
#         self._set_runtime_now = set_runtime_now
#
#         try:
#             results = func(self, *args, **kwargs)
#         finally:
#             if original_set_runtime:
#                 self._set_runtime_now = original_set_runtime
#             else:
#                 delattr(self, "_set_runtime_now")
#
#         if not runtime_set_early:
#             end_time = time.time()
#             self.total_runtime = float(f"{end_time - start_time:.2f}")
#             self.runtime_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
#
#         return results
#
#     return wrapper
