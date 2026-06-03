from __future__ import annotations

import time
from datetime import datetime


def log(message: str, *, tag: str = "INFO") -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{tag}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def elapsed_str(self) -> str:
        return format_duration(self.elapsed())


def should_log(index: int, total: int | None = None, every: int = 0) -> bool:
    if index <= 1:
        return True
    if total is not None and index >= total:
        return True
    return every > 0 and index % every == 0
