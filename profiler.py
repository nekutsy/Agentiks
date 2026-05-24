
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

from logger_setup import get_logger


class PerformanceTracker:
    def __init__(self, report_every_n_steps: int = 10):
        self._lock = threading.Lock()
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._steps = 0
        self._report_every = report_every_n_steps
        self._session_start = time.monotonic()
        # Ollama-native metrics
        self._total_prompt_tokens = 0
        self._total_gen_tokens = 0
        self._total_prefill_sec = 0.0
        self._total_decode_sec = 0.0

    # ---------- Recording ----------
    @contextmanager
    def measure(self, name: str):
        """Context manager: measure wall-clock time of a block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.record(name, elapsed)

    def record(self, name: str, seconds: float):
        with self._lock:
            self._samples[name].append(seconds)

    def increment(self, counter: str, amount: int = 1):
        with self._lock:
            self._counters[counter] += amount

    def record_ollama_metrics(self, response_or_chunk: dict):
        if not isinstance(response_or_chunk, dict):
            return
        prefill_ns = response_or_chunk.get('prompt_eval_duration', 0)
        decode_ns = response_or_chunk.get('eval_duration', 0)
        prompt_tokens = response_or_chunk.get('prompt_eval_count', 0)
        gen_tokens = response_or_chunk.get('eval_count', 0)

        if prefill_ns or decode_ns or prompt_tokens or gen_tokens:
            with self._lock:
                if prefill_ns:
                    self._total_prefill_sec += prefill_ns / 1e9
                if decode_ns:
                    self._total_decode_sec += decode_ns / 1e9
                self._total_prompt_tokens += prompt_tokens
                self._total_gen_tokens += gen_tokens

    def step(self):
        with self._lock:
            self._steps += 1
            should_report = (self._steps % self._report_every) == 0
        if should_report:
            self.report()

    # ---------- Reporting ----------
    def report(self):
        with self._lock:
            samples_snapshot = {k: list(v) for k, v in self._samples.items()}
            counters_snapshot = dict(self._counters)
            steps = self._steps
            total_prefill = self._total_prefill_sec
            total_decode = self._total_decode_sec
            total_prompt_tok = self._total_prompt_tokens
            total_gen_tok = self._total_gen_tokens

        log = get_logger()
        log.info("=" * 60)
        log.info(f"📊 PERFORMANCE REPORT (step {steps})")
        log.info(f"  Session uptime: {time.monotonic() - self._session_start:.1f}s")

        if samples_snapshot:
            log.info("  --- Stage timings (seconds) ---")
            rows = []
            for name, values in sorted(samples_snapshot.items()):
                n = len(values)
                total = sum(values)
                avg = total / n
                p95 = sorted(values)[int(n * 0.95)] if n >= 5 else max(values)
                rows.append((name, n, total, avg, p95))
            rows.sort(key=lambda r: r[2], reverse=True)  # by total time
            for name, n, total, avg, p95 in rows:
                log.info(f"    {name:25s}  n={n:4d}  total={total:7.2f}s  "
                         f"avg={avg:.3f}s  p95={p95:.3f}s")

        # Ollama-native stats
        if total_prompt_tok or total_gen_tok:
            log.info("  --- Ollama inference ---")
            log.info(f"    Prompt tokens:  {total_prompt_tok}")
            log.info(f"    Generated tokens: {total_gen_tok}")
            if total_prefill > 0:
                tok_per_s = total_prompt_tok / total_prefill if total_prefill else 0
                log.info(f"    Prefill: {total_prefill:.2f}s  ({tok_per_s:.0f} tok/s)")
            if total_decode > 0:
                tok_per_s = total_gen_tok / total_decode if total_decode else 0
                log.info(f"    Decode:  {total_decode:.2f}s  ({tok_per_s:.1f} tok/s)")

        if counters_snapshot:
            log.info("  --- Counters ---")
            for k, v in sorted(counters_snapshot.items()):
                log.info(f"    {k}: {v}")

        log.info("=" * 60)

    def summary_dict(self) -> dict:
        with self._lock:
            return {
                'steps': self._steps,
                'uptime_sec': time.monotonic() - self._session_start,
                'prompt_tokens': self._total_prompt_tokens,
                'gen_tokens': self._total_gen_tokens,
                'prefill_sec': self._total_prefill_sec,
                'decode_sec': self._total_decode_sec,
                'stages': {k: {'n': len(v), 'total': sum(v)}
                           for k, v in self._samples.items()},
                'counters': dict(self._counters),
            }


# Module-level singleton (easy to import)
_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()


def get_profiler() -> PerformanceTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = PerformanceTracker()
    return _tracker