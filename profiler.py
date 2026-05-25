import json
import os
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
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
        self._total_prompt_tokens = 0
        self._total_gen_tokens = 0
        self._total_prefill_sec = 0.0
        self._total_decode_sec = 0.0

    @contextmanager
    def measure(self, name: str):
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
            rows.sort(key=lambda r: r[2], reverse=True)
            for name, n, total, avg, p95 in rows:
                log.info(f"    {name:25s}  n={n:4d}  total={total:7.2f}s  avg={avg:.3f}s  p95={p95:.3f}s")

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
                'stages': {k: {'n': len(v), 'total': sum(v)} for k, v in self._samples.items()},
                'counters': dict(self._counters),
            }

    def update_aggregated_metrics(self, model_name: str, filepath: str = "metrics.json"):
        with self._lock:
            total_time = time.monotonic() - self._session_start
            total_prompt_tokens = self._counters.get('ollama_prompt_tokens', 0)
            total_gen_tokens = self._counters.get('ollama_stream_tokens', 0)
            total_prefill_sec = self._total_prefill_sec
            total_decode_sec = self._total_decode_sec

            tool_calls = {}
            for key, count in self._counters.items():
                if key.startswith('tool_calls:'):
                    tool_name = key[len('tool_calls:'):]
                    tool_calls[tool_name] = count

            tool_total_duration = {}
            tool_count_calls = {}
            for name, durations in self._samples.items():
                if name.startswith('tool:'):
                    tool_name = name[len('tool:'):]
                    tool_total_duration[tool_name] = sum(durations)
                    tool_count_calls[tool_name] = len(durations)

        existing = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    if not isinstance(existing, dict):
                        existing = {}
            except (json.JSONDecodeError, IOError):
                existing = {}

        model_data = existing.get(model_name, {
            "model": model_name,
            "run_count": 0,
            "total_steps": 0,
            "total_time_sec": 0.0,
            "total_prompt_tokens": 0,
            "total_gen_tokens": 0,
            "total_prefill_sec": 0.0,
            "total_decode_sec": 0.0,
            "tool_calls_total": {},
            "tool_total_duration_sec": {},
        })

        model_data["run_count"] += 1
        model_data["total_steps"] += self._steps
        model_data["total_time_sec"] += total_time
        model_data["total_prompt_tokens"] += total_prompt_tokens
        model_data["total_gen_tokens"] += total_gen_tokens
        model_data["total_prefill_sec"] += total_prefill_sec
        model_data["total_decode_sec"] += total_decode_sec

        for tool_name, cnt in tool_calls.items():
            model_data["tool_calls_total"][tool_name] = model_data["tool_calls_total"].get(tool_name, 0) + cnt

        for tool_name, dur in tool_total_duration.items():
            model_data["tool_total_duration_sec"][tool_name] = model_data["tool_total_duration_sec"].get(tool_name, 0.0) + dur

        model_data["avg_prompt_tok_per_sec"] = (
            model_data["total_prompt_tokens"] / model_data["total_prefill_sec"]
            if model_data["total_prefill_sec"] > 0 else 0.0
        )
        model_data["avg_gen_tok_per_sec"] = (
            model_data["total_gen_tokens"] / model_data["total_decode_sec"]
            if model_data["total_decode_sec"] > 0 else 0.0
        )
        model_data["avg_tool_duration_sec"] = {}
        for tool_name, total_dur in model_data["tool_total_duration_sec"].items():
            call_count = model_data["tool_calls_total"].get(tool_name, 0)
            model_data["avg_tool_duration_sec"][tool_name] = total_dur / call_count if call_count > 0 else 0.0

        model_data["last_run_timestamp"] = datetime.now().isoformat()

        existing[model_name] = model_data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        get_logger().info(f"Aggregated metrics for model '{model_name}' updated in {filepath} (total runs: {model_data['run_count']})")

    def save_metrics_to_file(self, model_name: str, filepath: str = "metrics.json"):
        self.update_aggregated_metrics(model_name, filepath)


_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()

def get_profiler() -> PerformanceTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = PerformanceTracker()
    return _tracker