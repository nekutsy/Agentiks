"""
Background I/O manager. Offloads blocking disk operations from the main agent loop:
  - Session JSON saves
  - Prompt dumps (write_current_input)
  - Heavy log writes (removed)
"""
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from logger_setup import get_logger


@dataclass
class _IOTask:
    kind: str           # 'save_session' | 'write_file' | 'shutdown'
    payload: Any
    callback: Optional[Callable] = None


class BackgroundIOManager:
    def __init__(self, max_queue_size: int = 1000):
        self._queue: "queue.Queue[_IOTask]" = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="bg-io")
        self._running = True
        self._tasks_processed = 0
        self._tasks_dropped = 0
        self._thread.start()

    # ---------- Public API ----------
    def save_session(self, path: str, data: dict):
        """Async save of session JSON (atomic write via tmp file)."""
        self._enqueue(_IOTask('save_session', (path, data)))

    def write_file(self, path: str, content: str):
        """Async write of arbitrary text file."""
        self._enqueue(_IOTask('write_file', (path, content)))

    def flush(self, timeout: float = 5.0):
        """Block until queue is empty or timeout."""
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)

    def shutdown(self, timeout: float = 5.0):
        """Graceful shutdown: drain queue, then stop worker."""
        self.flush(timeout)
        self._enqueue(_IOTask('shutdown', None))
        self._thread.join(timeout=timeout)
        self._running = False

    def stats(self) -> dict:
        return {
            'pending': self._queue.qsize(),
            'processed': self._tasks_processed,
            'dropped': self._tasks_dropped,
        }

    # ---------- Internals ----------
    def _enqueue(self, task: _IOTask):
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            self._tasks_dropped += 1
            if task.kind == 'save_session':
                get_logger().error("BackgroundIO queue full — CRITICAL session save dropped!")

    def _worker(self):
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task.kind == 'shutdown':
                break
            try:
                self._dispatch(task)
                self._tasks_processed += 1
            except Exception as e:
                get_logger().error(f"BackgroundIO task '{task.kind}' failed: {e}")

    def _dispatch(self, task: _IOTask):
        if task.kind == 'save_session':
            path, data = task.payload
            self._atomic_json_write(path, data)
        elif task.kind == 'write_file':
            path, content = task.payload
            self._text_write(path, content)

    @staticmethod
    def _atomic_json_write(path: str, data: dict):
        """Write JSON via tmp file + rename to avoid corruption on crash."""
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass
            raise

    @staticmethod
    def _text_write(path: str, content: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)