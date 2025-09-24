import os
import json
import threading
import time
from typing import Any, Dict, Optional


class TraceRecorder:
    """
    Lightweight, thread-safe JSONL tracer.

    Writes events to per-paper files located at:
      <run_output_dir>/<file_stem>/_trace.jsonl

    Each line is a JSON object with fields:
      ts, event, file_stem, topic, step, payload
    """

    def __init__(self, run_output_dir: str):
        self._base_dir = os.path.abspath(run_output_dir or ".")
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def for_file(self, file_stem: str) -> "FileTrace":
        return FileTrace(self, file_stem)

    def _get_lock(self, file_path: str) -> threading.Lock:
        with self._global_lock:
            lk = self._locks.get(file_path)
            if lk is None:
                lk = threading.Lock()
                self._locks[file_path] = lk
            return lk

    def _write_event(self, file_stem: str, event: str, payload: Dict[str, Any], topic: Optional[str] = None, step: Optional[str] = None) -> None:
        try:
            # Ensure parent dir exists (per-paper subdir already created by main scripts)
            out_dir = os.path.join(self._base_dir, file_stem)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "_trace.jsonl")
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "event": str(event),
                "file_stem": str(file_stem),
                "topic": (str(topic) if topic is not None else None),
                "step": (str(step) if step is not None else None),
                "payload": payload or {},
            }
            line = json.dumps(record, ensure_ascii=False)
            lock = self._get_lock(path)
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Never raise from tracing
            pass


class FileTrace:
    def __init__(self, recorder: TraceRecorder, file_stem: str):
        self._rec = recorder
        self._file = file_stem

    def record(self, event: str, payload: Dict[str, Any], topic: Optional[str] = None, step: Optional[str] = None) -> None:
        self._rec._write_event(self._file, event, payload, topic=topic, step=step)


