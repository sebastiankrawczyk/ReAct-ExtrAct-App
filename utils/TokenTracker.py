import os
import time
import json
from typing import Any, Dict, Optional

try:
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
except Exception:  # Fallback if callbacks are unavailable
    CallbackManager = None
    TokenCountingHandler = None


class TokenTracker:
    def __init__(self) -> None:
        self._handler = TokenCountingHandler() if TokenCountingHandler else None
        self._manager = CallbackManager([self._handler]) if (CallbackManager and self._handler) else None
        self._start_time: Optional[float] = None

    def install(self) -> None:
        if not self._manager:
            return
        try:
            from llama_index.core import Settings
            Settings.callback_manager = self._manager
        except Exception:
            pass

    def start(self) -> None:
        self._start_time = time.time()

    def _safe_get(self, obj: Any, names: list[str]) -> Optional[int]:
        for n in names:
            try:
                v = getattr(obj, n, None)
                if v is not None:
                    return int(v)
            except Exception:
                continue
        return None

    def report(self) -> Dict[str, Any]:
        duration = (time.time() - self._start_time) if self._start_time else None
        data: Dict[str, Any] = {
            "duration_seconds": duration,
            "total_llm_token_count": None,
            "total_embedding_token_count": None,
            "total_prompt_tokens": None,
            "total_completion_tokens": None,
            "total_token_count": None,
            "callbacks_available": bool(self._handler is not None),
        }
        if self._handler is not None:
            data["total_llm_token_count"] = self._safe_get(
                self._handler,
                [
                    "total_llm_token_count",
                    "total_llm_token_usage",
                ],
            )
            data["total_embedding_token_count"] = self._safe_get(
                self._handler,
                [
                    "total_embedding_token_count",
                    "total_embedding_token_usage",
                ],
            )
            data["total_prompt_tokens"] = self._safe_get(
                self._handler,
                [
                    "total_prompt_tokens",
                    "total_prompt_token_count",
                ],
            )
            data["total_completion_tokens"] = self._safe_get(
                self._handler,
                [
                    "total_completion_tokens",
                    "total_completion_token_count",
                ],
            )
            data["total_token_count"] = self._safe_get(
                self._handler,
                [
                    "total_token_count",
                    "total_tokens",
                ],
            )
        return data

    def write_report(self, output_dir: str, filename: str = "usage.json") -> None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(self.report(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass


