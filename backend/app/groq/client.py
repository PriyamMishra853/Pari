"""Thin Groq client: JSON chat with model fallback, and speech-to-text.

Credentials come from GROQ_API_KEY in .env (never hard-coded).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

_ASCII = str.maketrans({
    " ": " ", " ": " ", " ": " ", "‑": "-", "‐": "-", "–": "-", "—": " - ",
    "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...", "°": " degrees",
})


def clean(text: str) -> str:
    return (text or "").translate(_ASCII).strip()


class GroqClient:
    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
        except Exception:
            pass
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.fallback_model = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
        self.stt_model = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        self._client = None
        self.last_error: str | None = None
        self.calls = 0
        self.latencies: list[float] = []
        if self.api_key:
            try:
                from groq import Groq

                self._client = Groq(api_key=self.api_key, timeout=10.0, max_retries=1)
            except Exception as exc:  # pragma: no cover
                self.last_error = f"groq SDK unavailable: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    def status(self) -> dict[str, Any]:
        lat = sorted(self.latencies[-50:])
        return {
            "available": self.available,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "stt_model": self.stt_model,
            "calls": self.calls,
            "median_latency_ms": round(lat[len(lat) // 2]) if lat else None,
            "last_error": self.last_error,
            "reason": None if self.available else "GROQ_API_KEY missing in .env",
        }

    def chat_json(self, system: str, user: str, max_tokens: int = 500, timeout: float = 8.0,
                  allow_fallback: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not self.available:
            return None, {"source": "unavailable", "error": "GROQ_API_KEY missing"}
        models = (self.model, self.fallback_model) if allow_fallback else (self.model,)
        for model in models:
            t0 = time.perf_counter()
            try:
                kw: dict[str, Any] = dict(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                    temperature=0.2,
                )
                if model.startswith("openai/gpt-oss"):
                    kw["reasoning_effort"] = "low"
                resp = self._client.with_options(timeout=timeout, max_retries=0).chat.completions.create(**kw)
                ms = (time.perf_counter() - t0) * 1000.0
                self.calls += 1
                self.latencies.append(ms)
                data = json.loads(resp.choices[0].message.content or "{}")
                for k, v in list(data.items()):
                    if isinstance(v, str):
                        data[k] = clean(v)
                    elif isinstance(v, list):
                        data[k] = [clean(x) if isinstance(x, str) else x for x in v]
                self.last_error = None
                return data, {"source": f"groq:{model}", "latency_ms": round(ms)}
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        return None, {"source": "groq-error", "error": self.last_error}

    def transcribe(self, audio: bytes, filename: str, prompt: str = "") -> tuple[str | None, dict[str, Any]]:
        if not self.available:
            return None, {"source": "unavailable", "error": "GROQ_API_KEY missing"}
        t0 = time.perf_counter()
        try:
            r = self._client.audio.transcriptions.create(
                file=(filename, audio), model=self.stt_model, prompt=prompt[:800], language="en",
                response_format="json", temperature=0.0,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            return clean(getattr(r, "text", "") or ""), {"source": f"groq:{self.stt_model}", "latency_ms": round(ms)}
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            return None, {"source": "groq-error", "error": self.last_error}


_client: GroqClient | None = None


def get_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient()
    return _client
