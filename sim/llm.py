"""Minimal OpenAI-compatible LLM client with JSON output, disk cache and transcript."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_JSON_CLAUSE = (
    "Return exactly one JSON object. Do not output explanations, markdown fences, "
    "or any text outside the JSON object."
)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class LLMClient:
    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        cache_path: Path | None = None,
        transcript_path: Path | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env)
        self.cache_path = Path(cache_path) if cache_path else None
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.calls = 0
        self.tokens = 0
        self.cache_hits = 0
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._cache[row["key"]] = row["value"]
                except Exception:  # noqa: BLE001
                    continue
        for p in (self.cache_path, self.transcript_path):
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        full_system = system.rstrip() + "\n\n" + _JSON_CLAUSE + "\nJSON schema:\n" + schema_hint.strip()
        key = hashlib.sha256(
            (self.model + "|" + str(self.temperature) + "|" + full_system + "|" + user).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if not self.api_key:
            raise LLMError(
                f"Environment variable {self.api_key_env!r} is not set; cannot run the LLM branch."
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_s)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"HTTP {resp.status_code}", raw=resp.text[:500])
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}", raw=resp.text[:1000])
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                tok = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
                obj = _parse_json(content)
                self.calls += 1
                self.tokens += tok
                self._cache[key] = obj
                if self.cache_path:
                    with self.cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "value": obj}, ensure_ascii=False) + "\n")
                if self.transcript_path:
                    with self.transcript_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                            "model": self.model,
                            "system": full_system,
                            "user": user,
                            "response": obj,
                            "tokens": tok,
                        }, ensure_ascii=False) + "\n")
                return obj
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        raise LLMError(f"LLM call failed after 3 attempts: {last_exc}")


def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t.lstrip("`")
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j <= i:
            raise LLMError("response is not JSON", raw=text[:1000])
        obj = json.loads(t[i:j + 1])
    if not isinstance(obj, dict):
        raise LLMError("response JSON is not an object", raw=text[:1000])
    return obj
