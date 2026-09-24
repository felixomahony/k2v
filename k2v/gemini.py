"""Gemini implementations of the two model interfaces, as used in the paper.

  GeminiLanguage  gemini-3.8-flash, temperature 0.2, thinking level HIGH
  GeminiImage     gemini-3.1-flash-image

Both read the API key from $GEMINI_API_KEY and use the Gemini Developer API.
"""

from __future__ import annotations

import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

from .models import ModelError, ToolCall, downscale

#: Transient failures worth waiting out: overload, rate limits, server errors.
_RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
              "500", "INTERNAL", "deadline", "timeout")
_RETRY_HINT = re.compile(r"retry in ([0-9.]+)s", re.I)


def _client(api_key_env: str):
    try:
        from google import genai
    except ImportError as exc:
        raise ModelError("google-genai is not installed: pip install google-genai") from exc
    key = os.environ.get(api_key_env)
    if not key:
        raise ModelError(f"no API key: export {api_key_env}")
    # vertexai=False pins the Developer API even if GOOGLE_GENAI_USE_VERTEXAI is set.
    return genai.Client(api_key=key, vertexai=False)


class _NoImage(ModelError):
    """The image model answered without an image. It usually does on the next try."""


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, _NoImage):
        return True
    text = str(exc)
    if "limit: 0" in text or "PerDay" in text:     # a daily quota will not clear by waiting
        return False
    return any(token in text for token in _RETRYABLE)


def _backoff(attempt: int, exc: Exception | None) -> float:
    if exc is not None and (m := _RETRY_HINT.search(str(exc))):
        return min(90.0, float(m.group(1)) + 1.0)
    return min(60.0, 4.0 * (2 ** attempt) * (0.75 + 0.5 * random.random()))


def _with_retries(call, what: str, retries: int):
    last = None
    for attempt in range(retries):
        try:
            return call()
        except Exception as exc:
            last = exc
            if not _retryable(exc):
                break
        if attempt < retries - 1:
            time.sleep(_backoff(attempt, last))
    raise ModelError(f"{what} failed after {retries} attempts: {last}")


def _image_part(path: Path):
    from google.genai import types
    path = downscale(Path(path))
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def _parts(response):
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        yield from getattr(content, "parts", None) or []


class GeminiLanguage:
    def __init__(self, model: str = "gemini-3.8-flash", *, temperature: float = 0.2,
                 thinking_level: str = "HIGH", api_key_env: str = "GEMINI_API_KEY",
                 retries: int = 6) -> None:
        self.model, self.temperature = model, temperature
        self.thinking_level, self.retries = thinking_level, retries
        self._api_key_env, self._client = api_key_env, None

    def _contents(self, turns):
        from google.genai import types
        contents: list[Any] = []
        for turn in turns:
            if turn.get("raw") is not None:
                contents.append(turn["raw"])            # replayed exactly as it came back
                continue
            if turn.get("role") == "tool":
                parts = [types.Part.from_function_response(
                    name=turn["name"], response={"result": turn.get("result", "")})]
                parts += [_image_part(p) for p in turn.get("images") or ()]
                contents.append(types.Content(role="user", parts=parts))
                continue
            parts = [_image_part(p) for p in turn.get("images") or ()]
            if turn.get("text"):
                parts.append(types.Part.from_text(text=turn["text"]))
            parts += [types.Part.from_function_call(name=c["name"], args=c.get("args") or {})
                      for c in turn.get("calls") or ()]
            contents.append(types.Content(role=turn.get("role", "user"), parts=parts))
        return contents

    def converse(self, turns, tools, *, force=None) -> ToolCall:
        from google.genai import types
        self._client = self._client or _client(self._api_key_env)
        config: dict[str, Any] = {
            "temperature": self.temperature,
            "tools": [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters=t["parameters"]) for t in tools])],
            "thinking_config": types.ThinkingConfig(include_thoughts=True,
                                                    thinking_level=self.thinking_level),
            # The tools are declarations the planner executes itself; the SDK's own
            # automatic function calling never applies, so switch it (and its warning) off.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if force:
            config["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=list(force)))

        def call():
            response = self._client.models.generate_content(
                model=self.model, contents=self._contents(turns),
                config=types.GenerateContentConfig(**config))
            said, fn = "", None
            for part in _parts(response):
                if getattr(part, "function_call", None) and fn is None:
                    fn = part.function_call
                elif getattr(part, "text", None) and not getattr(part, "thought", False):
                    said += part.text
            raw = getattr(response.candidates[0], "content", None)
            if fn is None:
                return ToolCall(text=said.strip(), raw=raw)
            return ToolCall(name=fn.name, args=dict(fn.args or {}), text=said.strip(), raw=raw)

        return _with_retries(call, self.model, self.retries)


class GeminiImage:
    def __init__(self, model: str = "gemini-3.1-flash-image", *,
                 api_key_env: str = "GEMINI_API_KEY", retries: int = 6) -> None:
        self.model, self.retries = model, retries
        self._api_key_env, self._client = api_key_env, None

    def edit(self, instruction: str, images: Sequence[Path], out_path: Path) -> Path:
        self._client = self._client or _client(self._api_key_env)
        contents: list[Any] = [_image_part(p) for p in images] + [instruction]

        def call():
            response = self._client.models.generate_content(model=self.model,
                                                            contents=contents)
            for part in _parts(response):
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(inline.data)
                    return out_path
            raise _NoImage("response carried no image")

        return _with_retries(call, self.model, self.retries)
