"""The two model interfaces the keyframe planner talks to.

K2V needs exactly two capabilities from outside:

  LanguageModel.converse  one turn of a tool-using conversation: given the history and
                          the tools on offer, return the tool the model wants to call.
  ImageModel.edit         an instruction plus zero or more images in, one image out.
                          With no images it is text-to-image.

`k2v/gemini.py` implements both with the models used in the paper. To use another
provider, implement these two methods; nothing else in K2V needs to change.

Conversation turns are plain dicts:
  {"role": "user",  "text": str, "images": [Path, ...]}
  {"role": "model", "text": str, "raw": <provider object>, "calls": [{"name", "args"}]}
  {"role": "tool",  "name": str, "result": str, "images": [Path, ...]}
A "model" turn carries `raw`, the provider's own representation, so it can be sent back
verbatim (Gemini signs function calls and refuses a rebuilt one).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2


class ModelError(RuntimeError):
    """A provider is unusable: missing package, missing key, or a failed request."""


@dataclass
class ToolCall:
    """The model's reply to one turn. No `name` means it answered in prose."""

    name: str = ""
    args: dict = field(default_factory=dict)
    text: str = ""
    raw: Any = None


class LanguageModel(Protocol):
    def converse(self, turns: list[dict], tools: list[dict], *,
                 force: list[str] | None = None) -> ToolCall:
        """`tools` are JSON-schema function declarations ({name, description,
        parameters}). `force`, when given, restricts the reply to a call of one of
        those tool names."""
        ...


class ImageModel(Protocol):
    def edit(self, instruction: str, images: Sequence[Path], out_path: Path) -> Path:
        """Write the result to `out_path` and return it."""
        ...


#: Images sent to a model are capped at this long edge. Benchmark frames can be 4K;
#: the extra detail cannot survive into a 720p video and only costs input tokens.
LONG_EDGE = 1920
_CACHE = Path(tempfile.gettempdir()) / "k2v_resized"


def downscale(path: Path, long_edge: int = LONG_EDGE) -> Path:
    """`path`, or a cached copy no larger than `long_edge` on its long side."""
    path = Path(path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return path
    stat = os.stat(path)
    key = hashlib.sha256(
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{long_edge}".encode()
    ).hexdigest()[:20]
    _CACHE.mkdir(parents=True, exist_ok=True)
    out = _CACHE / f"{key}{path.suffix or '.jpg'}"
    if not out.exists():
        scale = long_edge / max(h, w)
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out
