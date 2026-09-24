"""Wan2.2, conditioned on the keyframe stack.

Finds a Wan2.2 checkout (the bundled submodule, or your own), turns the planner's
keyframes into the frames each model conditions on, and runs generation with the
settings used in the paper:

  14B  Wan2.2-I2V-A14B   1280x720, 81 frames at 16 fps, 40 steps, guide 3.5, shift 5.0
  5B   Wan2.2-TI2V-5B    1280x704, 121 frames at 24 fps, 50 steps, guide 5.0, shift 5.0

Where Wan is looked for, in order: `wan_dir=` / --wan-dir, $K2V_WAN_DIR, then
third_party/Wan2.2. Checkpoints: `ckpt_dir=` / --ckpt-dir, $K2V_CKPT_DIR, then
<wan>/Wan2.2-I2V-A14B or <wan>/Wan2.2-TI2V-5B (where Wan's README downloads them).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .planner import FPS as PLAN_FPS, Keyframe

REPO = Path(__file__).resolve().parent.parent
SUBMODULE = REPO / "third_party" / "Wan2.2"

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，细节模糊不清，字幕，风格，作品，画作，画面，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走")

SETTINGS = {
    "14B": dict(task="i2v-A14B", ckpt="Wan2.2-I2V-A14B", size="1280*720", frame_num=81,
                fps=16, steps=40, guide_scale=3.5, shift=5.0),
    "5B": dict(task="ti2v-5B", ckpt="Wan2.2-TI2V-5B", size="1280*704", frame_num=121,
               fps=24, steps=50, guide_scale=5.0, shift=5.0),
}

#: A held span -- the scene claimed to be still between two keyframes -- is conditioned
#: as still throughout: the same image is repeated at this interval across the span,
#: not only pinned at its two ends.
FILL_EVERY = 1.0


def find_wan(wan_dir: str | Path | None = None) -> Path:
    """Locate a Wan2.2 checkout and make its `wan` package importable."""
    candidates = [wan_dir, os.environ.get("K2V_WAN_DIR"), SUBMODULE]
    for c in filter(None, candidates):
        path = Path(c).expanduser().resolve()
        if (path / "wan" / "__init__.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise FileNotFoundError(
        "no Wan2.2 checkout found. Clone this repo with --recursive (or run "
        "`git submodule update --init`), or point --wan-dir / $K2V_WAN_DIR at an "
        "existing Wan2.2 directory.")


def schedule(keyframes: list[Keyframe]) -> list[tuple[float, Path]]:
    """(seconds, image) pairs to condition on, with held spans filled."""
    out, previous = [], 0.0
    for k in keyframes:
        if k.held:
            t = previous + FILL_EVERY
            while t < k.seconds:
                out.append((t, k.image))
                t += FILL_EVERY
        out.append((k.seconds, k.image))
        previous = k.seconds
    return out


def to_frames(times: list[tuple[float, Path]], model: str) -> dict[int, Path]:
    """Seconds -> the frame index each model conditions, on its own clock.

    Frames are multiples of 4 (a latent covers 4 frames). The planner works at 16 fps,
    so for the 14B this is exact; the 5B runs at 24 fps and is re-timed by the second,
    so a keyframe at 5.0 s stays at the end of the clip.
    """
    s = SETTINGS[model]
    last = s["frame_num"] - 1 - (s["frame_num"] - 1) % 4
    out: dict[int, Path] = {}
    for seconds, image in times:
        frame = round(seconds * PLAN_FPS)                 # the planner's own frame
        anchor = int(round(s["fps"] * (frame / PLAN_FPS) / 4)) * 4
        if model == "5B":
            anchor = max(4, min(last, anchor))
        out[anchor] = image                               # a later image wins a clash
    return out


class Wan:
    def __init__(self, model: str = "14B", *, wan_dir=None, ckpt_dir=None):
        if model not in SETTINGS:
            raise ValueError(f"model must be one of {sorted(SETTINGS)}")
        self.model, self.settings = model, SETTINGS[model]
        self.wan_dir = find_wan(wan_dir)
        self.ckpt_dir = Path(ckpt_dir or os.environ.get("K2V_CKPT_DIR")
                             or self.wan_dir / self.settings["ckpt"])
        if not self.ckpt_dir.is_dir():
            raise FileNotFoundError(
                f"no {self.settings['ckpt']} checkpoint at {self.ckpt_dir}. Download it "
                f"as in Wan2.2's README, or pass --ckpt-dir.")
        self._pipe = None

    def generate(self, first_frame: Path, caption: str, keyframes: list[Keyframe],
                 *, seed: int, out_path: Path) -> Path:
        frames = to_frames(schedule(keyframes), self.model)
        if self.model == "14B":
            from . import conditioning_14b as conditioning
        else:
            from . import conditioning_5b as conditioning
        if self._pipe is None:
            self._pipe = conditioning.load(self.ckpt_dir)
        return conditioning.generate(
            self._pipe, first_frame, caption, frames, seed=seed, out_path=Path(out_path),
            negative_prompt=NEGATIVE_PROMPT, **{k: v for k, v in self.settings.items()
                                                if k not in ("task", "ckpt")})
