"""Colour correction, before and after generation.

Image editors drift in colour: a rendered keyframe can come back slightly re-graded
relative to the frame it was edited from, and a video conditioned on it inherits the
shift. K2V corrects twice, with the same robust fit:

  match_keyframe  each rendered keyframe onto the reference frame (the input image, or
                  under text-to-video the opening frame the agent drew), when the drift
                  in mean grey level is large enough to matter.
  regrade_video   each frame of the finished video back onto its frame 0 (and, from
                  the plan's rest time, holds the frame -- see hold.py).

The fit is a per-channel gain and offset in LAB with Huber weights. Pixels where the
scene genuinely changed fail to fit and are down-weighted, so the correction is a
grade, not a repaint. The gain is clamped to [1/1.2, 1.2] so it cannot flatten contrast
to explain real content change.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GAIN_LIMIT = 1.2
#: Grey-level drift below which a keyframe is left untouched: correcting a
#: well-graded render only perturbs it.
KEYFRAME_THRESHOLD = 6.0


def fit_gain_offset(x: np.ndarray, y: np.ndarray, iterations: int,
                    gain_limit: float = GAIN_LIMIT) -> tuple[float, float]:
    """Huber-weighted gain g and offset o with y ~ g*x + o, gain clamped."""
    g, o = 1.0, float(np.median(y) - np.median(x))
    for _ in range(iterations):
        res = y - (g * x + o)
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        w = np.clip(1.345 * scale / np.maximum(np.abs(res), 1e-6), 0, 1)
        sw = w.sum()
        mx, my = (w * x).sum() / sw, (w * y).sum() / sw
        vxx = (w * (x - mx) ** 2).sum()
        g = ((w * (x - mx) * (y - my)).sum() / vxx) if vxx > 1e-6 else 1.0
        g = float(np.clip(g, 1 / gain_limit, gain_limit))
        o = my - g * mx
    return g, o


def _grade(image: np.ndarray, reference: np.ndarray, iterations: int) -> np.ndarray:
    """`image` (BGR) re-graded onto `reference`, fitted on every 7th pixel."""
    src = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src.copy()
    for c in range(3):
        g, o = fit_gain_offset(src[:, :, c].ravel()[::7], ref[:, :, c].ravel()[::7],
                               iterations)
        out[:, :, c] = g * src[:, :, c] + o
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _grey(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32).mean())


def match_keyframe(render: Path, reference: Path, out_path: Path) -> Path:
    """Correct a rendered keyframe onto the reference frame; always writes `out_path`."""
    edit = cv2.imread(str(render))
    ref = cv2.imread(str(reference))
    if edit is None or ref is None:
        raise FileNotFoundError(f"cannot read {render} or {reference}")
    ref = cv2.resize(ref, (edit.shape[1], edit.shape[0]))
    if abs(_grey(edit) - _grey(ref)) >= KEYFRAME_THRESHOLD:
        edit = _grade(edit, ref, iterations=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), edit, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return out_path


def regrade_video(src: Path, out_path: Path, fps: int, *, regrade: bool = True,
                  hold_from: int | None = None) -> Path:
    """Re-grade every frame of `src` onto its frame 0 and write H.264 to `out_path`.

    `hold_from`, if given, repeats that (re-graded) frame to the end of the clip -- the
    rest step in hold.py. `regrade=False` writes the frames ungraded, for a run with
    colour correction switched off that still holds."""
    import imageio

    cap = cv2.VideoCapture(str(src))
    frames = []
    while (ok_frame := cap.read())[0]:
        frames.append(ok_frame[1])
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {src}")
    n = len(frames)
    if hold_from is not None and hold_from < n - 1:          # frames after it are replaced
        frames = frames[:hold_from + 1]
    out = [_grade(f, frames[0], iterations=8) if regrade else f for f in frames]
    out += [out[-1]] * (n - len(out))                        # ... by the held frame
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for frame in out:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()
    return out_path
