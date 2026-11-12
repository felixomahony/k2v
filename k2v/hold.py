"""Hold the video still once the plan says the scene has come to rest.

Even with a held span conditioned every second, the video model keeps introducing
small spurious motion over an interval the plan describes as static. So the last step
of K2V enforces the plan's own claim: from the rest time to the end of the clip, every
frame repeats the frame at the rest time.

The rest time is read from the plan only:

  declared  the plan ends in held keyframes (the same image reused); the scene is at
            rest from the keyframe before them
  inferred  the last two keyframes are near-identical renders (mean absolute difference
            below MAE grey levels at 480x270) -- the agent drew the settled state twice
            instead of reusing it; the scene is at rest from the earlier of the two

A plan with neither leaves the video untouched. The inferred rule can mistake periodic
motion for rest (a pendulum drawn twice at the same point of its swing).
"""

from __future__ import annotations

import cv2
import numpy as np

from .planner import FPS as PLAN_FPS, Keyframe

#: Grey levels below which two keyframe renders count as the same state.
MAE = 4.0


def rest_seconds(keyframes: list[Keyframe], mae: float = MAE) -> tuple[float | None, str | None]:
    """When the plan says the scene stops moving, and whether it said so or we inferred it."""
    previous = 0.0
    for i, k in enumerate(keyframes):
        if k.held and all(x.held for x in keyframes[i:]):
            return previous, "declared"
        previous = k.seconds
    if len(keyframes) >= 2:
        a, b = [cv2.imread(str(k.image)) for k in keyframes[-2:]]
        if a is not None and b is not None:
            a, b = [cv2.resize(x, (480, 270)).astype(np.float32) for x in (a, b)]
            if np.abs(a - b).mean() < mae:
                return keyframes[-2].seconds, "inferred"
    return None, None


def rest_frame(seconds: float, fps: int) -> int:
    """The video frame for a rest time, on the same grid the keyframes were placed on
    (multiples of 4, re-timed by the second for a 24 fps backbone; see wan.to_frames)."""
    frame = round(seconds * PLAN_FPS)
    return int(round(fps * (frame / PLAN_FPS) / 4)) * 4 if frame else 0

