"""K2V: Keyframes to Video.

The whole method is `run` below: a planning agent turns the caption (and the image, if
there is one) into a stack of keyframes and an expanded caption, and Wan2.2 generates
the video conditioned on them. Everything it calls lives in its own module:

  planner.py          the keyframe agent             gemini.py / models.py  its models
  wan.py              Wan2.2 and the keyframe schedule   conditioning_*.py  per backbone
  colour.py           colour correction of keyframes and of the finished video
  hold.py             holding the video still once the plan says the scene is at rest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .colour import regrade_video
from .gemini import GeminiImage, GeminiLanguage
from .hold import rest_frame, rest_seconds
from .planner import KeyframePlanner, Plan
from .wan import Wan


def run(caption: str | None, image: Path | None = None, *, model: str = "14B", seed: int = 42,
        out: Path = Path("outputs/k2v"), wan_dir=None, ckpt_dir=None,
        colour_correction: bool = True, hold: bool = True,
        keyframes: Path | None = None) -> Path:
    out = Path(out)
    if keyframes is not None:      # a saved or hand-written plan.json: skip planning
        plan = Plan.load(keyframes)
        plan.caption = caption or plan.caption
        plan.first_frame = Path(image).resolve() if image else plan.first_frame
    else:
        planner = KeyframePlanner(GeminiLanguage(), GeminiImage(), out_dir=out)
        plan = planner.plan(caption, image)                # keyframe stack + expanded caption
    plan.rest_seconds, _ = rest_seconds(plan.keyframes)     # when the plan says it stops
    out.mkdir(parents=True, exist_ok=True)
    plan.save(out / "plan.json")

    wan = Wan(model, wan_dir=wan_dir, ckpt_dir=ckpt_dir)
    video = wan.generate(plan.first_frame, plan.caption, plan.keyframes,
                         seed=seed, out_path=out / "video_raw.mp4")

    fps = wan.settings["fps"]
    rest = rest_frame(plan.rest_seconds, fps) if hold and plan.rest_seconds is not None else None
    if colour_correction or rest is not None:      # re-grade onto frame 0; still from the rest
        video = regrade_video(video, out / "video.mp4", fps=fps,
                              regrade=colour_correction, hold_from=rest)
    return video


def cli() -> None:
    p = argparse.ArgumentParser(prog="k2v", description=__doc__.splitlines()[0])
    p.add_argument("--caption", help="what happens in the video "
                                     "(with --keyframes, replaces the plan's caption)")
    p.add_argument("--image", type=Path, help="the first frame; omit for text-to-video")
    p.add_argument("--keyframes", type=Path, metavar="PLAN_JSON",
                   help="skip planning and use this plan.json (saved or hand-written)")
    p.add_argument("--model", choices=["14B", "5B"], default="14B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs/k2v"))
    p.add_argument("--wan-dir", help="an existing Wan2.2 checkout (default: the submodule)")
    p.add_argument("--ckpt-dir", help="Wan2.2 checkpoint directory")
    p.add_argument("--no-colour-correction", action="store_true")
    p.add_argument("--no-hold", action="store_true",
                   help="do not hold the video still from the plan's rest time")
    a = p.parse_args()
    if a.caption is None and a.keyframes is None:
        p.error("--caption is required unless --keyframes is given")
    video = run(a.caption, a.image, model=a.model, seed=a.seed, out=a.out,
                wan_dir=a.wan_dir, ckpt_dir=a.ckpt_dir,
                colour_correction=not a.no_colour_correction, hold=not a.no_hold,
                keyframes=a.keyframes)
    print(f"wrote {video}")


if __name__ == "__main__":
    cli()
