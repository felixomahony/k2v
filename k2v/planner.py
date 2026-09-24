"""The keyframe planner: an agent that builds a stack of keyframes for a video.

The agent is given the input frame (or, for text-to-video, nothing) and the caption.
It works through five tools:

  render_edit      edit the picture on top of the stack into a new photograph
                   (with an empty stack under text-to-video, draw the scene fresh)
  evaluate_render  say what the render actually shows -- required after every render
  add_keyframe     put one of its images on the stack at a time in seconds
  view_image       look again at an earlier image
  finish           stop, and give the caption for the whole clip

Budgets are shown to the agent after every tool result: 24 turns and 8 renders. The
loop is deliberately plain -- ask, run the tool, hand back the result -- with two
rules: after a render the only tool offered is `evaluate_render`, and on the last turn
the only tool offered is `finish`, so every run ends by decision.

The result is a `Plan`: the expanded caption, the frame the video starts from, and the
keyframes in seconds. A keyframe that reuses the previous one's image is marked `held`
(the scene stays still between them), and the last keyframe is held to the end of the
clip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .colour import match_keyframe
from .models import ImageModel, LanguageModel

PROMPTS = Path(__file__).parent / "prompts"

#: The clock the agent plans against: 81 frames at 16 fps (5.0 s). Keyframes snap to
#: every 4th frame, the positions the video model's latents can address.
FPS, FRAME_NUM = 16, 81


@dataclass
class Keyframe:
    seconds: float
    image: Path
    note: str = ""
    held: bool = False          # same image as the keyframe before: nothing changes


@dataclass
class Plan:
    caption: str                # the expanded caption the video model is given
    first_frame: Path           # the input image, or the opening frame the agent drew
    keyframes: list[Keyframe]   # interior keyframes; the first frame is not repeated here
    transcript: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        data = asdict(self)
        data["first_frame"] = str(self.first_frame)
        data["keyframes"] = [dict(k, image=str(k["image"])) for k in data["keyframes"]]
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


class KeyframePlanner:
    def __init__(self, language: LanguageModel, image: ImageModel, *, out_dir: Path,
                 max_turns: int = 24, max_renders: int = 8, context_images: int = 3):
        self.language, self.image_model = language, image
        self.out_dir = Path(out_dir) / "keyframes"
        self.max_turns, self.max_renders = max_turns, max_renders
        self.context_images = context_images

    def plan(self, caption: str, image: Path | None = None) -> Plan:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        session = _Session(self, Path(image) if image else None)
        brief = (PROMPTS / ("i2v.md" if image else "t2v.md")).read_text().format(
            prompt=caption, turns=self.max_turns, renders=self.max_renders,
            duration=f"{(FRAME_NUM - 1) / FPS:.2f}")
        turns = [{"role": "user", "text": brief, "images": [session.first] if image else []}]
        specs = tool_specs(t2v=image is None)
        tools = {s["name"]: getattr(session, s["name"]) for s in specs}
        transcript = []

        while session.turns_left > 0 and session.finished is None:
            if session.turns_left == 1:
                offered, force = [s for s in specs if s["name"] == "finish"], ["finish"]
            elif session.pending_evaluation:
                offered = [s for s in specs if s["name"] == "evaluate_render"]
                force = ["evaluate_render"]
            else:
                offered, force = specs, None
            call = self.language.converse(turns, offered, force=force)
            session.turns_left -= 1
            if call.text:
                transcript.append(call.text)
            if not call.name:
                turns.append({"role": "model", "text": call.text, "raw": call.raw})
                turns.append({"role": "user",
                              "text": "Use a tool to act on that." + session.budget()})
                continue
            turns.append({"role": "model", "text": call.text, "raw": call.raw,
                          "calls": [{"name": call.name, "args": call.args}]})
            fn = tools.get(call.name)
            try:
                result, images = (fn(**call.args) if fn else (f"no such tool: {call.name}", []))
            except TypeError as exc:
                result, images = f"bad arguments: {exc}", []
            except Exception as exc:          # a broken tool must not end the run
                result, images = f"{call.name} failed: {type(exc).__name__}: {exc}", []
            transcript.append(f"[{call.name}] {json.dumps(call.args, ensure_ascii=False)}"
                              f" -> {result}")
            turns.append({"role": "tool", "name": call.name, "result": result,
                          "images": list(images)})
            _prune(turns, keep=self.context_images)

        return session.harvest(caption, transcript)


class _Session:
    """The stack, the numbered image list, the budgets, and the tools that act on them."""

    def __init__(self, planner: KeyframePlanner, first: Path | None):
        self.p = planner
        self.first = first
        self.t2v = first is None
        #: every photograph available as a keyframe; index 0 is the input image, if any
        self.images: list[Path] = [first] if first else []
        self.stack: list[tuple[int, Path, str]] = []     # (anchor frame, image, note)
        self.evaluations: dict[int, str] = {}
        self.turns_left, self.renders_left = planner.max_turns, planner.max_renders
        self.pending_evaluation = False
        self.finished: str | None = None
        self.renders = 0

    # -- numbering and budget ----------------------------------------------------
    def label(self, index: int) -> int:        # list position -> number shown
        return index + (1 if self.t2v else 0)

    def resolve(self, number: int) -> int:     # number given -> list position
        return number - (1 if self.t2v else 0)

    def budget(self) -> str:
        return (f"\n[turns left: {self.turns_left} of {self.p.max_turns} · "
                f"renders left: {self.renders_left} of {self.p.max_renders}]")

    def anchor_for(self, seconds: float) -> int:
        frame = max(0, min(FRAME_NUM - 1, int(round(float(seconds) * FPS))))
        return int(round(frame / 4)) * 4

    def describe_stack(self) -> str:
        if not self.stack:
            return "the stack holds only the handover frame at 0.0s"
        rows = ["the stack, in order:", "  0.00s  frame 0  the handover frame"]
        rows += [f"  {a / FPS:.2f}s  frame {a}  {note or '(no note)'}"
                 for a, _, note in self.stack]
        return "\n".join(rows)

    # -- tools -------------------------------------------------------------------
    def render_edit(self, instruction: str = "") -> tuple[str, list]:
        if self.renders_left <= 0:
            return "no renders left. Push what you have and finish." + self.budget(), []
        source = self.stack[-1][1] if self.stack else self.first
        self.renders += 1
        out = self.p.out_dir / f"render{self.renders}.jpg"
        try:
            raw = self.p.image_model.edit(instruction, [] if source is None else [source], out)
        except Exception as exc:
            self.renders_left -= 1
            return (f"the image model failed: {type(exc).__name__}. "
                    f"{self.renders_left} renders left."), []
        self.renders_left -= 1
        # Corrected once, here, so what the agent judges is what will condition the
        # video. The reference is image 0: the input frame, or under text-to-video the
        # opening frame -- which passes through uncorrected while it is being drawn.
        fixed = (Path(raw) if not self.images else
                 match_keyframe(Path(raw), self.images[0], out.with_name(out.stem + "_corrected.jpg")))
        self.pending_evaluation = True
        self.images.append(fixed)
        n, first = self.label(len(self.images) - 1), self.label(0)
        earlier = (f" Images {first} to {n - 1} are still available: add one of those "
                   f"instead by giving its number, or look at it again with "
                   f"view_image. Neither costs a render." if self.renders >= 2 else "")
        drawn = ("drawn from your instruction alone" if source is None
                 else "drawn from the picture on top of the stack")
        return (f"this is image {n}, {drawn}. "
                f"Compare it against what you asked for before pushing it: is everything "
                f"you asked for present, in the same places, at the same orientations, "
                f"touching what it touches?" + earlier + self.budget(), [fixed])

    def evaluate_render(self, evaluation: str = "") -> tuple[str, list]:
        text = (evaluation or "").strip()
        if not text:
            return "say what the render shows, compared with what you asked for" + self.budget(), []
        last = len(self.images) - 1
        self.evaluations[last] = text
        self.pending_evaluation = False
        return (f"recorded for image {self.label(last)}. If it does not depict the moment "
                f"you meant, adjust the instruction and render again; if it does, add "
                f"it to the stack." + self.budget(), [])

    def add_keyframe(self, seconds: float = 0.0, note: str = "",
                     image: int | None = None) -> tuple[str, list]:
        if image is None:
            index = len(self.images) - 1
        else:
            try:
                index = self.resolve(int(image))
            except (TypeError, ValueError):
                return "image must be a number" + self.budget(), []
        if not 0 <= index < len(self.images):
            low, high = self.label(0), self.label(len(self.images) - 1)
            whence = "" if self.t2v else ", where 0 is the handover frame"
            return (f"there is no image {self.label(index)}. Available: {low} to "
                    f"{high}{whence}." + self.budget(), [])
        anchor = self.anchor_for(seconds)
        if self.stack and anchor <= self.stack[-1][0]:
            return (f"{seconds:.2f}s snaps to frame {anchor}, which is not after the "
                    f"keyframe already on the stack at frame {self.stack[-1][0]}. "
                    f"Keyframes must move forward in time." + self.budget(), [])
        self.stack.append((anchor, self.images[index], note))
        return (f"added image {self.label(index)} at frame {anchor} = {anchor / FPS:.2f}s "
                f"(you asked for {seconds:.2f}s; times snap to every 4th frame).\n"
                + self.describe_stack() + self.budget(), [])

    def view_image(self, index: int = 0) -> tuple[str, list]:
        try:
            n = self.resolve(int(index))
        except (TypeError, ValueError):
            return "give the number of a render" + self.budget(), []
        if not 0 <= n < len(self.images):
            return (f"there is no image {index}. Available: {self.label(0)} to "
                    f"{self.label(len(self.images) - 1)}." + self.budget(), [])
        what = "the handover frame" if n == 0 and not self.t2v else f"image {self.label(n)}"
        return f"{what} again" + self.budget(), [self.images[n]]

    def finish(self, prompt: str = "") -> tuple[str, list]:
        self.finished = prompt.strip()
        return "finished", []

    # -- result ------------------------------------------------------------------
    def harvest(self, caption: str, transcript: list[str]) -> Plan:
        keyframes: list[Keyframe] = []
        for anchor, image, note in self.stack:
            held = bool(keyframes) and str(keyframes[-1].image) == str(image)
            keyframes.append(Keyframe(anchor / FPS, image, note, held))
        # The last frame is conditioned either way, so a stack that stops short is
        # read as "and then it stays like that": its final keyframe is held to the end.
        end = (FRAME_NUM - 1) - (FRAME_NUM - 1) % 4
        if keyframes and keyframes[-1].seconds < end / FPS:
            keyframes.append(Keyframe(end / FPS, keyframes[-1].image,
                                      keyframes[-1].note, held=True))
        first = self.first
        if self.t2v:
            # Text-to-video: the keyframe at 0 s is the opening frame the video starts
            # from, so it is not repeated as an interior keyframe.
            opening = next((k for k in keyframes if k.seconds == 0), None)
            if opening is None:
                raise RuntimeError("the agent committed no keyframe at 0.00s, so there "
                                   "is no opening frame to start the video from")
            first = opening.image
            keyframes = [k for k in keyframes if k is not opening]
        return Plan(caption=self.finished or caption, first_frame=first,
                    keyframes=keyframes, transcript=transcript)


def _prune(turns: list[dict], keep: int) -> None:
    """Keep only the most recent `keep` tool images in the conversation (never the
    first turn's). Every image is re-sent on every later turn, so without this the
    input cost grows quadratically with the number of renders."""
    seen = 0
    for turn in reversed(turns[1:]):
        if not turn.get("images"):
            continue
        if seen >= keep:
            turn["images"] = []
        else:
            seen += 1


def tool_specs(t2v: bool) -> list[dict]:
    edits = ("Draw a photograph from your instruction. While the stack is empty "
             "this draws the scene fresh; once a keyframe is on the stack it "
             "edits the picture on top of it. Costs one render."
             if t2v else
             "Edit the picture on top of the stack according to your instruction, "
             "and give the result back as a photograph. Costs one render.")
    numbering = ("Your renders are numbered from 1, in the order you made them."
                 if t2v else "Image 0 is the handover frame, and renders follow from 1.")
    holding = ("" if t2v else
               " Image 0 is the handover frame, so adding image 0 says nothing has "
               "changed since the start.")
    return [
        {"name": "render_edit", "description": edits,
         "parameters": {"type": "object", "properties": {
             "instruction": {"type": "string",
                             "description": "what to change, and what to keep"}},
             "required": ["instruction"]}},
        {"name": "evaluate_render",
         "description": (
             "Say what the render you just made actually shows, compared with the "
             "picture it was edited from and the instruction you gave. Required "
             "immediately after every render: nothing else can be done until it is "
             "answered. Go through the scene item by item -- is everything you "
             "expected present, in the same number, in the same places, leaning the "
             "same way, touching what it touches, and has anything appeared or "
             "changed that you did not ask for. Costs no render."),
         "parameters": {"type": "object", "properties": {
             "evaluation": {"type": "string",
                            "description": ("what the render shows, item by item, and "
                                            "where it differs from what you asked for")}},
             "required": ["evaluation"]}},
        {"name": "add_keyframe",
         "description": ("Add one of your images to the keyframe stack at a time in "
                         "seconds. Times snap to every 4th frame. Uses the most "
                         "recent image unless `image` names another. Nothing is used "
                         "up: the same image may be added at several times, which is "
                         "how you say the scene does not change between them." + holding),
         "parameters": {"type": "object", "properties": {
             "seconds": {"type": "number"},
             "note": {"type": "string", "description": "what this moment shows"},
             "image": {"type": "integer",
                       "description": f"which image to add. {numbering} Omit for the "
                                      f"most recent."}},
             "required": ["seconds", "note"]}},
        {"name": "view_image",
         "description": f"Look again at one of your images. {numbering} Costs no render.",
         "parameters": {"type": "object", "properties": {
             "index": {"type": "integer", "description": numbering}},
             "required": ["index"]}},
        {"name": "finish",
         "description": ("Stop, and give the prompt describing the whole clip that "
                         "will be sent to the video model."),
         "parameters": {"type": "object", "properties": {
             "prompt": {"type": "string"}}, "required": ["prompt"]}},
    ]
