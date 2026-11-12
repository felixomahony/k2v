"""CPU-only tests: no GPU, no API key, no Wan checkpoint needed."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from k2v.colour import KEYFRAME_THRESHOLD, fit_gain_offset, match_keyframe, regrade_video
from k2v.hold import rest_frame, rest_seconds
from k2v.models import ToolCall
from k2v.planner import Keyframe, KeyframePlanner, Plan
from k2v.wan import schedule, to_frames


# -- keyframe schedule ---------------------------------------------------------------

def test_held_span_is_filled_every_second():
    a, b = Path("a.jpg"), Path("b.jpg")
    keys = [Keyframe(1.25, a), Keyframe(5.0, a, held=True)]
    assert schedule(keys) == [(1.25, a), (2.25, a), (3.25, a), (4.25, a), (5.0, a)]
    assert schedule([Keyframe(2.0, a), Keyframe(5.0, b)]) == [(2.0, a), (5.0, b)]


def test_frames_per_backbone():
    a = Path("a.jpg")
    times = [(1.25, a), (5.0, a)]
    assert to_frames(times, "14B") == {20: a, 80: a}          # 16 fps, exact
    assert to_frames(times, "5B") == {32: a, 120: a}          # 24 fps, re-timed by second


# -- colour correction ---------------------------------------------------------------

def test_fit_recovers_gain_and_offset():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 200, 5000)
    g, o = fit_gain_offset(x, 1.1 * x + 7, iterations=12)
    assert g == pytest.approx(1.1, abs=1e-3) and o == pytest.approx(7, abs=0.1)


def test_keyframe_matched_only_past_threshold(tmp_path):
    ref = np.random.default_rng(1).integers(40, 200, (72, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    for shift, should_change in ((2, False), (int(KEYFRAME_THRESHOLD) + 20, True)):
        render = np.clip(ref.astype(int) + shift, 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / "r.png"), render)
        out = cv2.imread(str(match_keyframe(tmp_path / "r.png", tmp_path / "ref.png",
                                            tmp_path / "o.png")))
        drift = abs(out.astype(float).mean() - ref.mean())
        assert (drift < shift / 4) == should_change


# -- hold at rest --------------------------------------------------------------------

def test_rest_declared_by_held_keyframes(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    keys = [Keyframe(0.5, b), Keyframe(1.25, a), Keyframe(5.0, a, held=True)]
    assert rest_seconds(keys) == (1.25, "declared")
    # a held span in the middle that is followed by more motion is not a rest
    assert rest_seconds([Keyframe(1.0, a), Keyframe(2.0, a, held=True), Keyframe(5.0, b)])[1] != "declared"


def test_rest_inferred_from_near_identical_renders(tmp_path):
    rng = np.random.default_rng(2)
    img = rng.integers(30, 220, (270, 480, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    cv2.imwrite(str(tmp_path / "b.png"), np.clip(img.astype(int) + 2, 0, 255).astype(np.uint8))
    cv2.imwrite(str(tmp_path / "c.png"), 255 - img)
    a, b, c = (tmp_path / f"{x}.png" for x in "abc")
    assert rest_seconds([Keyframe(1.5, a), Keyframe(5.0, b)]) == (1.5, "inferred")
    assert rest_seconds([Keyframe(1.5, a), Keyframe(5.0, c)]) == (None, None)


def test_rest_frame_per_backbone():
    assert rest_frame(1.25, 16) == 20 and rest_frame(1.25, 24) == 32     # as wan.to_frames
    assert rest_frame(0.0, 24) == 0


def test_video_held_from_rest_frame(tmp_path):
    import imageio
    src, out = tmp_path / "raw.mp4", tmp_path / "held.mp4"
    w = imageio.get_writer(str(src), fps=16, codec="libx264", quality=8)
    for t in range(12):                       # a bright square sliding to the right
        f = np.full((64, 128, 3), 40, np.uint8); f[20:40, 8 * t:8 * t + 16] = 230
        w.append_data(f)
    w.close()
    regrade_video(src, out, fps=16, regrade=False, hold_from=5)
    cap, got = cv2.VideoCapture(str(out)), []
    while (r := cap.read())[0]:
        got.append(r[1].astype(float))
    assert len(got) == 12
    assert max(np.abs(f - got[5]).mean() for f in got[6:]) < 1.0         # still from frame 5
    assert np.abs(got[4] - got[5]).mean() > 3.0                          # moving before it


# -- planner loop --------------------------------------------------------------------

class ScriptedLanguage:
    def __init__(self, script):
        self.script, self.offered = list(script), []

    def converse(self, turns, tools, *, force=None):
        self.offered.append(([t["name"] for t in tools], force))
        name, args = self.script.pop(0)
        return ToolCall(name=name, args=args)


class CopyImage:
    def edit(self, instruction, images, out_path):
        img = (cv2.imread(str(images[0])) if images
               else np.full((72, 128, 3), 128, np.uint8))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return out_path


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "first.png"
    cv2.imwrite(str(path), np.full((72, 128, 3), 100, np.uint8))
    return path


def test_evaluation_forced_and_plan_harvested(tmp_path, frame):
    lm = ScriptedLanguage([
        ("render_edit", {"instruction": "x"}),
        ("evaluate_render", {"evaluation": "looks right"}),
        ("add_keyframe", {"seconds": 1.3, "note": "event"}),      # snaps to 1.25 s
        ("add_keyframe", {"seconds": 2.0, "note": "same", "image": 1}),
        ("finish", {"prompt": "EXPANDED"}),
    ])
    plan = KeyframePlanner(lm, CopyImage(), out_dir=tmp_path).plan("caption", frame)
    assert lm.offered[1] == (["evaluate_render"], ["evaluate_render"])
    assert plan.caption == "EXPANDED" and plan.first_frame == frame
    assert [(k.seconds, k.held) for k in plan.keyframes] == [(1.25, False), (2.0, True),
                                                            (5.0, True)]


def test_text_to_video_opening_frame(tmp_path):
    lm = ScriptedLanguage([
        ("render_edit", {"instruction": "draw the scene"}),
        ("evaluate_render", {"evaluation": "ok"}),
        ("add_keyframe", {"seconds": 0.0, "note": "opening"}),
        ("finish", {"prompt": "EXPANDED"}),
    ])
    plan = KeyframePlanner(lm, CopyImage(), out_dir=tmp_path).plan("caption")
    assert plan.first_frame.name == "render1.jpg"
    assert [(k.seconds, k.held) for k in plan.keyframes] == [(5.0, True)]


def test_plan_paths_are_absolute(tmp_path, frame, monkeypatch):
    monkeypatch.chdir(tmp_path)                    # relative image and out dir
    lm = ScriptedLanguage([
        ("render_edit", {"instruction": "x"}),
        ("evaluate_render", {"evaluation": "ok"}),
        ("add_keyframe", {"seconds": 2.0, "note": "n"}),
        ("finish", {"prompt": "P"}),
    ])
    plan = KeyframePlanner(lm, CopyImage(), out_dir="out").plan("c", frame.name)
    assert plan.first_frame.is_absolute()
    assert all(k.image.is_absolute() for k in plan.keyframes)


def test_last_turn_forces_finish(tmp_path, frame):
    lm = ScriptedLanguage([("view_image", {"index": 0})] * 3 + [("finish", {"prompt": "P"})])
    KeyframePlanner(lm, CopyImage(), out_dir=tmp_path, max_turns=4).plan("c", frame)
    assert lm.offered[-1] == (["finish"], ["finish"])


# -- loading a plan instead of planning --------------------------------------------

def test_saved_plan_round_trips(tmp_path, frame):
    lm = ScriptedLanguage([
        ("render_edit", {"instruction": "x"}),
        ("evaluate_render", {"evaluation": "ok"}),
        ("add_keyframe", {"seconds": 2.0, "note": "n"}),
        ("finish", {"prompt": "P"}),
    ])
    plan = KeyframePlanner(lm, CopyImage(), out_dir=tmp_path).plan("c", frame)
    plan.save(tmp_path / "plan.json")
    loaded = Plan.load(tmp_path / "plan.json")
    assert (loaded.caption, loaded.first_frame) == (plan.caption, plan.first_frame)
    assert loaded.keyframes == plan.keyframes


def test_hand_written_plan(tmp_path, frame):
    (tmp_path / "kf.jpg").write_bytes(frame.read_bytes())
    (tmp_path / "plan.json").write_text(json.dumps({
        "caption": "C", "first_frame": frame.name,           # relative to the plan file
        "keyframes": [{"seconds": 2.0, "image": "kf.jpg"},
                      {"seconds": 5.0, "image": "kf.jpg"}]}))
    plan = Plan.load(tmp_path / "plan.json")
    assert plan.first_frame == frame.resolve()
    assert [(k.seconds, k.held) for k in plan.keyframes] == [(2.0, False), (5.0, True)]
    (tmp_path / "bad.json").write_text(json.dumps({
        "caption": "C", "first_frame": frame.name,
        "keyframes": [{"seconds": 0.0, "image": "kf.jpg"}]}))
    with pytest.raises(ValueError):
        Plan.load(tmp_path / "bad.json")
