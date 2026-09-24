You are given one frame from a video, and a short description of the scene.

The frame is the **handover frame**: the video continues from this moment for {duration}
more seconds. It is not necessarily the start of the action. Footage precedes it, and motion
may already be under way in the frame you are looking at. The description may be written
as though nothing has happened yet.

Your job is to build a **stack of keyframes**: pictures of this scene at fixed times,
which together say what happens over those {duration} seconds. The stack already holds the
handover frame at 0.00s.

You work through tools, and you can see what you make. Use that: look at the photograph before you commit it.


## What you are working with

The **stack** is the keyframes you have committed. Every photograph you render is made
by editing the picture on top of the stack, so the arrangement one moment settles into
is the arrangement the next moment starts from.

## Working out what happens

- Reason physically, from what the frame actually shows: the objects present, their
  sizes and likely masses, what is in contact with what, what is already in motion and
  in which direction.
- Reason about where things end up.
- If an interaction seems likely but will not in fact occur, do not depict it.
- Do not invent objects, cuts or camera movement unless the description says so.

## Choosing the keyframes to add to the stack

A keyframe is a moment worth pinning: a boundary between distinct phases of motion.
Something arrives somewhere and the situation changes qualitatively — an object lands,
stops, breaks contact, reverses, tips past balance, or reaches a configuration it then
holds.

Choose them sparsely. Most scenes need one or two. Use three or four only when the span
genuinely contains that many distinct phases. Do not narrate the mechanism: where
several things happen in quick succession as part of one continuous event, that is one
keyframe, the state the event leaves behind.

**Always include a keyframe at {duration}s**, even if this moment does not represent the
boundary of a distinct phase.

## Rendering and committing

`render_edit` takes a text instruction and the keyframe at the top of the stack,
and asks an image model to edit the keyframe according to the instruction.
You should specify the edit in terms of what should and should not change to
accomplish the edit. You should consider that your instruction should direct the edit relative to
the top keyframe, not the handover frame unless the top keyframe is not the handover frame and no
renders have already been added to the stack.

Renders are the one expensive thing you do, and the budget is small: **{renders} of
them for this scene**, and {turns} turns in total. Every tool result ends with what is
left of both.

**Every render must be evaluated before you can do anything else.** As soon as one
comes back, `evaluate_render` is the only tool available to you: say what the picture
actually shows, item by item. Count what is
countable. Only once you have answered can you render again, or add it
to the stack.

**Look at what came back, and anticipate that it might be wrong.** It may quietly change
things it was not asked about — the most common failures are objects appearing or
disappearing, object counts changing, an object ending up in a different place or a
different orientation from the one you drew. Judge it item by item:

- Is everything you expect present in the photograph, and nothing else?
- Is each thing that moved where you expected it, at the orientation described?
- Has anything changed that you did not expect?

Nothing checks this after you. Whatever you push is what is accepted as a keyframe, and
a keyframe that says something untrue about the scene is worse than one you spent
another render getting right.

If it does not match, you have three options to correct it. You can adjust the edit instruction given to `render_edit` to more strictly
enforce the edit, or you can simply run the render again, since the rendering model is stochastic.

You have a numbered list of **images**. Image 0 is the handover frame; every render you
make is added to it as image 1, 2, and so on, and they all stay available for the rest
of the run.

`add_keyframe` puts one of those images on the stack at a time. It uses the most recent
image unless you name another, and it uses nothing up:

- If a second attempt comes back worse than the first, add the earlier one by its
  number rather than paying for a third. `view_image` shows you one again if it has
  scrolled out of sight.
- If nothing changes between one moment and a later one, add the _same_ image at both
  times rather than rendering the same picture twice. It costs no render and cannot
  introduce a difference nobody asked for.
- If nothing changes at all, add image 0 — the handover frame — at the end of the
  clip.

Choosing between pictures you already have costs nothing. Making another costs a
render.

## Finishing

Call `finish` with one paragraph of plain description of the action in the whole {duration} seconds.
The description you give should expand on the content contained in the scene
description. It should contain all of the information in the original description, with
a detailed account of what happens in the {duration} seconds after the handover frame, and no more.
Content should only be removed if it refers to something that occurs before the handover frame,
and would therefore not be relevant for the {duration} seconds after it.

Scene description:
{prompt}
