You are given a short description of a video. There is no picture: the description is
everything you have, and the whole video is yours to invent.

Your job is to build a **stack of keyframes**: pictures of this scene at fixed times,
which together say what happens over {duration} seconds. The stack is empty. The first
keyframe you commit is the opening frame of the video, and the last one is the moment it
ends.

You work through tools, and you can see what you make. Use that: look at the photograph before you commit it.

## What you are working with

The **stack** is the keyframes you have committed, in time order. While it is empty, a
render draws the scene from nothing but your instruction. Once it holds something, every
render is made by editing the picture on top of it, so the arrangement one moment
settles into is the arrangement the next moment starts from.

## The opening frame

Make it first, before you think about anything later. It is the picture the video
begins from, so it should show the scene _before_ the action in the description has
happened — everything in place, nothing yet under way. A scene that opens halfway
through has nowhere left to go.

Decide the framing here and keep it. The camera does not move unless the description
says it does, so the viewpoint, distance and framing you choose for the opening frame
are the ones every later keyframe inherits — and since every later render is an edit of
this picture, they are inherited whether you restate them or not.

## Working out what happens

- Take the description literally, including any conditions it states — where the scene
  is, what the objects are made of, what temperature it is at. Those conditions are
  part of the scene and not decoration. Reason about what follows **under the stated
  conditions**, not about what would happen in the most familiar version of the scene.
- The keyframes are what the video will show, so an outcome you leave vague is an outcome that will
  be invented for you, and usually invented as the ordinary case.
- Reason from the objects: their sizes and likely masses, what is in contact with what,
  what starts in motion and in which direction.
- Reason about where things end up.
- If an interaction seems likely but will not in fact occur, do not depict it.
- Do not invent objects, cuts or camera movement the description does not call for.

## Choosing the keyframes to add to the stack

Past the opening frame, a keyframe is a moment worth pinning: a boundary between
distinct phases of motion. Something arrives somewhere and the situation changes
qualitatively — an object lands, stops, breaks contact, reverses, tips past balance,
changes state, or reaches a configuration it then holds.

Choose them sparsely. Most scenes need the opening frame and one or two more. Use more
only when the span genuinely contains that many distinct phases. Do not narrate the
mechanism: where several things happen in quick succession as part of one continuous
event, that is one keyframe, the state the event leaves behind.

**Always begin with a keyframe at 0.00s and finish with one at {duration}s**, whether or
not anything is still moving by then.

## Rendering and committing

`render_edit` takes a text instruction and asks an image model for a photograph. While
the stack is empty it draws the scene fresh, so the instruction should describe the
whole picture you want: the subject, the setting, and the framing. Once the stack holds
a keyframe, the same tool instead edits the picture on top of it, and the instruction
should say what changes and what must stay exactly as it is.

Renders are the one expensive thing you do, and the budget is small: **{renders} of
them for this scene**, and {turns} turns in total. The opening frame costs one of them.
Every tool result ends with what is left of both. Work out the arrangement before
spending a render — thinking it through and looking at earlier images cost a turn and
nothing else.

**Every render must be evaluated before you can do anything else.** As soon as one
comes back, `evaluate_render` is the only tool available to you: say what the picture
actually shows, item by item, against what you asked for. Count what is countable. Only
once you have answered can you render again, or add it to the stack.

**Look at what came back, and anticipate that it might be wrong.** It may quietly
change things it was not asked about — the most common failures are objects appearing or disappearing, object counts
changing, an object ending up in a different place or a different orientation from the
one you intended, and a stated colour or state reverting to the ordinary one. Judge it
item by item:

- Is everything you asked for present in the photograph, and nothing else?
- Is each thing that moved where you intended, at the orientation you intended?
- Has anything changed that you did not ask to change?
- Where the description states a condition or an outcome, does the photograph show
  that one, or has it drifted back to the everyday version?

Nothing checks this after you. Whatever you push is what is accepted as a keyframe, and
a keyframe that says something untrue about the scene is worse than one you spent
another render getting right.

If it does not match, you have two options to correct it. You can adjust the edit
instruction to enforce the edit more strictly, or you can simply run the render again,
since the rendering model is stochastic.

Your renders are collected in a numbered list of **images**, in the order you made
them, starting at 1. They all stay available for the rest of the run.

`add_keyframe` puts one of those images on the stack at a time. It uses the most recent
image unless you name another, and it uses nothing up:

- If a second attempt comes back worse than the first, add the earlier one by its
  number rather than paying for a third. `view_image` shows you one again if it has
  scrolled out of sight.
- If nothing changes between one moment and a later one, add the _same_ image at both
  times rather than rendering the same picture twice. It costs no render and cannot
  introduce a difference nobody asked for.

Choosing between pictures you already have costs nothing. Making another costs a
render.

## Finishing

Call `finish` with one paragraph of plain description of the action across the whole
{duration} seconds. It should contain all of the information in the original
description, together with a detailed account of what happens, and no more. Where you
committed to a particular outcome, say what it is rather than leaving it open.

Scene description:
{prompt}
