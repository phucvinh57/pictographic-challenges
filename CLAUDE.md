# CLAUDE.md

## Commands

```bash
uv sync
uv run skeletonize
uv run skeletonize --input <input> --output <output>
uv run python -m skeletonize
uv run ruff check .
uv run pyright
```

## Architecture

The pipeline binarizes an image with Otsu thresholding, splits the ink into the
strokes that overlap at its junctions, takes the medial axis of each stroke on
its own, and writes those back out as smooth stroked SVG curves.

`medial_axis.py::medial_axis` thins an ink mask down to a one-pixel,
topology-preserving axis with OpenCV's Zhang-Suen thinning
(`cv2.ximgproc.thinning`), and returns it with the ink's distance map, so every
axis pixel carries the radius of the disc the ink fits around it there.
`graph.py::build_skeleton_graph` traces that axis into edges and collects the
clusters of branch pixels strokes meet at. Thinning usually splits a crossing
into two branch pixels a step apart, so clusters whose discs overlap are fused:
no stroke of any length survives between them.

That whole-drawing axis is only used to find the junctions. The axis it draws
through one is nothing anyone drew — at a crossing it hangs the meeting off as a
branch of its own, and every branch into it bends towards the middle for about a
stroke width before it gets there. `strokes.py::trace_strokes` uses it to work
out which strokes overlap where, then goes back to the ink for their shape.

`strokes.py::_describe_junctions` sorts every branch leaving a junction. A
branch is a stroke's *arm* when a length of it survives outside the meeting at
the stroke's own width; `_trim_index` says where that starts — out of the disc
the ink fits around the junction and back down to the ridge a stroke sits at —
and gives up after a junction's own reach, or a stroke drawn heavier than the
drawing's median width would be trimmed away whole. What is left over is a
*spur*: the junction's own shape rather than a stroke.

A spur is one of two things and `strokes.py::_resolve_spur` tells them apart by
where the arms' own lines cross. A stroke turning a corner runs straight to the
point it turns at and away along the other side, so the crossing lands exactly
where the spur settles back to the stroke's width — at the point of a round join,
at the far end of a miter. Two strokes leaving one shared cap — the middle of a
`3` — overlap all the way out to it and never turn into one another, so their
lines cross back inside the swelling instead, an axis width or more away. The
first is a corner the two arms turn at; the second is a *fold*, and each of its
arms runs out to the shared cap on its own.

`strokes.py::_pair_arms` then pairs the arms one stroke carries on through,
the corner pair first and the rest straightest-first while they turn by less
than `max_turn` degrees, which leaves the bar of a T unpaired to start a stroke
of its own. `strokes.py::_chains` follows the pairings into the run of branches
each stroke is made of, and `strokes.py::_guide` lays one out end to end: each
branch trimmed of the junctions at its ends, a `_bridge` across every junction
it passes (a Hermite arc, or a straight turn through the point of a corner), and
a `_terminal` where it stops at one. A stroke stops where its own disc stops
fitting, the same place a free end does, so the bar of a T reaches the middle of
the stem and the shaft of an arrow reaches the point of its head.

`strokes.py::_refine` is where the overlap is actually resolved.
`_stroke_region` takes the ink within `half_width + BAND_SLACK` of where the
stroke was laid out: outside a junction that band is wider than the ink and the
stroke gets all of it, and inside one it is the stroke's own share of ink both
strokes cover. That region is a plain ribbon with none of the branching that put
the junction there, so its medial axis is a single clean line, and the stroke is
walked onto it point by point rather than traced along it — a stroke may come
back on itself, as the tail of a `6` does where it runs into its own loop, and
only the order it was laid out in says which way round the ribbon it goes. The
walk is bounded to a half-width so a point between two strokes running side by
side, as the back and the seat of a chair do, cannot step onto its neighbour's
axis. The region is cut out with a background border before thinning: the
distance map only sees the array it is given, and a stroke flush against the edge
would read as running on past it.

Corners a junction never spoke for — a join drawn no wider than the pen leaves
nothing hanging off the axis to mark it — are found by `strokes.py::_path_corners`
from the turn itself. A corner turns all at once, over about a half-width, while
a curve keeps turning, so measuring the same bend over three half-widths tells
them apart. `strokes.py::_sharpen` then replaces the stretch each corner rounded
off, `_reach` wide either way, with the point the tangents on both sides cross
at, and marks it as a break. Only the one pass the stroke makes through a corner
is replaced: a stroke that comes back past it later, as the bottom of a
rectangle returns to the corner it started at, keeps everything in between.

`curves.py::smooth_chain` resamples each run between breaks at even arc length
and refits it as a centripetal Catmull-Rom spline returned as its cubic Bezier
spans, so the strokes keep their path but lose the faceting of the walk while
the corners stay the points they were solved to be. A stroke's runs go out as
one path, so a corner is a join of the pen's own outline rather than two ends
laid next to each other.

`svg.py::bezier_svg` writes those chains as stroked SVG paths, one polybezier
each. The curves are the axis, so the drawing is recovered by stroking them at
the ink's width rather than by outlining them: round caps and joins put back the
disc the axis carried. `medial_axis.py::stroke_half_width` measures that width
as the median axis distance — the median ignores the ends, where the disc
shrinks into the cap, and the junctions, where it swells to span two strokes at
once — less the half pixel of it that lies outside the ink, and `--stroke-width`
(`process_image(stroke_width=...)`) overrides it. An SVG has no background of its
own, so the paths are laid over an opaque white rectangle.

`challenge_1` provides the standalone challenge 1 raster pipeline, simplified
contour debug output, and vector output.
`skeletonize.py::process_image` writes one file, `<stem>.svg`.

Binary images use ink = 0 (black) and background = 255 (white).

## Rules

- DO NOT add comments unnecessarily.
- DO NOT write tests
- Run `uv run ruff check .` and `uv run pyright` after modifying code.
- Explain shortly, human-readable, informal words.
