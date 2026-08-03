# CLAUDE.md

## Commands

```bash
uv sync
uv run skeletonize
uv run skeletonize <input> <output>
uv run python -m skeletonize
uv run ruff check .
uv run pyright
```

## Architecture

The pipeline binarizes an image with Otsu thresholding, extracts the medial
axis of the original ink mask with distance-ordered, topology-preserving
thinning, and writes the axis back out as smooth stroked SVG curves.

`medial_axis.py::axis_disc_contacts` recovers the maximal-inscribed disc of
every axis pixel and the boundary patches it touches: two on a regular point,
one on an end point, three or more on a junction. Each contact is placed by
its position along a `cv2.findContours` contour and flagged as a corner when
the boundary turns sharply around it. A thinned pixel sits up to a pixel off
the true axis, which spreads the distances to the patches it should touch by a
share of its radius, so the disc is widened by `tolerance` pixels and by `slack`
of the radius before the patches are looked up — without the relative part a
wide junction loses the corner contact its neighbours all see, and the run
breaks in two right where it matters most. `medial_axis.py::corner_runs` walks
each traced axis line and collects every run where a corner drives the disc,
the fan that corner rounds the axis through; `medial_axis.py::transition_points`
takes the ends of those runs, the borders between the smooth pieces of the axis,
skipping any whose disc already touches three patches — the taxonomy gives each
point one class and that one is a junction point.

Both kinds of bend are then cut away as an `AxisCut`.
`graph.py::junction_cuts` gives each junction the axis it owns: it walks every
branch out of the junction and stops on the branch's first transition point, so
the whole bend is cut rather than a fixed circle of it. A branch with no
transition point within `max_reach` radii falls back to the maximal-inscribed
disc.
`graph.py::corner_cuts` then takes each remaining fan bounded by two transition
points; fans a junction already claimed are skipped.
`graph.py::remove_axis_cuts` blanks the claimed pixels, leaving the strokes with
free ends. `graph.py::sample_axis_lines` resamples every cut stroke at even arc
length first; `graph.py::intersection_tangents` then matches each free end to
the cut it borders, fits its tangent to the pixels inside its first sample
interval, crosses each cut's tangents pairwise, and takes the centroid of those
crossings as the cut's focus — the point of a corner, the meeting point of a
junction. Thinning often splits a crossing into two adjacent branch points;
junctions whose discs overlap are fused and the fused junction sits at the mean
of their foci, while corners take no part in that. `graph.py::merge_tangent_foci` folds each focus into the
sampled strokes as a shared node, joining every cut end back to it, and returns
the result as a point/segment `SampledGraph`.

`graph.py::smooth_sampled_graph` then replaces those straight segments with
curves. It splits the graph into the chains a single curve may span — breaking
at every stroke end, every focus, and every point more than two segments meet
at — and refits each chain as a centripetal Catmull-Rom spline returned as its
cubic Bezier spans. The spline interpolates the samples it was built from, so
the strokes keep their path but lose the faceting of the arc-length walk, while
breaking at the foci keeps a corner the sharp point the cut solved it to be.

`svg.py::bezier_svg` writes those chains as stroked SVG paths, one polybezier
each. The curves are the axis, so the drawing is recovered by stroking them at
the ink's width rather than by outlining them: round caps and joins put back the
disc the axis carried. `medial_axis.py::axis_stroke_width` measures that width
as twice the median maximal-inscribed radius — the median ignores the ends,
where the disc shrinks into the cap, and the junctions, where it swells to span
two strokes at once — and `--stroke-width` (`process_image(stroke_width=...)`)
overrides it. An SVG has no background of its own, so the paths are laid over an
opaque white rectangle.

`skeletonize.py::process_image` writes one file, `<stem>.svg`.

Binary images use ink = 0 (black) and background = 255 (white).

## Rules

- DO NOT add comments unnecessarily.
- DO NOT write tests
- Run `uv run ruff check .` and `uv run pyright` after modifying code.
- EXPLAIN SHORTLY.