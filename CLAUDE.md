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
thinning, and creates diagnostic images for the axis and its intersections.

Cutting each junction's maximal-inscribed disc leaves the strokes with free
ends; `--cut-radius-scale` (`build_skeleton_graph(radius_scale=...)`) scales
that radius, so the same value drives the cut, the stroke-end lookup, and the
tangent reach. `graph.py::sample_axis_lines` resamples every cut stroke at even arc
length first; `graph.py::intersection_tangents` then fits each free end's
tangent to the pixels inside its first sample interval, crosses each
junction's tangents pairwise, and takes the centroid of those crossings as the
junction's focus. Thinning often splits a crossing into two adjacent branch
points; junctions whose discs overlap are fused and the fused junction sits at
the mean of their foci. `graph.py::merge_tangent_foci` folds each focus into the
sampled strokes as a shared node, joining every cut end back to it, and returns
the result as a point/segment `SampledGraph`.

`skeletonize.py::process_image` writes `<stem>-1-binarize.png`,
`<stem>-2-skeleton.png`, `<stem>-3-skeleton-cut-intersections.png`,
`<stem>-4-skeleton-samples.png`, `<stem>-5-tangent-crossings.png`, and
`<stem>-6-merged-graph.png`.

Binary images use ink = 0 (black) and background = 255 (white).

## Rules

- DO NOT add comments unnecessarily.
- DO NOT write tests
- Run `uv run ruff check .` and `uv run pyright` after modifying code.
- EXPLAIN SHORTLY.