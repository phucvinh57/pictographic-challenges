# Skeletonize pipeline: PNG → SVG

This document walks through how a single input image becomes an SVG,
stage by stage, referencing the actual functions involved. For install/run
commands see the top-level [README.md](../../README.md); for a short summary
see [CLAUDE.md](../../CLAUDE.md).

## Overview

```
PNG (grayscale)
  │
  ▼  skeletonize.py:binarize
Otsu binary image (ink = 0/black, background = 255/white)
  │
  ├─▶ skeletonize.py:detect_edges ──▶ Canny edges  (debug overlays only)
  │
  ▼  centerline.py:find_centerline_points
Centerline midpoint cloud + estimated stroke_width
  │
  ▼  vectorize.py:vectorize_midpoints
SVG string + list of vector paths
  │
  ▼  skeletonize.py:process_image
Files written to the output directory
```

Two independent techniques are used, one per half of the pipeline:

- **Centerline estimation** (`centerline.py`) never thins pixels. It matches
  pairs of points across the ink's own boundary contours to estimate where
  the middle of each stroke is.
- **Vectorization** (`vectorize.py`) is the only place pixels get thinned —
  it rasterizes the midpoint cloud into a blob and thins *that* to smooth
  out the raw midpoints' jaggedness, then fits vector curves to the result.

**Image polarity convention**: binary/edge images throughout this codebase
use ink = 0 (black), background = 255 (white) — the opposite of OpenCV's
usual convention. Functions that touch `cv2.ximgproc` thinning or contour
extraction invert internally at their boundaries; keep this convention in
any new code.

## Stage 1 — Binarization (`skeletonize.py`)

- `binarize(gray)` — Otsu-threshold the grayscale input into a binary image.
- `detect_edges(gray)` — Canny edge detection on the *original* grayscale
  image (not the binarized one). Used only to render the two debug overlay
  PNGs later; it plays no role in the actual centerline/vector math.

## Stage 2 — Centerline estimation (`centerline.py`)

`find_centerline_points(binary, ...)` is the core algorithm. It finds, for
every point on the ink's boundary, the point directly opposite it across the
stroke, and takes the midpoint of that pair as a centerline sample.

1. **`extract_contours`** — walks OpenCV contours (`RETR_LIST`,
   `CHAIN_APPROX_NONE`) of the binarized ink. This includes both outer
   silhouettes and hole boundaries (e.g. the inside of an "O"), and
   contours shorter than 8 points are dropped as noise.

2. **Distance transform** — `cv2.distanceTransform` on the ink mask gives,
   at every ink pixel, its distance to the nearest background pixel. This
   drives both stroke-width estimation and the "genuine crossing" check
   below.

3. **`_estimate_stroke_width`** — when `stroke_width` isn't passed in, it's
   derived from the ridge of the distance transform: pixels that are a
   local maximum of the distance field (`distance >= dilate3x3(distance)`)
   sit on the centerline already, so twice their median distance value is a
   good estimate of stroke width.

4. **`_tangent_normals`** — for each boundary point, the local tangent is
   estimated from points `tangent_span` steps ahead/behind it along the
   *same* contour; the tangent's perpendicular is the candidate normal.

5. **`_orient_normals_inward`** — flips each normal so it points into the
   ink rather than out of it, by sampling the distance transform a small
   `normal_eps` step along `+normal` vs. `-normal` and keeping whichever
   side has the larger (i.e. more interior) value.

6. **Pair selection (`select_pairs`)** — for each boundary point `i`, a
   KD-tree (`search_radius = stroke_width * search_margin`) finds nearby
   boundary points `j`. A candidate pair must satisfy all of:
   - **Not the same local stretch of contour**: points within
     `exclusion_span` steps along the *same* contour are skipped, since
     adjacent boundary points are always trivially close.
   - **Width band**: `|i − j|` falls within
     `[stroke_width * width_min_ratio, stroke_width * width_max_ratio]`.
   - **Facing normals**: the direction from `i` to `j` must align with `i`'s
     inward normal, and the reverse direction must align with `j`'s inward
     normal (`angle_tolerance_deg`) — a signed check, not just "roughly
     parallel," so two nearby-but-unrelated boundary segments don't match.
   - **Genuine narrowest crossing**: the pair's midpoint distance-transform
     value must be at least `medial_tolerance * dist / 2` — i.e. the
     midpoint is genuinely about as far from *both* boundaries as the pair
     is long, ruling out chords that graze across a corner or a junction
     notch instead of crossing straight through a stroke.

   Among all candidates for a point, the one whose length is closest to the
   current `stroke_width` estimate wins.

7. **Width refinement** — if `stroke_width` was auto-estimated, it's
   recomputed as the median of the first pass's matched-pair distances; if
   that moved the estimate by more than ~2%, pair selection reruns once
   with the refined width.

8. **Output** — the midpoints of all surviving pairs, plus the
   (possibly refined) `stroke_width`.

Debug helper: `overlay_edges_midpoints` draws the Canny edges (red) and the
midpoint cloud (blue dots) into one image for visual sanity-checking.

## Stage 3 — Vectorization (`vectorize.py`)

`vectorize_midpoints(midpoints, shape, stroke_width, ...)` turns the raw
point cloud into a smooth SVG:

1. **`rasterize_midpoints`** — draws each midpoint as a small filled circle
   (`raster_radius`) into a blank mask, so adjacent midpoints along a
   stroke fuse into one solid blob.

2. **`thin_mask`** — `cv2.ximgproc.thinning` (Guo-Hall) reduces each blob to
   a 1px-wide skeleton. This is the step that smooths out the raw
   midpoints' pixel-level jaggedness — the only pixel-thinning in the whole
   pipeline.

3. **`trace_skeleton`** — walks the thinned mask's 8-connected pixel graph
   into ordered point chains (`lines`), splitting at endpoints/junctions
   (degree ≠ 2), or tracing all the way around for a closed loop with no
   junction at all. Also returns each pixel's `degree` for use below.

4. **`prune_spurs`** — drops two kinds of thinning artifacts, using
   `min_length = stroke_width * spur_length_ratio` and
   `min_junction_gap = stroke_width * junction_gap_ratio`:
   - short dangling branches hanging off a junction (one free end, one
     junction end, short);
   - the tiny sliver edges connecting the handful of pixels a single wide
     junction gets fragmented into (both ends at junctions, short).
   Lines with two free ends are always kept, since they're a genuine
   separate stroke regardless of length.

5. **`connect_opposite_lines`** — bridges dangling ends that face each
   other head-on: close together (within
   `stroke_width * connect_gap_ratio`) with outward tangents pointing at
   one another (`angle_tolerance_deg`). This reconnects a single stroke
   that got broken by a gap in the centerline midpoints — typically near a
   corner or junction where `find_centerline_points` rejected candidates as
   ungenuine grazes. Runs iteratively (closest opposing pair first),
   re-deriving which ends are still "dangling" from the current line set on
   each pass, since pruning upstream can leave a lone surviving edge whose
   *stale* graph degree still looks like it's part of a multi-way junction.

6. **`connect_ends_to_lines`** — extends a still-dangling end onto another
   line it runs straight into (within `stroke_width * attach_gap_ratio`,
   `attach_angle_deg`), closing T-junctions that step 5 can't: e.g. the
   crossbar of an "H" stops short of the stem's centerline, because the
   stem passes straight through and never has a dangling end there to weld
   to. Deliberately runs *after* `connect_opposite_lines` so genuine
   head-on continuations are merged first and aren't mistaken for
   T-attachments.

7. **`simplify_polyline`** (`cv2.approxPolyDP`, `simplify_epsilon`) — thins
   out redundant nearly-collinear points before curve fitting.

8. **`_catmull_rom_bezier_segments` / `polyline_to_svg_path`** — fits a
   smooth cubic-Bezier curve through the simplified points (Catmull-Rom
   tangents), closing the path with `Z` for loops.

9. **`build_svg`** — renders each path as an SVG `<path>` stroked at the
   estimated `stroke_width`, with `fill="none"` and round caps/joins (round
   caps also visually seal any small gap that step 5/6 didn't bridge).

Debug helper: `overlay_vector_paths` draws the Canny edges (red) and the
final vector paths (green) into one image.

## Stage 4 — Orchestration (`skeletonize.py::process_image`)

Ties the above together for one input file and writes, into the output
directory:

| File | Contents |
|---|---|
| `<stem>-binarize.png` | Otsu-thresholded black & white image |
| `<stem>-canny-edges.png` | Canny edges of the original grayscale image |
| `<stem>-edges-midpoints-overlay.png` | edges (red) + centerline midpoints (blue dots) |
| `<stem>-vector-skeleton-overlay.png` | edges (red) + final vector paths (green) |
| `<stem>.svg` | the vectorized result |

`__init__.py::main` is the CLI entry point: it parses `input`/`output`
(positional, defaulting to `input/skeletonize` / `output/skeletonize`) plus
four tunable ratios, then either processes a single file or recurses over a
directory of images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).

## Tunable parameters (CLI flags)

| Flag | Default | Effect |
|---|---|---|
| `--width-min-ratio` | `0.95` | Lower bound (× stroke width) on accepted boundary-pair chord length |
| `--width-max-ratio` | `1.05` | Upper bound (× stroke width) on accepted boundary-pair chord length |
| `--attach-gap-ratio` | `1.0` | Max distance (× stroke width) for extending a dangling end onto a line it runs into (T-junctions) |
| `--attach-angle` | `60.0` | Max angle (degrees) between a dangling end's direction and the line it's being attached to |

Everything else (`search_margin`, `exclusion_span`, `angle_tolerance_deg` in
`centerline.py`; `spur_length_ratio`, `junction_gap_ratio`,
`connect_gap_ratio`, `simplify_epsilon` in `vectorize.py`) is derived from
the auto-detected `stroke_width` and exposed as function defaults rather
than CLI flags — override them by calling the module functions directly if
a particular image needs it.
