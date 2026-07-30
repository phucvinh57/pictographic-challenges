# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv sync                        # install/update .venv from pyproject.toml + uv.lock
uv run skeletonize              # process input/skeletonize -> output/skeletonize (default args)
uv run skeletonize <in> <out>   # process a single file or a directory into <out>
uv run skeletonize <in> <out> --width-min-ratio 0.85 --width-max-ratio 1.15  # tune the accepted chord-length band
uv run python -m skeletonize    # equivalent to the `skeletonize` script, without install
uv add <package> / uv add --dev <package> / uv remove <package>
uv lock --upgrade && uv sync    # upgrade locked dependencies
```

There is no test suite or lint config in this repo currently.

## Architecture

The pipeline (`src/skeletonize/`) takes a grayscale image of a single-color-ink
drawing/glyph and estimates its stroke centerlines *without* a skeletonization
(thinning / medial-axis) step. Instead, it works directly from the ink's
boundary contours:

1. `skeletonize.py::binarize` — Otsu-threshold the grayscale image.
2. `skeletonize.py::detect_edges` — Canny edge detection on the original
   grayscale image, used only for the debug overlay output.
3. `centerline.py::find_centerline_points` — the core algorithm:
   - `extract_contours` walks OpenCV contours of the binarized ink (outer
     silhouettes and hole boundaries alike).
   - `_tangent_normals` estimates, at each boundary point, the local tangent
     (from points a fixed span ahead/behind along the contour) and its
     perpendicular normal.
   - `_estimate_stroke_width` derives `stroke_width` up front from the ridge
     of the ink's own distance transform (local maxima of the distance field,
     doubled), when one isn't given. `search_radius` and `exclusion_span`
     scale from that width when not given.
   - `_orient_normals_inward` flips each boundary point's normal so it points
     into the ink, using the distance transform to tell which side is
     interior.
   - For each boundary point, a KD-tree lookup finds nearby boundary points;
     among those, a candidate must lie within a `stroke_width`-relative
     distance band, have both endpoints' normals facing each other (a signed
     alignment check, not just parallel), and have its midpoint's distance
     transform value close to half the chord length (i.e. the chord is a
     genuine locally-narrowest crossing, not a graze across a corner or
     junction notch). Among surviving candidates, the one whose length is
     closest to `stroke_width` is kept as the point directly across the
     stroke. Points within `exclusion_span` steps along the *same* contour
     are skipped since they're always trivially close.
   - `stroke_width` is refined once from the median of the first pass's
     matched-pair distances, and pair selection reruns with the refined value
     if it moved by more than ~2%.
   - The midpoints of the surviving pairs are the estimated centerline
     points.
4. `skeletonize.py::process_image` orchestrates the above per image, writing
   `<stem>-binarize.png`, `<stem>-canny-edges.png`, and
   `<stem>-edges-midpoints-overlay.png` (edges + midpoints composited via
   `centerline.py::overlay_edges_midpoints`) into the output directory.
5. `__init__.py::main` is the CLI entry point: parses `input`/`output`
   (positional, defaulting to `input/skeletonize` / `output/skeletonize`), and
   either processes a single file or recurses over a directory of images
   (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).

**Image polarity convention**: throughout this codebase, binary/edge images
use foreground (ink) = 0 (black), background = 255 (white) — the opposite of
OpenCV's usual white-foreground convention. Functions that consume or
produce such images (e.g. `cv2.ximgproc`-style thinning, contour extraction)
invert as needed at their boundaries; keep new code consistent with this
polarity.

`input/` and `output/` are gitignored — images placed there for local testing
don't need to be committed.

## Rules

- DO NOT add comments unnecessarily.