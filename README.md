# pictographic

Binarizes images, then estimates each stroke's centerline directly from the
ink's boundary contours — no skeletonization (medial axis / thinning) step
involved. The stroke width is auto-detected from the ink's distance
transform. For each boundary point, the local tangent gives a perpendicular
line pointing into the ink; among nearby boundary points lying roughly along
that line, with a midpoint that is a genuine narrowest crossing of the
stroke (not a graze past a corner or junction), the one whose distance is
closest to the stroke width is treated as the point directly across the
stroke, and the midpoint of the pair approximates a centerline point.

For each input image, the following outputs are produced:

- `<name>-binarize.png` — Otsu-thresholded black & white image
- `<name>-canny-edges.png` — Canny edge detection on the original grayscale
  image (traces the outline of each stroke)
- `<name>-edges-midpoints-overlay.png` — the Canny edges (red) and the
  collected centerline midpoints (blue dots) drawn into a single image, so
  the midpoints' position inside the stroke's outline can be checked at a
  glance

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/). Install `uv`
first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this directory, install the project and its dependencies into a
local virtual environment (`.venv`):

```bash
uv sync
```

`uv sync` reads `pyproject.toml` / `uv.lock` and creates/updates `.venv`
automatically — you don't need to activate it yourself, `uv run` (below) does
that for you.

## Run

Process every image under `input/skeletonize` and write results to
`output/skeletonize` (this is the default when no arguments are given):

```bash
uv run skeletonize
```

Process a single file, output going to a chosen directory:

```bash
uv run skeletonize input/skeletonize/letter_K.png output/skeletonize
```

Process a different directory into another directory:

```bash
uv run skeletonize input/challenge_1 output/challenge_1
```

The ink stroke width is auto-detected per image from its distance transform,
so no manual tuning is needed. A boundary-point pair is only kept if its
distance falls within `--width-min-ratio`/`--width-max-ratio` of that
detected width (default `0.9`/`1.1`, i.e. within ±10%):

```bash
uv run skeletonize input/skeletonize output/skeletonize --width-min-ratio 0.85 --width-max-ratio 1.15
```

You can also run the module directly without the installed script name:

```bash
uv run python -m skeletonize
```

## Managing the project

Add a runtime dependency:

```bash
uv add <package>
```

Add a dev-only dependency (e.g. a linter or test runner):

```bash
uv add --dev <package>
```

Remove a dependency:

```bash
uv remove <package>
```

Upgrade all dependencies to the latest versions allowed by `pyproject.toml`:

```bash
uv lock --upgrade
uv sync
```

Run an ad-hoc command inside the project's environment:

```bash
uv run <command>
```

Open a Python shell inside the environment:

```bash
uv run python
```

## Project layout

```
pyproject.toml              project metadata & dependencies (edit by hand or via `uv add`/`uv remove`)
uv.lock                      locked dependency versions (committed, don't edit by hand)
.python-version              pinned Python version for uv
src/skeletonize/
  __init__.py                CLI entry point (argument parsing, directory walking)
  skeletonize.py              binarize(), detect_edges(), and process_image() orchestration
  centerline.py                contour extraction + perpendicular-matching centerline estimation
input/                        source images, organized by challenge
output/                       generated binarize/edges/overlay results
```
