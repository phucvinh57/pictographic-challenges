# pictographic

Binarizes images and recovers the pen strokes a drawing was made of.

Each input image produces one output, `<name>.svg`: one cubic Bezier path per
stroke, smooth along its length while corners come to a point, stroked at the
ink's own width with round caps and joins over a white background.

For a stage-by-stage explanation of the outline vectorizer, see
[Challenge 1: raster outline to SVG](docs/challenge-1.md).

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

Binarize every challenge 1 image, trace each foreground contour, collapse its
pixel staircases, preserve its corners, and fit the smooth runs as cubic Bezier
curves:

```bash
uv run challenge_1
```

Each input produces `<name>-1-binarize.png`, the color debug image
`<name>-2-contours.png`, `<name>-vector.svg`, and `<name>-filled.svg`. You can
also process a different image or directory:

The contour image shows the closed boundaries extracted from the binary image
after pixel staircases and redundant points are removed. The vector preview and
compact filled SVG come from the fitted contours. The vector output marks the
Bezier anchors over each colored contour. The filled output puts every closed
contour in one compound path and fills alternate regions using SVG's even-odd
rule.

The binary image uses black foreground pixels on white. The contour debug image
uses one random color per contour.

```bash
uv run challenge_1 --input input/challenge_1/cabinet.png --output output/challenge_1
uv run challenge_1 -i path/to/images -o path/to/output
```

Otsu thresholding is used by default. Provide a fixed threshold from 0 to 255
when needed:

```bash
uv run challenge_1 --threshold 128
```

The tracer follows VTracer's spline stages. A four-connected boundary walker
traces foreground components and enclosed holes as straight runs with opposite
winding. It then removes one-pixel staircase turns, simplifies the remaining
lattice path, detects corners, and repeatedly subdivides long smooth segments
before fitting Beziers. A direction change of 60 degrees or more stays pinned
as a corner by default:

```bash
uv run challenge_1 --angle-threshold 75
```

Bezier fitting may deviate by at most 1.5 pixels from the smoothed contour by
default. A smaller tolerance follows it more closely and usually creates more
curves; a larger tolerance produces a simpler result:

```bash
uv run challenge_1 --smooth-tolerance 1
```

Process every image under `input/skeletonize` and write results to
`output/skeletonize` (this is the default when no arguments are given):

```bash
uv run skeletonize
```

Process a single file, output going to a chosen directory:

```bash
uv run skeletonize --input input/skeletonize/letter_K.png --output output/skeletonize
```

Stroke samples are spaced 50 pixels apart by default. Choose another spacing with
`--sample-spacing`:

```bash
uv run skeletonize -s 5 -i input/skeletonize/letter_K.png -o output/skeletonize
```

A junction is where strokes overlap, not a hole to be patched: the drawing's
medial axis is used only to find where the strokes meet and which branches carry
the same one. Each stroke then claims the ink around the branches it runs along,
junctions included, so two strokes crossing simply share the ink there — and its
own centre line is the medial axis of that, which no longer has the crossing in
it to bend around.

Where two strokes leave one shared cap, as in the middle of a `3`, they overlap
all the way out to it and both run to the cap. Where one stroke turns a corner,
its two sides meet at the point their own lines cross, and the stroke keeps that
point instead of being smoothed back through it.

The curves are stroked at the ink's own width, twice the median distance from
the axis to the edge of the ink. Override it with `--stroke-width`:

```bash
uv run skeletonize -w 12 -i input/skeletonize/letter_K.png -o output/skeletonize
```

Process a different directory into another directory:

```bash
uv run skeletonize --input input/challenge_1 --output output/challenge_1
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

Lint the project:

```bash
uv run ruff check .
```

Type-check the project with the same engine used by VS Code's Pylance:

```bash
uv run pyright
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
src/challenge_1/
  __init__.py                 binarization, contour smoothing, previews, and CLI
src/pictographic/
  curves.py                   shared centerline and closed-contour Bézier fitting
  graph.py                    shared skeleton graph types and tracing
  svg.py                      shared stroked SVG output
src/skeletonize/
  __init__.py                CLI entry point (argument parsing, directory walking)
  skeletonize.py              binarize() and process_image() orchestration
  medial_axis.py              Zhang-Suen medial-axis thinning and the ink distance map
  graph.py                    axis tracing into edges and junctions
  strokes.py                  stroke separation at the junctions and per-stroke axes
  curves.py                   arc-length resampling and Catmull-Rom Bezier fitting
  svg.py                      stroked SVG output
input/                        source images, organized by challenge
output/                       generated SVG results
```
