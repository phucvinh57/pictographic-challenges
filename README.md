# pictographic

Binarizes images and recovers the pen strokes a drawing was made of.

Each input image produces one output, `<name>.svg`: one cubic Bezier path per
stroke, smooth along its length while corners come to a point, stroked at the
ink's own width with round caps and joins over a white background.

For a stage-by-stage explanation of the outline vectorizer, see
[Challenge 1: raster outline to SVG](docs/challenge_1.md).

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

Threshold every challenge 1 image, trace each ink boundary, collapse its pixel
staircases, preserve its corners, and fit the smooth runs as cubic Bezier
curves:

```bash
uv run challenge_1
```

Each input produces `<name>-vector.svg` and `<name>-filled.svg`, both from the
same fitted contours. The vector output is a preview: one random color per
contour, with the Bezier anchors marked. The filled output puts every closed
contour in one compound path and fills alternate regions using SVG's even-odd
rule, so enclosed holes come out as holes.

You can also process a different image or directory:

```bash
uv run challenge_1 --input input/challenge_1/cabinet.png --output output/challenge_1
uv run challenge_1 -i path/to/images -o path/to/output
```

Otsu thresholding is used by default. Provide a fixed threshold from 0 to 255
when needed:

```bash
uv run challenge_1 --threshold 128
```

The threshold only picks a level; the contours are then traced through the
original grayscale pixels with marching squares, so antialiased edges keep their
subpixel position instead of becoming staircases. Redundant points are dropped,
flat spans are measured and kept as exact lines, and the rest is fitted with
cubic Beziers. A direction change of 60 degrees or more stays pinned as a corner
by default:

```bash
uv run challenge_1 --angle-threshold 75
```

Bezier fitting may deviate by at most 0.75 pixels from the smoothed contour by
default. A smaller tolerance follows it more closely and usually creates more
curves; a larger tolerance produces a simpler result:

```bash
uv run challenge_1 --smooth-tolerance 1
```

Process every challenge 2 image under `input/challenge_2` and write results to
`output/challenge_2` (this is the default when no arguments are given):

```bash
uv run challenge_2
```

Each input produces `<name>.svg` and two debug images.
`<name>-skeleton.png` is the whole-drawing medial axis on its own, black on
white. `<name>-debug.png` is what the strokes were solved out of: the ink greyed
back, that same axis over it, and the axis pixels each cut claimed painted in —
red for a junction, orange for a corner's fan. A junction's focus is ringed at
the disc it was cut with, the rails it unfolded an overlap into are green, and
the smoothed strokes go on top, one color each.

Process a single file, output going to a chosen directory:

```bash
uv run challenge_2 --input input/challenge_2/letter_K.png --output output/challenge_2
```

Stroke samples are spaced 50 pixels apart by default. Choose another spacing with
`--sample-spacing`:

```bash
uv run challenge_2 -s 5 -i input/challenge_2/letter_K.png -o output/challenge_2
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
uv run challenge_2 -w 12 -i input/challenge_2/letter_K.png -o output/challenge_2
```

Process a different directory into another directory:

```bash
uv run challenge_2 --input input/challenge_1 --output output/challenge_1
```

You can also run the module directly without the installed script name:

```bash
uv run python -m challenge_2
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
  __init__.py                 public API exports
  cli.py                      CLI parsing, validation, and image discovery
  contours.py                 boundary tracing, cleanup, and curve fitting
  pipeline.py                 one-image processing and output paths
  raster.py                   threshold level selection and debug colors
src/pictographic/
  curves.py                   shared centerline and closed-contour Bézier fitting
  graph.py                    shared skeleton graph types and tracing
  svg.py                      shared stroked SVG output
src/challenge_2/
  __init__.py                CLI entry point (argument parsing, directory walking)
  pipeline.py                 binarize() and process_image() orchestration
  medial_axis.py              Zhang-Suen medial-axis thinning and the ink distance map
  graph.py                    axis tracing into edges and junctions
  strokes.py                  stroke separation at the junctions and per-stroke axes
  curves.py                   arc-length resampling and Catmull-Rom Bezier fitting
  debug.py                    the skeleton and solved-stroke debug images
  svg.py                      stroked SVG output
input/                        source images, organized by challenge
output/                       generated SVG results
```
