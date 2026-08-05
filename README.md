# pictographic

Binarizes images and traces their ink boundaries as smooth vector contours.

Each input image produces a colored contour preview and a compact filled SVG.

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
  curve_fitting.py            corner-aware adaptive Bézier fitting
  pipeline.py                 one-image processing and output paths
  raster.py                   threshold level selection and debug colors
  svg.py                      contour preview and filled SVG output
input/challenge_1/            source images
output/challenge_1/           generated SVG results
```
