# pictographic

Binarizes images and extracts each drawing's medial axis from the original
ink mask.

For each input image, the following outputs are produced:

- `<name>-binarize.png` — Otsu-thresholded black & white image
- `<name>-skeleton.png` — one-pixel medial axis of the original ink
- `<name>-skeleton-cut-intersections.png` — skeleton with every intersection's
  maximal-inscribed circle removed
- `<name>-sharpened.png` — the medial axis after each junction is refilled by
  pairing the branches that continue into each other, then extending every
  remaining branch until it cuts that through-stroke
- `<name>-sharpened-overlay.png` — the rounded medial axis (red) overlaid with
  the sharpened geometry (green)
- `<name>-edges-medial-axis-overlay.png` — Canny edges (red) with the medial
  axis overlaid in green
- `<name>.svg` — the sharpened medial-axis paths rendered at the stroke width
  estimated from the ink's distance transform

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
src/skeletonize/
  __init__.py                CLI entry point (argument parsing, directory walking)
  skeletonize.py              binarize() and process_image() orchestration
  medial_axis.py              distance-ordered medial-axis thinning and width estimation
  graph.py                    graph construction and intersection removal
  sharpen.py                  junction line-fitting and vertex sharpening
  vectorize.py                medial-axis tracing and SVG rendering
input/                        source images, organized by challenge
output/                       generated raster diagnostics and SVG results
```
