# pictographic

Binarizes images, thins them down to a 1px skeleton using the Zhang-Suen
algorithm (via OpenCV's `cv2.ximgproc.thinning`), then vectorizes that
skeleton's centerline into an SVG drawn with a single fixed stroke width.

For each input image, three outputs are produced:

- `<name>-binarize.png` — Otsu-thresholded black & white image
- `<name>-ZhangSuen-skeletonize.png` — thinned skeleton of the binary image
- `<name>-ZhangSuen.svg` — the skeleton's centerline traced into polylines
  and rendered as SVG `<path>` elements with a fixed `stroke-width`

Vectorizing walks the skeleton's foreground pixels as a graph: endpoints and
branch points become path breaks, straight runs get simplified with
Ramer-Douglas-Peucker, and short spurious branches left by thinning (a known
Zhang-Suen artifact, worst on near-45-degree strokes) are pruned using a
threshold derived from the image's own ink thickness.

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

Process every image under `input/challenge_2` and write results to
`output/challenge_2` (this is the default when no arguments are given):

```bash
uv run challenge_2
```

Process a single file, output going to a chosen directory:

```bash
uv run challenge_2 input/challenge_2/letter_K.png output/challenge_2
```

Process a different directory into another directory:

```bash
uv run challenge_2 input/challenge_1 output/challenge_1
```

Override the fixed stroke width used for the vectorized SVG (default `45`):

```bash
uv run challenge_2 input/challenge_2 output/challenge_2 --stroke-width 20
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

Open a Python shell inside the environment:

```bash
uv run python
```

## Project layout

```
pyproject.toml              project metadata & dependencies (edit by hand or via `uv add`/`uv remove`)
uv.lock                      locked dependency versions (committed, don't edit by hand)
.python-version              pinned Python version for uv
src/challenge_2/
  __init__.py                CLI entry point (argument parsing, directory walking)
  skeletonize.py              binarize() and thin_zhang_suen() implementations
  vectorize.py                 skeleton -> polylines -> SVG (spur pruning, RDP simplification)
input/                        source images, organized by challenge
output/                       generated binarize/skeletonize/SVG results
```
