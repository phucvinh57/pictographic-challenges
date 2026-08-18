# pictographic

Binarizes images and traces their ink boundaries as smooth vector contours.

Each input image produces a compact filled SVG.

For a stage-by-stage explanation of the outline vectorizer, see:
- [Challenge 1: raster outline to SVG](src/challenge_1/README.md).

## Presequisite

### Git LFS

Images are stored in Git LFS. Please install `git-lfs` to pull them into your local repository.


### Set up project

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
## How to run

Active `venv` first:

```bash
source .venv/bin/activate
```

## Challenge 1

```bash
uv run challenge_1
```

Output will be under `./output/challenge_1`.

## Challenge 2

```bash
uv run challenge_2
```

This extracts glyph borders with Canny, then thins the enclosed ink to a
one-pixel skeleton. Output PNGs are written to `./output/challenge_2`.
