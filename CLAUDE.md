# CLAUDE.md

## Commands

```bash
uv sync
uv run skeletonize
uv run skeletonize <input> <output>
uv run python -m skeletonize
```

## Architecture

The pipeline binarizes an image with Otsu thresholding and extracts the medial
axis of the original ink mask with `skimage.morphology.medial_axis`.
`skeletonize.py::process_image` writes `<stem>-binarize.png` and
`<stem>-medial-axis.png`, plus an edge/medial-axis diagnostic overlay.

Binary images use ink = 0 (black) and background = 255 (white).

## Rules

- DO NOT add comments unnecessarily.
