# CLAUDE.md

## Commands

```bash
uv sync
uv run skeletonize
uv run skeletonize <input> <output>
uv run python -m skeletonize
uv run ruff check .
uv run pyright
```

## Architecture

The pipeline binarizes an image with Otsu thresholding and extracts the medial
axis of the original ink mask with distance-ordered, topology-preserving
thinning. It traces the axis into SVG paths and renders them at twice the median
medial-axis radius.

Junctions are sharpened by cutting every branch back to the rim of the
junction's inscribed disc and refilling the emptied disc: branches whose fitted
tangents oppose each other are paired into a stroke bridged smoothly through the
junction, and each remaining branch is extended along its own fitted curve until
it cuts that through-stroke. Which branches pair is decided by geometry, so
`sharpen_junctions` relabels the ends it resolves with synthetic nodes to make
`merge_degree_two_nodes` stitch the same branches it paired. A single shared
least-squares vertex is only the fallback for junctions with no opposing pair.

`skeletonize.py::process_image` writes `<stem>-binarize.png`, `<stem>-skeleton.png`,
`<stem>-skeleton-cut-intersections.png`, `<stem>-sharpened.png`, a sharpened/rounded
comparison overlay, an edge/medial-axis diagnostic overlay, and `<stem>.svg`.

Binary images use ink = 0 (black) and background = 255 (white).

## Rules

- DO NOT add comments unnecessarily.
- DO NOT write tests
- Run `uv run ruff check .` and `uv run pyright` after modifying code.
