# Challenge 1: Raster outline to SVG

Each image goes through these processing stages:

1. **Binarize and trace contours**: Use Otsu threshold to pick the ink/paper cutoff, then marching squares
   walks the boundary at sub-pixel precision. Outputs are closed contours of points.
2. **Simplify contours**: drop every point that sits close enough to the chord replacing
   it (potrace/RDP-like style). Thousands of points become a polygon of dozens.
3. **Find the structure**: locate where the shape actually turns a corner, and
   which stretches are genuinely straight lines. Straight runs get collapsed into
   a single segment; corners become cut points where curves are not allowed to
   bend through.
4. **Fit curves**: between cuts, fit a cubic Bézier to the points. When the error is
   too large, split at the worst point and fit both halves, recursing until it fits.
   Tangents at the cuts are shared, so neighbouring curves join smoothly.

The contour processing and curve fitting stages live in `common.vectorization`.
Challenge 1 supplies closed boundary contours; challenge 2 uses the same code for
open and closed skeleton contours.

Then write the whole thing as one `<path>` with `fill-rule="evenodd"`, so holes
(inner rings) punch through outer rings for free.

Everything that could be a "distance in pixels" is instead a **ratio of the
contour's perimeter**, with a floor. That way a tiny glyph and a huge poster get
simplified with the same aggressiveness.

## Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant M as main
    participant C as contour
    participant F as curve_fitting
    participant S as svg

    M->>C: read image
    C-->>M: grayscale (alpha flattened on white)

    M->>C: extract contours
    Note right of C: Otsu threshold -><br/>marching squares -><br/>closed rings of (x, y)
    C-->>M: contours

    loop each contour
        M->>C: process contour
        Note right of C: drop redundant points -><br/>mark break points (sharp turns,<br/>long edges) -> detect straight runs
        C-->>M: corner points + straight flags

        M->>F: fit closed contour
        Note right of F: cut at corners &<br/>straight-run ends -><br/>estimate tangents -><br/>fit cubics, split on error
        F-->>M: Bézier curves
    end

    M->>S: draw svg
    Note right of S: flat cubics -> L,<br/>merge collinear runs,<br/>single path, evenodd
    S-->>M: svg text
    M->>M: write <name>.svg
```

## Knobs

All tunable from the CLI (`uv run challenge_1 --help`). The ones worth touching:

- `--simplify-ratio`: how much of the staircase to throw away.
- `--break-angle-threshold` / `--break-span-ratio`: what counts as a corner.
- `--straight-*`: how forgiving "this is a straight line" is.
- `--fit-ratio`: how close a curve must hug the contour before we stop
  splitting. Smaller means more curves, bigger file, more faithful.
- `-d/--debug`: per-stage timings and point counts.
