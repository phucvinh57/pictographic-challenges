# Challenge 1: raster outline to SVG

Challenge 1 converts a dark-on-light raster image into a compact, filled SVG.
It traces the boundaries of the ink rather than finding stroke centerlines, so
the result preserves the silhouette, separate components, and enclosed holes
of the source image.

For installation and the short command reference, see the
[project README](../README.md).

## Pipeline at a glance

```text
Input image
    |
    v  read as grayscale
Grayscale pixels
    |
    v  Otsu or fixed threshold
Black-and-white image ---------------------> *-1-binarize.png
    v
4-connected foreground components and enclosed background components
    |
    v  closed boundary walk, with opposite winding for ink and holes
Raw lattice contours
    |
    v  remove staircases and redundant points
Simplified contours -----------------------> *-2-contours.png
    |
    v  detect corners and subdivide long smooth runs
Corner-aware point chains
    |
    v  adaptive cubic Bezier fitting
Closed Bezier contours
    |                    |
    v                    v
*-vector.svg        *-filled.svg
```

## 1. Input discovery and grayscale loading

`challenge_1.main` accepts either one image or a directory. Directory input is
searched recursively for PNG, JPEG, BMP, and TIFF files, and its relative
directory structure is preserved below the output directory.

`process_image` loads each image with `cv2.IMREAD_GRAYSCALE`. Processing stops
with a clear error if OpenCV cannot decode the file.

## 2. Binarization

`binarize` applies a normal binary threshold, giving the pipeline one polarity
throughout:

- ink/foreground: `0` (black)
- background: `255` (white)

By default, OpenCV's Otsu method chooses the threshold from the grayscale
histogram. `--threshold LEVEL` bypasses Otsu and uses a fixed value from 0 to
255. This pipeline therefore assumes that the subject is darker than its
background.

The result is written unchanged as `<stem>-1-binarize.png`.

## 3. Closed contour tracing

`foreground_contours` builds the shape directly from connected components in
the binary image:

1. Black pixels are labeled with four-connectivity. Every black component is
   traced as an outer contour.
2. White pixels are labeled in the same way. Components touching an image
   border are discarded; the remaining white components are enclosed holes.
3. `_trace_boundary` walks the boundary between the two pixel fields. It emits
   the end of each straight lattice run rather than every pixel-sized step.
4. Ink boundaries and hole boundaries are walked in opposite directions.

The output of this stage is a set of closed polygonal contours. Because holes
are found explicitly, shapes such as letters with counters and nested artwork
survive the conversion.

## 4. Staircase removal and corner-aware smoothing

Each raw contour first passes through `simplify_staircase` in
`pictographic.curves`, which:

- removes one-pixel zigzags caused by the square raster grid;
- limits how far a collapsed run may move from the original path; and
- removes points that are still collinear.

This is geometric cleanup, not image blur or morphological filtering.

The simplified contours are drawn in separate random colors on
`<stem>-2-contours.png`, making it possible to inspect exactly what this stage
passes to subdivision and fitting.

Next, `subdivide_closed_path` marks any direction change at or above
`--angle-threshold` as a corner. The default is 60 degrees. Long non-corner
segments are then repeatedly split with curvature-aware points, while marked
corners stay pinned. This creates enough samples for smooth fitting without
rounding deliberate sharp features.

A lower angle threshold preserves more direction changes as corners. A higher
value treats more of the contour as a smooth run.

## 5. Adaptive cubic Bezier fitting

`fit_closed_contour` divides the loop at preserved corners, curvature-direction
changes, and accumulated turns. Each section is densified to provide fitting
witnesses, then `_fit_cubics`:

1. estimates endpoint tangents;
2. solves the two cubic control-handle lengths by least squares;
3. measures the fitted curve against the witness points; and
4. recursively splits at the largest error until the requested tolerance is
   met.

The final curve is a closed chain of `BezierCurve` values. Adjacent pieces meet
continuously along smooth runs, while corner cuts allow a deliberate tangent
break.

`--smooth-tolerance` controls the accepted sampled fitting error in pixels. The
default is 1.5 pixels. A lower value follows the contour more closely and
usually creates more SVG curve segments; a higher value gives a smaller,
smoother result.

## 6. Outputs

The contour debug image is drawn after staircase removal. The vector and filled
outputs are drawn from the fitted Bezier contours.

| File | Purpose |
|---|---|
| `<stem>-1-binarize.png` | Thresholded source, with black ink on white |
| `<stem>-2-contours.png` | Staircase-simplified boundaries, one random color per contour |
| `<stem>-vector.svg` | Debug SVG with colored 1 px Bezier contours and anchor dots |
| `<stem>-filled.svg` | Final compact black-on-white filled SVG |

The final SVG combines all closed contours into one compound path and uses
`fill-rule="evenodd"`. Crossing each nested boundary toggles the fill, which
reconstructs holes without depending on contour order. Coordinates are rounded
to at most two decimal places, and straight cubic spans are emitted as simpler
SVG line commands where possible.

The random debug colors can change between runs. They do not affect the final
filled SVG.

## Running the pipeline

Process every supported image in the default input directory:

```bash
uv run challenge_1
```

Process one image:

```bash
uv run challenge_1 \
  --input input/challenge_1/cabinet.png \
  --output output/challenge_1
```

Tune classification and fitting independently:

```bash
uv run challenge_1 \
  --threshold 128 \
  --angle-threshold 75 \
  --smooth-tolerance 1
```

| Option | Default | Effect |
|---|---:|---|
| `--input`, `-i` | `input/challenge_1` | Source image or recursive source directory |
| `--output`, `-o` | `output/challenge_1` | Root directory for generated files |
| `--threshold`, `-t` | Otsu | Fixed grayscale split from 0 to 255 |
| `--angle-threshold`, `-a` | `60` | Minimum turn, in degrees, kept as a corner |
| `--smooth-tolerance`, `-s` | `1.5` | Maximum sampled Bezier fitting error, in pixels |

## Code map

| Function/module | Responsibility |
|---|---|
| `challenge_1.main` | CLI validation, recursive discovery, and output naming |
| `challenge_1.process_image` | Runs the stages for one image and writes all four outputs |
| `challenge_1.binarize` | Otsu or fixed thresholding |
| `challenge_1.foreground_contours` | Connected-component discovery and closed boundary tracing |
| `challenge_1.simplify_contours` | Removes pixel staircases and redundant contour points |
| `challenge_1.draw_contours` | Raster debug view of the simplified contours |
| `pictographic.curves.subdivide_closed_path` | Corner preservation and smooth-run subdivision |
| `pictographic.curves.fit_closed_contour` | Adaptive cubic Bezier fitting |
| `pictographic.svg.bezier_svg` | Colored vector debug view and anchor markers |
| `pictographic.svg.filled_bezier_svg` | Final compound even-odd SVG |

## Scope and trade-offs

- Challenge 1 vectorizes outlines. It does not infer the original pen
  centerlines or stroke widths; that is the separate `skeletonize` pipeline.
- The input should have a reasonably separable dark subject and light
  background. Uneven lighting or light-colored ink may need preprocessing or a
  manually selected threshold.
- Four-connectivity keeps diagonally touching black regions separate. This is
  intentional and matches the boundary walk, but a one-pixel diagonal contact
  may therefore behave differently from an edge-connected region.
- There is no automatic speckle removal. Small isolated foreground components
  can become their own SVG contours, so noisy scans should be cleaned before
  vectorization.
