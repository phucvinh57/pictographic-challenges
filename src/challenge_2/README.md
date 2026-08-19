# Challenge 2: Glyph skeleton extraction

For each image, the pipeline:

1. blurs the grayscale image and extracts its borders with Canny;
2. traces those borders with `findContours` and uses them to retain only ink
   components that belong to a substantial glyph boundary;
3. extracts the resulting binary glyph's medial axis;
4. builds a graph from the axis and detects where the larger radius around a
   junction settles to the ordinary stroke radius. If a junction-to-endpoint
   radius keeps falling without a knee, its first minimum-radius point is used
   as the transition instead;
5. removes each old junction, matches transition branches by their local
   direction, and rewrites the graph with merged edges or a replacement
   junction where unmatched branch tangents meet;
6. samples each rewritten edge at uniform arc-length intervals, then fits a
   cubic Bézier spline through those samples. Open-edge endpoints stay fixed and
   loops use cyclic tangents so their joins remain smooth.

The output is a white SVG whose graph edges are cubic Bézier paths drawn with a
fixed black stroke.

Run it with:

```bash
uv run challenge_2
```

Use `uv run challenge_2 --help` to change the input/output directories, Canny
thresholds, or minimum retained component area.

Use `--sample-spacing` to change the distance between points sampled before
Bézier smoothing (default `8` pixels). A larger value produces fewer, softer
curve spans.

Use `--stroke-width` to set the fixed width of every SVG edge (default `64`
pixel). Rounded caps and joins keep endpoints and reconstructed junctions
connected.

Transition sensitivity can be tuned with:

- `--transition-delta`: minimum relative radius drop from a junction to its
  stable stroke radius (default `0.05`);
- `--transition-flatness`: maximum stable-slope magnitude relative to the
  incoming slope (default `0.3`). Lower values require a sharper, flatter knee.

Pass `--debug` to also write two diagnostic images for each input:

- `<name>_debug.png` shows the original medial axis in light grey, the rewritten
  graph in dark grey, the smoothed graph in black, sampled points in purple, old
  junctions in blue, new junctions in green, and detected radius transitions in
  red;
- `<name>_radius_debug.png` plots every edge's raw, median-filtered, and fitted
  radius profile with its selected transition points.
