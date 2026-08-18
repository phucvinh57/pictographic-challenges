# Challenge 2: Glyph skeleton extraction

For each image, the pipeline:

1. blurs the grayscale image and extracts its borders with Canny;
2. traces those borders with `findContours` and uses them to retain only ink
   components that belong to a substantial glyph boundary;
3. thins the resulting binary glyph with Zhang-Suen thinning.

The output is a white PNG with a one-pixel black skeleton, preserving junctions
and loops in the original glyph.

Run it with:

```bash
uv run challenge_2
```

Use `uv run challenge_2 --help` to change the input/output directories, Canny
thresholds, or minimum retained component area.
