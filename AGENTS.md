# Agent Guide

This repository implements the *Modern Edge Detection Toolkit* as described in
`README.md`. The code relies on several deep learning models that must be
available in the `modern_edge_toolkit` subdirectories.

## Development rules

- Keep all code inside `modern_edge_toolkit/` aligned with the specification in
  the README. The entry point is `gui_modern.py`.
- No classical edge detectors such as Canny or Sobel are allowed.
- Do not add any CLI tools or Docker files.
- Unit tests reside in `tests/` and must be executed with `pytest` after
  changes.

