# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`alproj` is a Python package for georectification of alpine landscape photographs. It transforms ground-based camera images into geospatially referenced data using Digital Surface Models (DSM) and airborne photographs.

## Development Commands

```bash
# Install in development mode (use the existing .venv)
source .venv/bin/activate
pip install -e .

# Install with deep learning matching methods
pip install -e ".[vismatch]"

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_gcp.py

# Build documentation
cd docs && make html
```

## Architecture

The package follows a 4-step georectification pipeline:

```
1. Surface Generation (surface.py)
   - get_colored_surface(): Creates 3D mesh from DSM + aerial photo

2. Image Simulation (project.py)
   - sim_image(): Renders simulated view using OpenGL (moderngl)
   - reverse_proj(): Maps pixels to geographic coordinates

3. Feature Matching (gcp.py)
   - image_match(): Matches original photo with simulated image
   - set_gcp(): Creates Ground Control Points from matches

4. Parameter Optimization (optimize.py)
   - CMAOptimizer: Global search using CMA-ES
   - LsqOptimizer: Local refinement using least squares
```

### Key Data Flows

- **Camera params dict**: Contains x, y, z (position), fov, pan, tilt, roll (orientation), and 14 distortion coefficients (a1, a2, k1-k6, p1, p2, s1-s4)
- **Surface data**: `vert` (vertices), `col` (colors), `ind` (triangle indices), `offsets` (coordinate offsets)
- **GCPs DataFrame**: Columns `u, v` (image coords) and `x, y, z` (geographic coords in projected CRS like UTM)

### Matching Methods

- **Built-in** (no extra deps): `akaze`, `sift`
- **With vismatch package**: `minima-roma`, `superpoint-lightglue`, `roma`, `loftr`, etc.

## Code Conventions

- Coordinates use projected CRS (e.g., UTM in meters), not lat/lon
- Image coordinates: u (horizontal), v (vertical), origin at top-left
- Distortion model follows OpenCV conventions with extensions
- OpenGL rendering uses moderngl for cross-platform support
