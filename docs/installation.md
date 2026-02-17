# Installation

```{tip}
If you prefer a graphical interface, try [alproj-gui](https://github.com/0kam/alproj-gui) — a desktop GUI application for georectification powered by alproj, built with Tauri. No Python coding required.
```

**Requires Python 3.9-3.12**

## Basic Installation
```bash
pip install git+https://github.com/0kam/alproj
```

## Optional Dependencies

### Advanced Image Matching (vismatch)
For advanced deep learning-based image matching methods (RoMa, LoFTR, LightGlue, etc.), install with the vismatch option:

```bash
pip install "alproj[vismatch] @ git+https://github.com/0kam/alproj"
```

This installs the [vismatch](https://github.com/gmberton/vismatch) package along with PyTorch and other dependencies.

These methods can provide more robust matching for challenging images with different resolutions, lighting conditions, or low-texture regions. GPU (CUDA) is recommended for best performance.

## Dependencies
Core dependencies (automatically installed):
```
numpy>=1.20,<3
pandas>=1.3
rasterio>=1.2
opencv-python>=4.5
moderngl>=5.6
cmaes>=0.8
tqdm>=4.60
scipy>=1.7
```

## Tested environments
- Ubuntu 20.04 LTS / Ubuntu24.04 LTS / macOS with Python 3.9-3.12