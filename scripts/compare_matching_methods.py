#!/usr/bin/env python
"""
Compare all available matching methods for documentation.

This script compares all available matching methods (built-in and imm-based)
and outputs timing and match count results for documentation.

Output images are saved to docs/_static/ for inclusion in the documentation.
"""

import argparse
import platform
import time
import warnings
from pathlib import Path

import cv2
import torch

from alproj.gcp import image_match, IMM_METHODS

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")


# All available methods
BUILTIN_METHODS = ["akaze", "sift"]
ALL_METHODS = BUILTIN_METHODS + IMM_METHODS

# Output image size for web display
OUTPUT_IMAGE_SIZE = 640


def get_system_info():
    """Get system information for documentation."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def resize_image(img, max_size):
    """
    Resize image so that the longer side equals max_size.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (BGR).
    max_size : int
        Target size for the longer side.

    Returns
    -------
    numpy.ndarray
        Resized image.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sanitize_method_name(method):
    """Convert method name to a valid filename."""
    return method.replace("-", "_").lower()


def run_single_method(path_org, path_sim, method, device="cpu"):
    """
    Run a single matching method.

    Parameters
    ----------
    path_org : str
        Path to original photograph.
    path_sim : str
        Path to simulated image.
    method : str
        Matching method name.
    device : str
        Device for deep learning methods.

    Returns
    -------
    dict
        Result dictionary with time, matches, device, plot, and error (if any).
    """
    result = {
        "method": method,
        "time": None,
        "matches": 0,
        "device": "CPU" if method in BUILTIN_METHODS else device.upper(),
        "plot": None,
        "error": None,
    }

    try:
        start = time.time()
        if method in BUILTIN_METHODS:
            match, plot = image_match(
                path_org, path_sim,
                method=method,
                threshold=30,
                plot_result=True
            )
        else:
            match, plot = image_match(
                path_org, path_sim,
                method=method,
                device=device,
                threshold=30,
                plot_result=True
            )
        elapsed = time.time() - start

        result["time"] = elapsed
        result["matches"] = len(match)
        result["plot"] = plot

    except ImportError as e:
        result["error"] = f"ImportError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def run_comparison(path_org, path_sim, output_dir, device="cpu", methods=None):
    """
    Run comparison of matching methods.

    Parameters
    ----------
    path_org : str
        Path to original photograph.
    path_sim : str
        Path to simulated image.
    output_dir : str
        Directory to save output images.
    device : str
        Device for deep learning methods ("cpu", "cuda", or "mps").
    methods : list, optional
        List of methods to run. If None, runs all available methods.

    Returns
    -------
    dict
        Results dictionary keyed by method name.
    """
    if methods is None:
        methods = ALL_METHODS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for method in methods:
        print(f"Running {method}...", end=" ", flush=True)

        result = run_single_method(path_org, path_sim, method, device)

        if result["error"]:
            print(f"FAILED: {result['error']}")
        else:
            print(f"Time: {result['time']:.2f}s, Matches: {result['matches']}")

            # Save resized output image
            if result["plot"] is not None:
                resized_plot = resize_image(result["plot"], OUTPUT_IMAGE_SIZE)
                filename = f"matched_{sanitize_method_name(method)}.png"
                cv2.imwrite(str(output_path / filename), resized_plot)

        results[method] = result

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare all available matching methods"
    )
    parser.add_argument(
        "--org", default="devel_data/target_image.jpg",
        help="Path to original photograph"
    )
    parser.add_argument(
        "--sim", default="devel_data/sim_init.png",
        help="Path to simulated image"
    )
    parser.add_argument(
        "--output-dir", default="docs/_static",
        help="Directory to save output images"
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for deep learning methods"
    )
    parser.add_argument(
        "--methods", nargs="+",
        help="Specific methods to run (default: all)"
    )
    args = parser.parse_args()

    # Print system info
    print("=== System Info ===")
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print(f"  selected_device: {args.device}")
    print()

    # Get image dimensions
    img = cv2.imread(args.org)
    if img is None:
        print(f"Error: Could not read image: {args.org}")
        return 1
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h} pixels")
    print(f"Output image size: {OUTPUT_IMAGE_SIZE} pixels (longer side)")
    print()

    # Determine which methods to run
    methods = args.methods if args.methods else ALL_METHODS
    print(f"Methods to compare: {len(methods)}")
    print()

    # Run comparison
    print("=== Running Comparison ===")
    results = run_comparison(args.org, args.sim, args.output_dir, args.device, methods)

    # Print summary
    print()
    print("=== Summary ===")
    print("| Method | Time | Matches | Device | Status |")
    print("|--------|------|---------|--------|--------|")

    successful = 0
    failed = 0
    for method, data in results.items():
        if data["error"]:
            print(f"| {method} | - | - | - | FAILED |")
            failed += 1
        else:
            time_str = f"~{int(data['time'])} sec" if data['time'] >= 1 else f"{data['time']:.2f} sec"
            print(f"| {method} | {time_str} | {data['matches']} | {data['device']} | OK |")
            successful += 1

    print()
    print(f"Successful: {successful}/{len(results)}, Failed: {failed}/{len(results)}")
    print(f"Output images saved to {args.output_dir}/")

    return 0


if __name__ == "__main__":
    exit(main())
