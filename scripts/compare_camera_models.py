#!/usr/bin/env python
"""Compare pinhole and fisheye camera models using Karasawa data.

Loads the optimized fisheye parameters, constructs equivalent pinhole
parameters, generates surface on the fly, and produces:
  1. Overlay comparison image (target + simulated blend)
  2. Matching result visualizations
  3. Reprojection error comparison plot

Usage (from project root):
    python scripts/compare_camera_models.py
"""

import json
import warnings

import cv2
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from alproj.gcp import image_match, set_gcp, filter_gcp_distance
from alproj.optimize import project, CMAOptimizer, LsqOptimizer
from alproj.project import sim_image, reverse_proj
from alproj.surface import get_colored_surface

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path("devel_data/karasawa")
OUTPUT_DIR = Path("docs/_static/camera_model_comparison")

TARGET_IMAGE_PATH = DATA_DIR / "target_image.jpg"
AIRBORNE_PATH = DATA_DIR / "airborne.tif"
DSM_PATH = DATA_DIR / "dsm.tif"
PARAMS_FISHEYE_PATH = DATA_DIR / "optimized_params.json"


def load_data():
    """Load raster data and optimized fisheye parameters."""
    with open(PARAMS_FISHEYE_PATH) as f:
        params_fisheye = json.load(f)

    # Build pinhole params from fisheye (same position/orientation, zero distortion)
    params_pinhole = {
        "x": params_fisheye["x"],
        "y": params_fisheye["y"],
        "z": params_fisheye["z"],
        "fov": params_fisheye["fov"],
        "pan": params_fisheye["pan"],
        "tilt": params_fisheye["tilt"],
        "roll": params_fisheye["roll"],
        "w": params_fisheye["w"],
        "h": params_fisheye["h"],
        "cx": params_fisheye["cx"],
        "cy": params_fisheye["cy"],
        "a1": 1, "a2": 1,
        "k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0, "k6": 0,
        "p1": 0, "p2": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0,
    }

    return params_fisheye, params_pinhole


def generate_surface(params):
    """Generate colored surface from raster data."""
    airborne = rasterio.open(str(AIRBORNE_PATH))
    dsm = rasterio.open(str(DSM_PATH))
    vert, col, ind, offsets = get_colored_surface(
        airborne, dsm, shooting_point=params, distance=4000, res=1.0)
    return vert, col, ind, offsets


def optimize_pinhole(vert, col, ind, offsets, params_pinhole):
    """Optimize pinhole model parameters using the same pipeline as fisheye."""
    target_path = str(TARGET_IMAGE_PATH)

    # Phase 1: position and orientation
    print("  Phase 1: position and orientation...")
    sim = sim_image(vert, col, ind, params_pinhole, offsets, min_distance=100)
    cv2.imwrite(str(OUTPUT_DIR / "sim_pinhole_init.png"), sim)
    df = reverse_proj(sim, vert, ind, params_pinhole, offsets)

    match, _ = image_match(
        target_path, str(OUTPUT_DIR / "sim_pinhole_init.png"),
        method="roma", plot_result=False, outlier_filter="fundamental",
        params=params_pinhole, resize=800, threshold=30.0,
        spatial_thin_grid=10, spatial_thin_selection="center")

    gcps = set_gcp(match, df)
    gcps = filter_gcp_distance(gcps, params_pinhole, min_distance=50)

    cma = CMAOptimizer(gcps[["x", "y", "z"]], gcps[["u", "v"]], params_pinhole)
    cma.set_target(["x", "y", "z", "fov", "pan", "tilt", "roll", "a1", "a2", "cx", "cy"])
    params_2nd, error = cma.optimize(
        generation=300, sigma=1.0, population_size=50, f_scale=10.0)
    print(f"    Error: {error:.2f} px")

    # Phase 2: distortion
    print("  Phase 2: distortion...")
    sim2 = sim_image(vert, col, ind, params_2nd, offsets, min_distance=100)
    cv2.imwrite(str(OUTPUT_DIR / "sim_pinhole_2nd.png"), sim2)
    df2 = reverse_proj(sim2, vert, ind, params_2nd, offsets)

    match, _ = image_match(
        target_path, str(OUTPUT_DIR / "sim_pinhole_2nd.png"),
        method="roma", plot_result=False, outlier_filter="essential",
        params=params_2nd, resize=800, threshold=30.0,
        spatial_thin_grid=10, spatial_thin_selection="center")

    gcps = set_gcp(match, df2)
    gcps = filter_gcp_distance(gcps, params_2nd, min_distance=100)

    cma = CMAOptimizer(gcps[["x", "y", "z"]], gcps[["u", "v"]], params_2nd)
    cma.set_target([
        "k1", "k2", "k3", "k4", "k5", "k6",
        "p1", "p2", "s1", "s2", "s3", "s4",
        "cx", "cy", "a1", "a2"])
    params_optim, error = cma.optimize(
        generation=300, sigma=1.0, population_size=50, f_scale=10.0)
    print(f"    Error: {error:.2f} px")

    return params_optim, gcps


def create_overlay(vert, col, ind, offsets, params_pinhole, params_fisheye):
    """Create target + simulation overlay comparison (3 panels)."""
    target = cv2.imread(str(TARGET_IMAGE_PATH))
    w, h = int(params_pinhole["w"]), int(params_pinhole["h"])
    target_resized = cv2.resize(target, (w, h))
    target_rgb = cv2.cvtColor(target_resized, cv2.COLOR_BGR2RGB)

    print("  Rendering pinhole simulation...")
    sim_pin = sim_image(vert, col, ind, params_pinhole, offsets)
    sim_pin_rgb = cv2.cvtColor(sim_pin, cv2.COLOR_BGR2RGB)

    print("  Rendering fisheye simulation...")
    sim_fish = sim_image(vert, col, ind, params_fisheye, offsets)
    sim_fish_rgb = cv2.cvtColor(sim_fish, cv2.COLOR_BGR2RGB)

    # Save simulations for matching
    cv2.imwrite(str(OUTPUT_DIR / "sim_pinhole.png"), sim_pin)
    cv2.imwrite(str(OUTPUT_DIR / "sim_fisheye.png"), sim_fish)

    # Overlay comparison
    alpha = 0.5
    blend_pin = cv2.addWeighted(sim_pin_rgb, alpha, target_rgb, 1 - alpha, 0)
    blend_fish = cv2.addWeighted(sim_fish_rgb, alpha, target_rgb, 1 - alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, img, title in [
        (axes[0], target_rgb, "Target Photo"),
        (axes[1], blend_pin, "Pinhole"),
        (axes[2], blend_fish, "Fisheye"),
    ]:
        ax.imshow(img, extent=[0, w, h, 0])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        "Simulation Overlay Comparison (50% blend)",
        fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUTPUT_DIR / "overlay_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def create_matching_results(params_pinhole, params_fisheye):
    """Run image_match for both models and save match visualizations."""
    target_path = str(TARGET_IMAGE_PATH)

    print("  Matching pinhole...")
    match_pin, plot_pin = image_match(
        target_path, str(OUTPUT_DIR / "sim_pinhole.png"),
        method="roma", plot_result=True, outlier_filter="fundamental",
        params=params_pinhole, resize=800, threshold=30.0,
        spatial_thin_grid=10, spatial_thin_selection="center")
    out_pin = OUTPUT_DIR / "matched_pinhole.png"
    cv2.imwrite(str(out_pin), plot_pin)
    print(f"  Saved: {out_pin} ({len(match_pin)} matches)")

    print("  Matching fisheye...")
    match_fish, plot_fish = image_match(
        target_path, str(OUTPUT_DIR / "sim_fisheye.png"),
        method="roma", plot_result=True, outlier_filter="fundamental",
        params=params_fisheye, resize=800, threshold=30.0,
        spatial_thin_grid=10, spatial_thin_selection="center")
    out_fish = OUTPUT_DIR / "matched_fisheye.png"
    cv2.imwrite(str(out_fish), plot_fish)
    print(f"  Saved: {out_fish} ({len(match_fish)} matches)")


def create_error_plot(gcps, params_pinhole, params_fisheye):
    """Compute and plot reprojection errors for both models."""
    img_arr = gcps[["u", "v"]].to_numpy()
    w, h = int(params_pinhole["w"]), int(params_pinhole["h"])

    proj_pin = project(gcps[["x", "y", "z"]], params_pinhole)
    err_pin = np.sqrt(
        (img_arr[:, 0] - proj_pin["u"].values) ** 2
        + (img_arr[:, 1] - proj_pin["v"].values) ** 2)

    proj_fish = project(gcps[["x", "y", "z"]], params_fisheye)
    err_fish = np.sqrt(
        (img_arr[:, 0] - proj_fish["u"].values) ** 2
        + (img_arr[:, 1] - proj_fish["v"].values) ** 2)

    mean_pin, mean_fish = np.mean(err_pin), np.mean(err_fish)

    # Print summary
    print(f"\n  {'Model':12s}  {'Mean':>8s}  {'Median':>8s}  {'P90':>8s}")
    print(f"  {'-'*42}")
    print(
        f"  {'Pinhole':12s}  {mean_pin:8.2f}  {np.median(err_pin):8.2f}  "
        f"{np.percentile(err_pin, 90):8.2f}")
    print(
        f"  {'Fisheye':12s}  {mean_fish:8.2f}  {np.median(err_fish):8.2f}  "
        f"{np.percentile(err_fish, 90):8.2f}")

    # Spatial error comparison (2 panels)
    vmax = max(np.percentile(err_pin, 95), np.percentile(err_fish, 95))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, errors, title in [
        (axes[0], err_pin, f"Pinhole (mean={mean_pin:.1f} px)"),
        (axes[1], err_fish, f"Fisheye (mean={mean_fish:.1f} px)"),
    ]:
        sc = ax.scatter(
            gcps["u"].values, gcps["v"].values,
            c=errors, cmap="RdYlGn_r", s=12, alpha=0.7,
            vmin=0, vmax=vmax)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Error (px)", shrink=0.8)

    fig.suptitle(
        "Reprojection Error Comparison",
        fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "reprojection_error.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params_fisheye, params_pinhole = load_data()

    print("=== Generating Surface ===")
    vert, col, ind, offsets = generate_surface(params_fisheye)

    print("\n=== Optimizing Pinhole Model ===")
    params_pinhole, gcps = optimize_pinhole(
        vert, col, ind, offsets, params_pinhole)

    print("\n=== Overlay Comparison ===")
    create_overlay(vert, col, ind, offsets, params_pinhole, params_fisheye)

    print("\n=== Matching Results ===")
    create_matching_results(params_pinhole, params_fisheye)

    print("\n=== Reprojection Error ===")
    create_error_plot(gcps, params_pinhole, params_fisheye)

    print(f"\nDone. Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
