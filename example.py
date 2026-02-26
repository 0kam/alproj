"""
Tateyama example: Pinhole camera model.

This example demonstrates georectification using the standard pinhole
projection model with 14 image-space distortion coefficients
(a1, a2, k1-k6, p1, p2, s1-s4).
For a fisheye camera example, see devel_data/karasawa/example_karasawa.py.
"""
from alproj.surface import get_colored_surface
from alproj.project import sim_image, reverse_proj, to_geotiff
from alproj.gcp import image_match, set_gcp, filter_gcp_distance
from alproj.optimize import CMAOptimizer
import rasterio
import cv2
import json

# Step 1: Load data
# ==============================================================================
res = 1.0  # resolution in m
airborne = rasterio.open("devel_data/tateyama/airborne.tif")
dsm = rasterio.open("devel_data/tateyama/dsm.tif")
target_image_path = "devel_data/tateyama/target_image.jpg"

# Step 2: Simulate initial image (pinhole, no distortion)
# ==============================================================================
# Pinhole parameters: uses a1, a2 (aspect ratio), k1-k6 (radial),
# p1, p2 (tangential), s1-s4 (prism) distortion coefficients.
params_init = {
    "x": 732731, "y": 4051171, "z": 2458,
    "fov": 75, "pan": 95, "tilt": 0, "roll": 0,
    "a1": 1, "a2": 1,
    "k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0, "k6": 0,
    "p1": 0, "p2": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0,
    "w": 5616, "h": 3744, "cx": 5616 / 2, "cy": 3744 / 2,
}

# Generate colored surface
vert, col, ind, offsets = get_colored_surface(
    airborne, dsm, shooting_point=params_init, distance=4000, res=res)
# Simulate image (mask closer than 100m to prevent mismatch)
sim = sim_image(vert, col, ind, params_init, offsets, min_distance=100)
cv2.imwrite("devel_data/tateyama/sim_init.png", sim)
# Reverse projection
df = reverse_proj(sim, vert, ind, params_init, offsets)

# Step 3: Optimize camera parameters (Phase 1: position and orientation)
# ==============================================================================
match, plot = image_match(
    target_image_path, "devel_data/tateyama/sim_init.png",
    method="ufm", plot_result=True, outlier_filter="fundamental",
    params=params_init, resize=800, threshold=30.0,
    spatial_thin_grid=100, spatial_thin_selection="center")

cv2.imwrite("devel_data/tateyama/matched_1st.png", plot)

gcps = set_gcp(match, df)
gcps = filter_gcp_distance(gcps, params_init, min_distance=50)

cma_optimizer = CMAOptimizer(gcps[["x", "y", "z"]], gcps[["u", "v"]], params_init)
cma_optimizer.set_target(["x", "y", "z", "fov", "pan", "tilt", "roll", "a1", "a2", "cx", "cy"])
params_2nd, error = cma_optimizer.optimize(
    generation=300, sigma=1.0, population_size=50, f_scale=10.0)
print("Phase 1 error:", error)

sim2 = sim_image(vert, col, ind, params_2nd, offsets, min_distance=100)
cv2.imwrite("devel_data/tateyama/sim_2nd.png", sim2)
df2 = reverse_proj(sim2, vert, ind, params_2nd, offsets)

# Step 4: Optimize pinhole distortion (Phase 2)
# ==============================================================================
match, plot = image_match(
    target_image_path, "devel_data/tateyama/sim_2nd.png",
    method="ufm", plot_result=True, outlier_filter="essential",
    params=params_2nd, resize=800, threshold=30.0,
    spatial_thin_grid=50, spatial_thin_selection="center")

cv2.imwrite("devel_data/tateyama/matched_2nd.png", plot)
gcps = set_gcp(match, df2)
gcps = filter_gcp_distance(gcps, params_2nd, min_distance=100)

cma_optimizer = CMAOptimizer(gcps[["x", "y", "z"]], gcps[["u", "v"]], params_2nd)
cma_optimizer.set_target(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4", "cx", "cy", "a1", "a2"])
params_optim, error = cma_optimizer.optimize(
    generation=300, sigma=1.0, population_size=50, f_scale=10.0)
print("Phase 2 error:", error)

# Or use least squares optimizer (much faster but might be less accurate)
# lsq_optimizer = LsqOptimizer(gcps[["x","y","z"]], gcps[["u","v"]], params_2nd)
# lsq_optimizer.set_target(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"])
# params_optim, error = lsq_optimizer.optimize(method="trf", max_nfev=1000)
# print("Error:", error)

# Save optimized parameters
with open("devel_data/tateyama/optimized_params.json", "w") as f:
    json.dump(params_optim, f, indent=4)

# Load optimized parameters
with open("devel_data/tateyama/optimized_params.json", "r") as f:
    params_optim = json.load(f)

# Simulate optimized image
sim_optimized = sim_image(vert, col, ind, params_optim, offsets)
cv2.imwrite("devel_data/tateyama/sim_optimized.png", sim_optimized)

# Step 5: Generate georectified image
# ==============================================================================
original = cv2.imread(target_image_path)
georectified = reverse_proj(original, vert, ind, params_optim, offsets)

to_geotiff(
    georectified,
    "devel_data/tateyama/georectified.tif",
    resolution=1.0,
    crs="EPSG:6690",
    bands=["R", "G", "B"],
    interpolate=True,
    max_dist=1.0,
    agg_func="mean",
    nodata=255,
)
