from alproj.surface import get_colored_surface
from alproj.project import sim_image, reverse_proj
from alproj.gcp import image_match, set_gcp, filter_gcp_distance
from alproj.optimize import CMAOptimizer
from alproj.project import to_geotiff
import rasterio
import cv2

# Step1: Load data
## ------------------------------------------------------------------------------
res = 1.0 # resolution in m
airborne = rasterio.open("devel_data/airborne.tif")
dsm = rasterio.open("devel_data/dsm.tif")
target_image_path = "devel_data/target_image.jpg"

# Step2: Simulate initial image
## ------------------------------------------------------------------------------
## Define camera parameters 
params_init = {"x":732731,"y":4051171, "z":2458, "fov":75, "pan":95, "tilt":0, "roll":0,\
     "a1":1, "a2":1, "k1":0, "k2":0.0, "k3":0, "k4":0, "k5":0.0, "k6":0, \
         "p1":0, "p2":0, "s1":0, "s2":0, "s3":0, "s4":0, \
         "w":5616, "h":3744, "cx":5616/2, "cy":3744/2}

## Generate colored surface
vert, col, ind, offsets = get_colored_surface(
    airborne, dsm, shooting_point=params_init, distance=2000, res=res) # This takes some minites.
## Simulate image
sim = sim_image(vert, col, ind, params_init, offsets, min_distance=100) # mask closer area than 100m for preventing missmatch in image matching
cv2.imwrite("devel_data/sim_init.png", sim)
## Reverse projection
df = reverse_proj(sim, vert, ind, params_init, offsets)

# Step3: Optimize camera parameters (Phase 1 and 2)
## ------------------------------------------------------------------------------
## Image matching between the original and simulated images
## Use spatial_thin_grid to ensure uniform distribution of matches
match, plot = image_match(
    target_image_path, "devel_data/sim_init.png",
    method="minima-roma", plot_result=True, outlier_filter="fundamental", params=params_init, resize=800, threshold=30.0,
    spatial_thin_grid=100, spatial_thin_selection="center")

cv2.imwrite("devel_data/matched_1st.png", plot)

## Set ground control points
gcps = set_gcp(match, df)

## Filter GCPs by distance (exclude points closer than 50m)
gcps = filter_gcp_distance(gcps, params_init, min_distance=50)

## Optimize camera parameters (Phase 1: adjust position, orientation, fov, and aspect ratio)
cma_optimizer = CMAOptimizer(gcps[["x","y","z"]], gcps[["u","v"]], params_init)
cma_optimizer.set_target(["x", "y", "z", "fov", "pan", "tilt", "roll", "a1", "a2"])
params_2nd, error = cma_optimizer.optimize(
    generation = 300, sigma = 1.0, population_size=50, f_scale=10.0)
print("Error:", error)

sim2 = sim_image(vert, col, ind, params_2nd, offsets, min_distance=100)
cv2.imwrite("devel_data/sim_2nd.png", sim2)
df2 = reverse_proj(sim2, vert, ind, params_2nd, offsets)

## Optimize camera parameters (Phase 2: adjust distortion parameters)
### ------------------------------------------------------------------------------
match, plot = image_match(
    target_image_path, "devel_data/sim_2nd.png",
    method="minima-roma", plot_result=True, outlier_filter="essential", params=params_2nd, resize=800, threshold=30.0,
    spatial_thin_grid=50, spatial_thin_selection="center")

cv2.imwrite("devel_data/matched_2nd.png", plot)
gcps = set_gcp(match, df2)

## Filter GCPs by distance (exclude points closer than 100m)
gcps = filter_gcp_distance(gcps, params_2nd, min_distance=100)

# Optimize camera parameters
cma_optimizer = CMAOptimizer(gcps[["x","y","z"]], gcps[["u","v"]], params_2nd)
cma_optimizer.set_target(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"])
params_optim, error = cma_optimizer.optimize(
    generation = 300, sigma = 1.0, population_size=50, f_scale=10.0)
print("Error:", error)

# Or use least squares optimizer (much faster but might be less accurate)
# lsq_optimizer = LsqOptimizer(gcps[["x","y","z"]], gcps[["u","v"]], params_2nd)
# lsq_optimizer.set_target(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"])
# params_optim, error = lsq_optimizer.optimize(method="trf", max_nfev=1000)
# print("Error:", error)

# Save optimized parameters
import json
with open("devel_data/optimized_params.json", "w") as f:
    json.dump(params_optim, f, indent=4)

# Load optimized parameters
with open("devel_data/optimized_params.json", "r") as f:
    params_optim = json.load(f)

# Simulate optimized image
sim_optimized = sim_image(vert, col, ind, params_optim, offsets)
cv2.imwrite("devel_data/sim_optimized.png", sim_optimized)

# Step4: Generate georectified image
## ------------------------------------------------------------------------------
original = cv2.imread(target_image_path)
georectified = reverse_proj(original, vert, ind, params_optim, offsets)

# Convert to GeoTIFF with automatic rasterization and interpolation
to_geotiff(
    georectified,
    "devel_data/georectified.tif",
    resolution=1.0,           # Pixel resolution in coordinate units (e.g., meters)
    crs="EPSG:6690",          # Coordinate Reference System
    bands=["R", "G", "B"],    # Which columns to use as bands
    interpolate=True,         # Fill small gaps using focal statistics
    max_dist=1.0,             # Maximum interpolation distance
    agg_func="mean",          # Aggregation function: "mean", "median", "max", "min"
    nodata=255                # NoData value for missing pixels
)