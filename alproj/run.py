from alproj.pointcloud import create_db, crop
from alproj.project import sim_image, reverse_proj
from alproj.gcp import akaze_match, set_gcp
from alproj.optimize import CMAOptimizer
import sqlite3
import numpy as np
import rasterio

res = 1.0 # resolution in m
aerial = rasterio.open("devel_data/tateyama2.tif")
dsm = rasterio.open("devel_data/tateyamadem_small.tif")
out_path = "devel_data/pc.db"

create_db(aerial, dsm, out_path)

# crop_pointcloud
conn = sqlite3.connect("/home/okamoto/lpmap/devel_data/pc.db")
params = {"x":732731,"y":4051171, "z":2458, "fov":70, "pan":100, "tilt":0, "roll":0,\
     "a1":1, "a2":1, "k1":0, "k2":0, "k3":0, "k4":0, "k5":0, "k6":0, \
         "p1":0, "p2":0, "s1":0, "s2":0, "s3":0, "s4":0, \
         "w":5616, "h":3744, "cx":5616/2, "cy":3744/2}
distance = 3000
chunksize = 1000000

vert, col, ind = crop(conn, params)

# make sim
import cv2
sim = sim_image(vert, col, ind, params)
cv2.imwrite("../devel_data/test.png", sim)

df = reverse_proj(sim, vert, ind, params)
del(vert, col, ind)


# Setting GCPs
path_org = "../devel_data/ttym_2016.jpg"
path_sim = "../devel_data/test.png"

match, plot = akaze_match(path_org, path_sim, ransac_th=200, plot_result=True)
cv2.imwrite("../devel_data/matched.png", plot)
gcps = set_gcp(match, df)

# Optimize camera parameters
obj_points = gcps[["x","y","z"]]
img_points = gcps[["u","v"]]
params_init = params
target_params = ["fov", "pan", "tilt", "roll", "a1", "a2", "k1", "k2", "k3"]#, "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"]

cma_optimizer = CMAOptimizer(obj_points, img_points, params_init)
cma_optimizer.set_target(target_params)
params_optim, error = cma_optimizer.optimize(generation = 1000, bounds = None, sigma = 1.0, population_size=50)

# Use optimized parameters
vert, col, ind = crop(conn, params_optim)
sim2 = sim_image(vert, col, ind, params_optim)
cv2.imwrite("../devel_data/optimized.png", sim2)
