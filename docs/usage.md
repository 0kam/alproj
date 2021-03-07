# Usage
Here I show an example of geo-rectification process using a photograph of [NIES' long-term monitoring]((https://db.cger.nies.go.jp/gem/ja/mountain/station.html?id=2)) taken at Tateyama Murodo-sanso, Toyama prefecture, Japan.

```python
# Loading requirements
from alproj.surface import create_db, crop
from alproj.project import sim_image, reverse_proj
from alproj.gcp import akaze_match, set_gcp
from alproj.optimize import CMAOptimizer
import sqlite3
import numpy as np
import rasterio
```
## Data Preparation
You should prepare below before starting.
- The target landscape photograph.
![](_static/ttym_2016.jpg)
- An orthorectificated airborne photograph.
![](_static/airborne.png)
- A Digital Surface Model.
![](_static/dem.png)

Both the airborne photograph and the DSM must cover hole the area where the target photograph covers. 
And both of them must be in a same planar Coordinate Reference System, e.g. Universal Transverse Mercator Coordinate System (UTM).

## Setting up Pointcloud Database
First, you should prepare a pointcloud database using the airborne photograph and the DSM.
```python
res = 1.0 # Resolution in m
aerial = rasterio.open("airborne.tif")
dsm = rasterio.open("dsm.tif")
out_path = "pointcloud.db" # Output database

create_db(aerial, dsm, out_path) # This takes some minutes
```
Then, an SQLite database that has two components like below
- vertices
```
# x, y and z stands for geographic coordinates of each point.
# r, g and b stands for the color of each point.
id x               y               z               r   g   b
0  7.34942032e+05, 2.54030493e+03, 4.05319697e+06, 96, 91, 82 
1       ...             ...              ...           ...
...
```
- indices
```
# Each row stands for which three points form a triangle.
v1       v2       v2
0        3        4
0        4        1
        ...   
7877845  7878552  7877846
```

## Define Initial Camera Parameters
Setting initial camera parameters for optmization.
Note that `alproj` does NOT support the estimation of camera location now.
- x, y, z  
A shooting point coordinate in the CRS of pointcloud database.
- fov  
  A Field of View in degree.
- pan, tilt, roll  
  A set of euler angle of the camera in degree.
- a1 ~ s4  
  Distortion coefficients. See [Algorithm](https://alproj.readthedocs.io/en/latest/overview.html#algorithm) for detail.
- w, h  
  The width and height of the target image in pixel.
- cx, cy  
  A coordinate of the principal point in pixel.
 
```python
params = {"x":732731,"y":4051171, "z":2458, "fov":70, "pan":100, "tilt":0, "roll":0,\
          "a1":1, "a2":1, "k1":0, "k2":0, "k3":0, "k4":0, "k5":0, "k6":0, \
          "p1":0, "p2":0, "s1":0, "s2":0, "s3":0, "s4":0, \
          "w":5616, "h":3744, "cx":5616/2, "cy":3744/2}
```

## Rendering a Simulated Landscape Image
To find a set of Ground Control Points, render a simulated landscape image with the pointcloud database and the initial camera parameters.

First, crop the pointcroud in fan shape
```python
conn = sqlite3.connect("pointcloud.db")

distance = 3000 # The radius of the fan shape
chunksize = 1000000

vert, col, ind = crop(conn, params, distance, chunksize) # This takes some minites.
```
Then you'll get three `np.array`s looks like below.
- vert  
  Vertex coordinates of each point. In x, z, y order.
  ```
  >>> vert
  array([[7.34942032e+05, 2.54030493e+03, 4.05319697e+06],
       [7.34943032e+05, 2.53846924e+03, 4.05319697e+06],
       [7.34941032e+05, 2.54056641e+03, 4.05319597e+06],
       ...,
       [7.34174032e+05, 2.15709058e+03, 4.04854197e+06],
       [7.34175032e+05, 2.15659692e+03, 4.04854197e+06],
       [7.34176032e+05, 2.15609204e+03, 4.04854197e+06]])
  ```
- col  
  Vertex colors in 0 to 1.
  ```
  >>> col
  array([[0.37647059, 0.35686275, 0.32156863],
       [0.36078431, 0.33333333, 0.30980392],
       [0.42352941, 0.40392157, 0.36078431],
       ...,
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ]])
  ```
- ind  
  The index that shows which three points form a triangle.
  ```
  >>> ind
  array([[      0,       3,       4],
       [      0,       4,       1],
       [      1,       4,       5],
       ...,
       [7877844, 7878551, 7877845],
       [7877845, 7878551, 7878552],
       [7877845, 7878552, 7877846]], dtype=int64)
    ```
Next, render a simulated landscape image.
```python
import cv2
sim = sim_image(vert, col, ind, params)
cv2.imwrite("devel_data/initial.png", sim)
```
![](_static/initial.png)

Every pixel in this image has geographic coordinates. Then you can get a table of image coordinates and geographic coordinates of it.
```python
df = reverse_proj(sim, vert, ind, params)
```
```
>>> df
             u     v            x           y            z      B      G      R
2058832   3376   366  734200.3125  4050691.75  2988.827881  116.0  120.0  124.0
2058833   3377   366  734199.6875  4050691.75  2988.624268  106.0  110.0  113.0
2058834   3378   366  734198.7500  4050691.25  2988.337402   82.0   86.0   88.0
2058835   3379   366  734198.0000  4050691.25  2988.081543   70.0   75.0   78.0
2058836   3380   366  734197.3750  4050691.25  2987.862061   60.0   65.0   68.0
...        ...   ...          ...         ...          ...    ...    ...    ...
21026299  5611  3743  732740.3125  4051161.75  2453.355469  113.0  117.0  148.0
21026300  5612  3743  732740.3125  4051161.75  2453.355469  113.0  117.0  148.0
21026301  5613  3743  732740.3125  4051161.75  2453.355713  113.0  117.0  148.0
21026302  5614  3743  732740.3125  4051161.75  2453.355713  113.0  117.0  148.0
21026303  5615  3743  732740.3125  4051161.75  2453.355713  113.0  117.0  148.0

[17336750 rows x 8 columns]
```

## Finding Ground Contorol Points
Then, you can add some Ground Contorol Points (GCPs) in the target image by matching target image and simulated image.
[AKAZE](https://docs.opencv.org/3.4/d0/de3/citelist.html#CITEREF_ANB13) local features and [FLANN](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#flann-based-matcher) matcher is available.

```python
path_org = "ttym_2016.jpg"
path_sim = "init.png"

match, plot = akaze_match(path_org, path_sim, ransac_th=200, plot_result=True)
cv2.imwrite("matched.png", plot)
gcps = set_gcp(match, df)
```
Matching result.
![](_static/matched.png)
```
>>> gcps
        u     v            x           y            z
0    2585  1127  733720.2500  4051094.25  2648.573486
1    3566   631  734078.1250  4050727.00  2912.292969
2    3502   689  733951.8750  4050792.75  2849.661865
3    3745   723  733976.0625  4050697.25  2848.271729
4    3833   766  733996.1250  4050657.25  2841.155518
..    ...   ...          ...         ...          ...
147  3355  1126  733688.0625  4050916.50  2648.639893
148  4618  1190  733593.2500  4050619.00  2622.293457
149  2195  1243  733770.3750  4051216.00  2626.165527
150  2533  1777  733437.5625  4051142.25  2474.067383
151  3351  1072  733726.8750  4050907.00  2668.884766
```
Where `u` and `v` stands for the x and y axis coordinates in the image coordinate system.

## Optimization of Camera Parameters
Finally, optimizing camera parameters using GCPs.
Camera parameters are optimized by minimizing [reproection errors](https://support.pix4d.com/hc/en-us/articles/202559369-Reprojection-error) with a [CMA-ES](https://github.com/CyberAgent/cmaes) optimizer.  
You can specify which parameters to be optimized.

```python
obj_points = gcps[["x","y","z"]] # Object points in a geographic coordinate system
img_points = gcps[["u","v"]] # Image points in an image coordinate system 
params_init = params # Initial parameters
target_params = ["fov", "pan", "tilt", "roll", "a1", "a2", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"] # Parameters to be optimized
cma_optimizer = CMAOptimizer(obj_points, img_points, params_init) # Create an optimizer instance.
cma_optimizer.set_target(target_params)
params_optim, error = cma_optimizer.optimize(generation = 500, bounds = None, sigma = 1.0, population_size=50)
```