import numpy as np
from math import pi, sin, cos, tan
import pandas as pd
from cmaes import CMA
from tqdm import tqdm

def intrinsic_mat(fov_x_deg, w, h, cx=None, cy=None):
    if cx == None:
        cx = w/2
    if cy == None:
        cy = h/2
    fov_x = fov_x_deg * pi / 180
    fov_y = fov_x * h / w
    fx = w / (2 * tan(fov_x/2))
    fy = h / (2 * tan(fov_y/2))
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return mat

def extrinsic_mat(pan_deg, tilt_deg, roll_deg, t_x, t_y, t_z):
    pan = pan_deg * pi / 180
    tilt = -(tilt_deg + 90) * pi / 180
    roll = -roll_deg * pi / 180
    rmat_z = np.array([
        [cos(pan), -sin(pan), 0],
        [sin(pan), cos(pan), 0],
        [0, 0, 1]
    ])
    rmat_x = np.array([
        [1, 0, 0],
        [0, cos(tilt), -sin(tilt)],
        [0, sin(tilt), cos(tilt)]
    ])
    rmat_y = np.array([
        [cos(roll), 0, sin(roll)],
        [0, 1, 0],
        [-sin(roll), 0, cos(roll)]
    ])
    rmat = np.dot(np.dot(rmat_x, rmat_y), rmat_z)
    tmat = np.array([
        [-t_x],
        [-t_y],
        [-t_z]
    ]) 
    tmat = np.dot(rmat, tmat)
    return np.vstack((np.hstack((rmat, tmat)), np.array([0,0,0,1])))

def distort(points, a1, a2, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4):
    x_norm = points[0,]/points[2,]
    y_norm = points[1,]/points[2,]
    r2 = (x_norm**2 + y_norm**2)
    r4 = r2**2
    r6 = r2*r4
    x_distort = (x_norm*((1+k1*r2+k2*r4+k3*r6) / (1+k4*r2+k5*r4+k6*r6)) \
        - 2*p1*x_norm*y_norm - p2*(r2+2*x_norm**2) - s1*r2 - s2*r4) * points[2,]
    y_distort = (y_norm*((1+a1+k1*r2+k2*r4+k3*r6) / (1+a2+k4*r2+k5*r4+k6*r6)) \
        - 2*p2*x_norm*y_norm - p1*(r2+2*y_norm**2) - s3*r2 - s4*r4) * points[2,]
    points_distort = np.vstack([x_distort, y_distort, points[2,:]])
    return points_distort

def project(obj_points, params):
    op = np.vstack((obj_points.to_numpy().T, np.ones([1,len(obj_points)])))
    imat = intrinsic_mat(params["fov"], params["w"], params["h"], params["cx"], params["cy"])
    emat = extrinsic_mat(params["pan"], params["tilt"], params["roll"], params["x"], params["y"], params["z"])
    op_cc = np.dot(emat, op) # Object points in camera coordinate system
    op_cc2 = distort(op_cc, params["a1"], params["a2"], params["k1"], params["k2"], params["k3"], params["k4"], \
        params["k5"], params["k6"], params["p1"], params["p2"], params["s1"], params["s2"], params["s3"], params["s4"])
    op_ic = np.dot(imat, op_cc2)
    uv = np.array([
        params["w"] - op_ic[0,:]/op_ic[2,:],
        op_ic[1,:]/op_ic[2,:]
    ])
    uv = pd.DataFrame(uv.T, columns = ["u", "v"])
    return uv

def rmse(img_points, projected):
    img_points = img_points.to_numpy()
    projected = projected.to_numpy()
    dist = ((img_points[:,0] - projected[:,0])**2 + (img_points[:,1] - projected[:,1])**2)**0.5
    rmse = np.mean(dist)
    return rmse

class CMAOptimizer():
    def __init__(self, obj_points, img_points, params_init):
        self.obj_points = obj_points
        self.img_points = img_points
        self.params_init = params_init

    def set_target(self, target_params = ["fov", "pan", "tilt", "roll", "a1", "a2", "k1", "k2", "k3", \
            "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"]):
        p = self.params_init
        t = target_params
        self.target_params = target_params
        self.target_params_init = np.array([p[ti] for ti in t])

    def _loss_function(self):
        params = self.params_init.copy()
        for t in self.target_params:
            params.pop(t)
        def _proj_error(values): # parameter values to optimize, as np.array
            params.update(dict(zip(self.target_params, values)))
            projected = project(self.obj_points, params)
            loss = rmse(self.img_points, projected)
            return loss
        return _proj_error
    
    def optimize(self, sigma=1.0, bounds=None, generation=500, population_size=50, n_max_resampling = 100):
        loss_function = self._loss_function()
        optimizer = CMA(mean=self.target_params_init.astype("float64"), sigma=float(sigma), bounds=bounds, population_size=population_size, n_max_resampling=n_max_resampling)
        for _ in tqdm(range(generation)):
            solutions = []
            for _ in range(population_size):
                x = optimizer.ask()
                value = loss_function(x)
                solutions.append((x, value))
            optimizer.tell(solutions)
        error = solutions[0][1]
        params = self.params_init.copy()
        for t in self.target_params:
            params.pop(t)
        params.update(dict(zip(self.target_params, solutions[0][0])))
        return params, error
        
