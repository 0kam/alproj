import numpy as np
import warnings
from math import pi, sin, cos, tan, atan
import pandas as pd
from cmaes import CMA
from scipy.optimize import least_squares
from tqdm import tqdm

def intrinsic_mat(fov_x_deg, w, h, cx=None, cy=None):
    """ 
    Makes an intrinsic camera matrix (in OpenCV style) from given parameters.
    See https://learnopencv.com/geometry-of-image-formation/ .

    Parameters
    ----------
    fov_x_deg : float
        Field of View in degree.
    w : int
        Width of the image
    h : int
        Height of the image
    cx : float default None
        X-coordinate of the principal point. If None, cx = w/2.
    cy : float default None
        Y-coordinate of the principal point. If None, cy = h/2.
    
    Returns
    -------
    mat : numpy.ndarray
        Intrinsic matrix.
    """
    if cx == None:
        cx = w/2
    if cy == None:
        cy = h/2
    fov_x = fov_x_deg * pi / 180
    fov_y = 2 * atan(tan(fov_x / 2) * h / w)
    fx = w / (2 * tan(fov_x/2))
    fy = h / (2 * tan(fov_y/2))
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return mat

def extrinsic_mat(pan_deg, tilt_deg, roll_deg, t_x, t_y, t_z):
    """
    Makes an extrinsic camera matrix from given parameters.
    See https://learnopencv.com/geometry-of-image-formation/ .

    Parameters
    ----------
    pan_deg : float
        Pan angle in degree.
    tilt_deg : float
        Tilt angle in degree.
    roll_deg : float
        Roll angle in degree.
    t_x : float
        X-axis coordinate of the camera location.
    t_y : float
        Y-axis coordinate of the camera location.
    t_z : float
        Z-axis coordinate of the camera location.
    
    Returns
    -------
    mat : numpy.ndarray
        Extrinsic camera matrix.
    """
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

def _distort(points, w, h, a1, a2, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4,
             cx=None, cy=None):
    """
    Apply lens distortion to 2D points using the camera's distortion model.

    For detailed parameter descriptions, see alproj.project.persp_proj().

    Parameters
    ----------
    cx : float, default None
        X-coordinate of the principal point used as distortion center.
        If None, defaults to (w - 1) / 2 for backward compatibility.
    cy : float, default None
        Y-coordinate of the principal point used as distortion center.
        If None, defaults to (h - 1) / 2 for backward compatibility.
    """
    if cx is None:
        cx = (w - 1) / 2
    if cy is None:
        cy = (h - 1) / 2
    centre = np.array([cx, cy], dtype='float32')
    x1 = (points[:,0] - centre[0]) / centre[0]
    y1 = (points[:,1] - centre[1]) / centre[1]
    r = (x1**2 + y1**2)**0.5
    r2 = r**2
    r4 = r**4
    r6 = r**6

    x1_d = x1 * (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6) + \
        2*p1*x1*y1 + p2*(r2 + 2*x1**2) + \
        s1*r2 + s2*r4
    y1_d = y1 * (1 + a1 + k1*r2 + k2*r4 + k3*r6) / (1 + a2 + k4*r2 + k5*r4 + k6*r6) + \
        p1*(r2 + 2*y1**2) + 2*p2*x1*y1 + s3*r2 + s4*r4
    x1_d = x1_d * centre[0] + centre[0]
    y1_d = y1_d * centre[1] + centre[1]
    pts_d = np.stack([x1_d, y1_d], axis = 0).T
    return pts_d

def _fisheye_project(obj_points, params):
    """
    Equidistant fisheye projection: 3D world coordinates -> 2D image coordinates.

    Uses the front angle convention where theta = -arctan2(r_cam, -Z) and
    image radius r = f * theta_d with f = w / fov_rad (equidistant focal length).

    Parameters
    ----------
    obj_points : pandas.DataFrame
        Coordinates of the points in 3D geographic coordinate system.
        The column names must be x, y, z.
    params : dict
        Camera parameters. Required keys:
        - x, y, z: Camera position
        - fov, pan, tilt, roll: Orientation
        - w, h: Image dimensions
        - cx, cy: Principal point
        - k1, k2, k3, k4: Angle-space distortion coefficients (default 0)
        - a1, a2: Aspect ratio correction (default 1.0). Scales horizontal
          and vertical focal lengths independently.
        - p1, p2: Tangential (decentering) distortion in image space (default 0)

    Returns
    -------
    uv : pandas.DataFrame
        2D projected coordinates with columns u, v.
    """
    if params["fov"] <= 0:
        raise ValueError(f"fov must be positive, got {params['fov']}")

    obj = obj_points[["x", "y", "z"]]
    op = np.vstack((obj.to_numpy().T, np.ones([1, len(obj)])))

    emat = extrinsic_mat(
        params["pan"], params["tilt"], params["roll"],
        params["x"], params["y"], params["z"])

    cam = np.dot(emat, op)[:3, :]
    X, Y, Z = cam[0], cam[1], cam[2]

    r_cam = np.sqrt(X**2 + Y**2)
    theta = -np.arctan2(r_cam, -Z)

    k1 = params.get("k1", 0.0)
    k2 = params.get("k2", 0.0)
    k3 = params.get("k3", 0.0)
    k4 = params.get("k4", 0.0)
    t2 = theta**2
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)

    fov_rad = params["fov"] * pi / 180.0
    f = params["w"] / fov_rad

    # Aspect ratio correction
    a1 = params.get("a1", 1.0)
    a2 = params.get("a2", 1.0)
    if abs(a1) < 1e-10 or abs(a2) < 1e-10:
        raise ValueError(f"a1 and a2 must be non-zero, got a1={a1}, a2={a2}")

    safe_r = np.where(r_cam > 1e-10, r_cam, 1.0)
    r_img = f * theta_d
    u = params["w"] - (r_img * (X / safe_r) * a1 + params["cx"])
    v = r_img * (Y / safe_r) * a2 + params["cy"]

    on_axis = r_cam < 1e-10
    u = np.where(on_axis, params["w"] - params["cx"], u)
    v = np.where(on_axis, params["cy"], v)

    # Tangential (decentering) distortion in image space
    p1 = params.get("p1", 0.0)
    p2 = params.get("p2", 0.0)
    if p1 != 0.0 or p2 != 0.0:
        w_half = params["w"] / 2
        h_half = params["h"] / 2
        x_n = (u - (params["w"] - params["cx"])) / w_half
        y_n = (v - params["cy"]) / h_half
        r2 = x_n**2 + y_n**2
        du = 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
        dv = p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n
        u = u + du * w_half
        v = v + dv * h_half

    return pd.DataFrame({"u": u, "v": v})


def _pinhole_project(obj_points, params):
    """
    Pinhole projection: 3D world coordinates -> 2D image coordinates.

    Uses intrinsic/extrinsic matrices and the pinhole distortion model.

    Parameters
    ----------
    obj_points : pandas.DataFrame
        Coordinates of the points in 3D geographic coordinate system.
        The column names must be x, y, z.
    params : dict
        Camera parameters. For detailed parameter descriptions, see alproj.project.persp_proj().

    Returns
    -------
    uv : pandas.DataFrame
        2D projected coordinates with columns u, v.
    """
    obj_points = obj_points[["x", "y", "z"]]
    op = np.vstack((obj_points.to_numpy().T, np.ones([1,len(obj_points)])))
    imat = intrinsic_mat(params["fov"], params["w"], params["h"], params["cx"], params["cy"])
    emat = extrinsic_mat(params["pan"], params["tilt"], params["roll"], params["x"], params["y"], params["z"])
    op_cc = np.dot(emat, op) # Object points in camera coordinate system
    op_ic = np.dot(imat, op_cc[:3,:]) # Object points in image coordinate system
    uv = np.array([
        params["w"] - op_ic[0,:]/op_ic[2,:],
        op_ic[1,:]/op_ic[2,:]
    ]).T
    uv_distort = _distort(
        uv, params["w"], params["h"], params["a1"], params["a2"],
        params["k1"], params["k2"], params["k3"], params["k4"], params["k5"], params["k6"],
        params["p1"], params["p2"], params["s1"], params["s2"], params["s3"], params["s4"],
        cx=params["cx"], cy=params["cy"])
    uv_df = pd.DataFrame(uv_distort, columns = ["u", "v"])
    return uv_df


def project(obj_points, params):
    """
    3D to 2D Perspective projection of given points with given camera parameters.

    Parameters
    ----------
    obj_points : pandas.DataFrame
        Coordinates of the points (usually GCPs) in 3D geographic coordinate system.
        The column names must be x, y, z.
    params : dict
        Camera parameters. For detailed parameter descriptions, see alproj.project.persp_proj().
        If params contains ``"model": "fisheye"``, uses equidistant fisheye projection.
        Otherwise uses the default pinhole projection.

    Returns
    -------
    uv : pandas.DataFrame
        2D projected coordinates with columns u, v.
    """
    if params.get("model") == "fisheye":
        return _fisheye_project(obj_points, params)
    return _pinhole_project(obj_points, params)

def mean_reprojection_error(img_points, projected, weights=None):
    """
    Calculate mean reprojection error (mean Euclidean distance) of the projection.

    Parameters
    ----------
    img_points : pandas.DataFrame
        Observed image coordinates with columns u, v.
    projected : pandas.DataFrame
        Projected image coordinates with columns u, v.
    weights : numpy.ndarray, default None
        Per-point weights. If provided, returns weighted average.

    Returns
    -------
    error : float
        Mean reprojection error in pixels.
    """
    img_arr = img_points[["u", "v"]].to_numpy()
    proj_arr = projected[["u", "v"]].to_numpy()
    dist = np.sqrt((img_arr[:, 0] - proj_arr[:, 0])**2 +
                   (img_arr[:, 1] - proj_arr[:, 1])**2)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(dist):
            raise ValueError(
                f"weights length ({len(weights)}) != number of points ({len(dist)})")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        return np.average(dist, weights=weights)
    return np.mean(dist)


def rmse(img_points, projected, weights=None):
    """
    Deprecated alias for :func:`mean_reprojection_error`.

    .. deprecated:: 1.2.0
        Use :func:`mean_reprojection_error` instead. This function computes
        mean Euclidean distance, not root-mean-square error.
    """
    warnings.warn(
        "rmse() is deprecated, use mean_reprojection_error() instead. "
        "This function computes mean Euclidean distance, not RMSE.",
        DeprecationWarning, stacklevel=2)
    return mean_reprojection_error(img_points, projected, weights)


def huber_loss(img_points, projected, f_scale=10.0, weights=None):
    """
    Calculate Huber loss for robust optimization.

    Huber loss is less sensitive to outliers than squared error loss.
    For residuals below f_scale, it behaves like L2 (squared) loss.
    For residuals above f_scale, it behaves like L1 (linear) loss.

    Parameters
    ----------
    img_points : pandas.DataFrame
        Observed image coordinates with columns u, v.
    projected : pandas.DataFrame
        Projected image coordinates with columns u, v.
    f_scale : float default 10.0
        Threshold in pixels. Residuals below f_scale use L2, above use L1.
    weights : numpy.ndarray, default None
        Per-point weights. If provided, returns weighted average.

    Returns
    -------
    loss : float
        Mean Huber loss value.
    """
    img_arr = img_points[["u", "v"]].to_numpy()
    proj_arr = projected[["u", "v"]].to_numpy()
    residuals = np.sqrt((img_arr[:, 0] - proj_arr[:, 0])**2 +
                        (img_arr[:, 1] - proj_arr[:, 1])**2)
    loss = np.where(
        residuals <= f_scale,
        0.5 * residuals**2,
        f_scale * (residuals - 0.5 * f_scale)
    )
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(loss):
            raise ValueError(
                f"weights length ({len(weights)}) != number of points ({len(loss)})")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        return np.average(loss, weights=weights)
    return np.mean(loss)


def compute_residuals(obj_points, img_points, params):
    """
    Compute residual vector for least_squares optimization.

    Parameters
    ----------
    obj_points : pandas.DataFrame
        Geographic coordinates of the Ground Control Points.
    img_points : pandas.DataFrame
        Image coordinates of the Ground Control Points.
    params : dict
        Camera parameters.

    Returns
    -------
    residuals : numpy.ndarray
        Flattened residual vector (observed - projected).
    """
    projected = project(obj_points, params)
    img_arr = img_points[["u", "v"]].to_numpy()
    proj_arr = projected.to_numpy()
    residuals = (img_arr - proj_arr).flatten()
    return residuals


DEFAULT_BOUND_WIDTHS = {
    "fov": 45, "pan": 45, "tilt": 45, "roll": 45,
    "x": 30, "y": 30, "z": 30,
    "cx": 50, "cy": 50,
    "a1": 0.2, "a2": 0.2,
    "k1": 0.2, "k2": 0.2, "k3": 0.2, "k4": 0.2, "k5": 0.2, "k6": 0.2,
    "p1": 0.2, "p2": 0.2,
    "s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2,
}

def bounds_to_array(params_init, target_params, bound_widths=None):
    """
    Convert bound widths (dict) to numpy array for CMA-ES optimizer.

    Parameters
    ----------
    params_init : dict
        Initial values of camera parameters.
    target_params : list
        Parameters to be optimized.
    bound_widths : dict default None
        Width from initial value for each parameter (e.g., {"fov": 45, "x": 30}).
        If None or parameter not specified, uses DEFAULT_BOUND_WIDTHS.

    Returns
    -------
    bounds : numpy.ndarray
        Bounds array with shape (len(target_params), 2).
    """
    if bound_widths is None:
        bound_widths = {}

    bounds = np.zeros((len(target_params), 2))
    for i, key in enumerate(target_params):
        value = params_init[key]
        width = bound_widths.get(key, DEFAULT_BOUND_WIDTHS.get(key, 0.2))
        bounds[i, :] = np.array([value - width, value + width])
    return bounds


class BaseOptimizer:
    """
    Base class for camera parameter optimizers.

    Attributes
    ----------
    obj_points : pandas.DataFrame
        Geographic coordinates of the Ground Control Points.
        The column names must be x, y, z. See alproj.gcp.set_gcp()
    img_points : pandas.DataFrame
        Image coordinates of the Ground Control Points.
        The column names must be u, v. See alproj.gcp.set_gcp()
    params_init : dict
        Initial values of camera parameters. Must contain:
        - x, y, z: Camera position in projected CRS (e.g., UTM in meters)
        - fov: Field of view in degrees
        - pan, tilt, roll: Orientation angles in degrees
        - a1, a2, k1-k6, p1, p2, s1-s4: Distortion coefficients
        - w, h: Image width and height in pixels
        - cx, cy: Principal point coordinates
    """

    def __init__(self, obj_points, img_points, params_init):
        self.obj_points = obj_points
        self.img_points = img_points
        self.params_init = params_init

    def set_target(self, target_params=["fov", "pan", "tilt", "roll", "a1", "a2", "k1", "k2", "k3",
                                         "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"]):
        """
        Set which parameters to be optimized.

        Parameters
        ----------
        target_params : list default ["fov", "pan", "tilt", "roll", "a1", "a2", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"]
            Parameters to be optimized. You can also include x, y, and z for camera location.
        """
        p = self.params_init
        t = target_params
        self.target_params = target_params
        self.target_params_init = np.array([p[ti] for ti in t])


class CMAOptimizer(BaseOptimizer):
    """
    Camera parameter optimizer using Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    See https://pypi.org/project/cmaes/ .
    You can select which parameters to be optimized, including camera location (x, y, z).
    """

    def _loss_function(self, bounds, f_scale=None, weights=None):
        """
        Create loss function with normalization.

        Parameters
        ----------
        bounds : numpy.ndarray
            Bounds array with shape (len(target_params), 2).
        f_scale : float default None
            Threshold for Huber loss in pixels. If None, uses mean error.
            If specified, uses Huber loss with this threshold.
        weights : numpy.ndarray, default None
            Per-GCP weights for weighted loss computation.
        """
        params = self.params_init.copy()
        for t in self.target_params:
            params.pop(t)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        def _proj_error(normalized_values):
            # Denormalize: [0, 1] -> [lower, upper]
            values = normalized_values * (upper - lower) + lower
            params.update(dict(zip(self.target_params, values)))
            projected = project(self.obj_points, params)
            if f_scale is None:
                loss = mean_reprojection_error(
                    self.img_points, projected, weights)
            else:
                loss = huber_loss(
                    self.img_points, projected, f_scale, weights)
            return loss
        return _proj_error

    def optimize(self, sigma=0.2, bound_widths=None, generation=1000, population_size=10,
                 n_max_resampling=100, f_scale=None, weights=None):
        """
        CMA-optimization of camera parameters.
        See https://github.com/CyberAgent/cmaes/blob/main/cmaes/_cma.py .

        Parameters are normalized to [0, 1] range internally for efficient optimization.

        Parameters
        ----------
        sigma : float default 0.2
            Initial standard deviation of covariance matrix (in normalized [0, 1] space).
        bound_widths : dict default None
            Width from initial value for each parameter (e.g., {"fov": 30, "x": 50}).
            If None or parameter not specified, uses default widths:
            +-45 degrees for fov, pan, tilt, roll; +-30 meters for x, y, z;
            +-0.2 for distortion coefficients.
            Note that large absolute values of distortion coefficients may cause broken projection.
        generation : int default 1000
            Number of generations to run.
        population_size : int default 10
            Population size.
        n_max_resampling : int default 100
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.
        f_scale : float default None
            Threshold for Huber loss in pixels. If None, uses mean reprojection error.
            If specified (e.g., 10.0), uses Huber loss which is more robust to outliers.
        weights : numpy.ndarray, default None
            Per-GCP weights for the loss function. Higher weights increase the
            importance of the corresponding GCPs. Useful for balancing center
            and edge region accuracy.

        Returns
        -------
        params : dict
            Optimized camera parameters.
        error : float
            Mean reprojection error in pixels (unweighted).
        """
        # Compute bounds in original space
        bounds = bounds_to_array(self.params_init, self.target_params, bound_widths)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        # Normalize initial values to [0, 1]
        # Initial value is at center of bounds, so normalized = 0.5
        normalized_init = (self.target_params_init - lower) / (upper - lower)

        # CMA-ES works in normalized [0, 1] space
        normalized_bounds = np.column_stack([np.zeros(len(self.target_params)),
                                              np.ones(len(self.target_params))])

        loss_function = self._loss_function(bounds, f_scale, weights)
        optimizer = CMA(
            mean=normalized_init.astype("float64"),
            sigma=float(sigma),
            bounds=normalized_bounds,
            population_size=population_size,
            n_max_resampling=n_max_resampling
        )

        best_loss = float("inf")
        best_normalized = normalized_init.copy()

        for _ in tqdm(range(generation)):
            solutions = []
            for _ in range(population_size):
                x = optimizer.ask()
                value = loss_function(x)
                solutions.append((x, value))
                if value < best_loss:
                    best_loss = value
                    best_normalized = x.copy()
            optimizer.tell(solutions)

        # Denormalize best solution found across all generations
        best_values = best_normalized * (upper - lower) + lower

        params = self.params_init.copy()
        for t in self.target_params:
            params.pop(t)
        params.update(dict(zip(self.target_params, best_values)))

        # Always return unweighted mean reprojection error for consistency
        projected = project(self.obj_points, params)
        error = mean_reprojection_error(self.img_points, projected)

        return params, error


class LsqOptimizer(BaseOptimizer):
    """
    Camera parameter optimizer using scipy.optimize.least_squares.
    Supports Trust Region Reflective (trf), dogbox, and Levenberg-Marquardt (lm) methods.
    """

    def _residual_function(self, weights=None):
        """
        Create residual function for least_squares.

        Parameters
        ----------
        weights : numpy.ndarray, default None
            Per-GCP weights. Applied as sqrt(weight) to residuals.

        Returns
        -------
        residual_func : callable
            Function that computes residuals given parameter values.
        """
        params = self.params_init.copy()
        for t in self.target_params:
            params.pop(t)

        if weights is not None:
            w = np.repeat(np.sqrt(weights), 2)
        else:
            w = None

        def _residuals(values):
            params.update(dict(zip(self.target_params, values)))
            r = compute_residuals(self.obj_points, self.img_points, params)
            if w is not None:
                r = r * w
            return r

        return _residuals

    def optimize(self, method="trf", bound_widths=None, loss="linear", f_scale=1.0,
                 weights=None, **kwargs):
        """
        Least-squares optimization of camera parameters.

        Parameters
        ----------
        method : str default "trf"
            Algorithm to use: "trf" (Trust Region Reflective), "dogbox", or "lm" (Levenberg-Marquardt).
            Note: "lm" does not support bounds or robust loss functions.
        bound_widths : dict default None
            Width from initial value for each parameter (e.g., {"fov": 30, "x": 50}).
            If None, uses default widths. Ignored when method="lm".
        loss : str default "linear"
            Loss function to use. Options:
            - "linear": Standard least squares (L2 loss).
            - "huber": Huber loss, robust to outliers.
            - "soft_l1": Smooth approximation of L1 loss.
            - "cauchy": Cauchy loss, strongly robust to outliers.
            - "arctan": Arctan loss.
            Note: Only "linear" is supported when method="lm".
        f_scale : float default 1.0
            Soft threshold for inlier residuals (in pixels). Residuals below f_scale
            are treated normally, while those above are down-weighted according to
            the loss function. Has no effect when loss="linear".
            Typical values: 1.0-20.0 pixels depending on expected outlier magnitude.
        weights : numpy.ndarray, default None
            Per-GCP weights for the loss function. Applied as sqrt(weight) scaling
            to the residual vector. Useful for region-balanced optimization.
        **kwargs :
            Additional arguments passed to scipy.optimize.least_squares.

        Returns
        -------
        params : dict
            Optimized camera parameters.
        error : float
            Mean reprojection error in pixels (unweighted).
        """
        if method == "lm" and bound_widths is not None:
            raise ValueError("method='lm' does not support bounds. Set bound_widths=None or use 'trf'/'dogbox'.")
        if method == "lm" and loss != "linear":
            raise ValueError("method='lm' does not support robust loss functions. Use loss='linear' or method='trf'/'dogbox'.")

        residual_func = self._residual_function(weights)

        if method == "lm":
            result = least_squares(
                residual_func,
                self.target_params_init,
                method=method,
                **kwargs
            )
        else:
            bounds = bounds_to_array(self.params_init, self.target_params, bound_widths)
            lower = bounds[:, 0]
            upper = bounds[:, 1]
            result = least_squares(
                residual_func,
                self.target_params_init,
                method=method,
                bounds=(lower, upper),
                loss=loss,
                f_scale=f_scale,
                **kwargs
            )

        best_values = result.x
        params = self.params_init.copy()
        for t in self.target_params:
            params.pop(t)
        params.update(dict(zip(self.target_params, best_values)))

        projected = project(self.obj_points, params)
        error = mean_reprojection_error(self.img_points, projected)

        return params, error
