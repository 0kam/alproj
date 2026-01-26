import warnings

import cv2
import numpy as np
import pandas as pd


# Default Lowe's ratio test threshold
_DEFAULT_LOWE_RATIO = 0.7

# List of methods that require the imm package
IMM_METHODS = [
    "sift-lightglue",
    "superpoint-lightglue",
    "minima-superpoint-lightglue",
    "roma",
    "tiny-roma",
    "minima-roma",
    "loftr",
    "minima-loftr",
    "ufm",
    "rdd",
    "master"
]

# Lightweight methods (LightGlue-based) that can handle full-resolution images
_LIGHTGLUE_METHODS = [
    "sift-lightglue",
    "superpoint-lightglue",
    "minima-superpoint-lightglue",
]

# Default resize for memory-intensive methods
_DEFAULT_RESIZE_HEAVY = 640


def _opencv_match(im_org, im_sim, detector_type="akaze", ratio=_DEFAULT_LOWE_RATIO):
    """
    OpenCV feature matching (no Homography RANSAC).

    Parameters
    ----------
    im_org : numpy.ndarray
        Original image (BGR).
    im_sim : numpy.ndarray
        Simulated image (BGR).
    detector_type : str, default "akaze"
        Feature detector type: "akaze" or "sift".
    ratio : float, default 0.7
        Lowe's ratio test threshold.

    Returns
    -------
    pts1, pts2 : numpy.ndarray
        Matched point pairs, shape (N, 2).
    """
    if detector_type.lower() == "akaze":
        detector = cv2.AKAZE_create()
    elif detector_type.lower() == "sift":
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")

    kp1, des1 = detector.detectAndCompute(im_org, None)
    kp2, des2 = detector.detectAndCompute(im_sim, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.array([]).reshape(0, 2).astype("int32"), np.array([]).reshape(0, 2).astype("int32")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) == 0:
        return np.array([]).reshape(0, 2).astype("int32"), np.array([]).reshape(0, 2).astype("int32")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).astype("int32")
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).astype("int32")

    return pts1, pts2


def _imm_match(path_org, path_sim, method, device, resize=None, **kwargs):
    """
    Internal imm-based matching implementation (no Homography RANSAC).

    Parameters
    ----------
    path_org : str
        Path to original image.
    path_sim : str
        Path to simulated image.
    method : str
        IMM matching method name.
    device : str
        Device for computation ("cpu" or "cuda").
    resize : int, optional
        Resize images to this size.
    **kwargs
        Additional arguments for imm's get_matcher.

    Returns
    -------
    pts1, pts2 : numpy.ndarray
        Matched point pairs, shape (N, 2).
    """
    try:
        from imm import get_matcher
    except ImportError:
        raise ImportError(
            f"Method '{method}' requires the 'imm' package. "
            "Install with: pip install alproj[imm]"
        )

    # Get original image sizes for coordinate scaling
    im_org_cv = cv2.imread(path_org)
    im_sim_cv = cv2.imread(path_sim)
    h_org, w_org = im_org_cv.shape[:2]
    h_sim, w_sim = im_sim_cv.shape[:2]

    # Suppress torchvision deprecation warnings about 'pretrained' parameter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        matcher = get_matcher(method, device=device, **kwargs)

    # Use matcher's load_image for proper preprocessing (may resize)
    img0 = matcher.load_image(path_org, resize=resize)
    img1 = matcher.load_image(path_sim, resize=resize)

    # Get the size after loading (to compute scale factors)
    # img shape is (C, H, W) or (1, C, H, W)
    if img0.dim() == 4:
        _, _, h0_loaded, w0_loaded = img0.shape
        _, _, h1_loaded, w1_loaded = img1.shape
    else:
        _, h0_loaded, w0_loaded = img0.shape
        _, h1_loaded, w1_loaded = img1.shape

    result = matcher(img0, img1)
    pts1 = result["matched_kpts0"]
    pts2 = result["matched_kpts1"]

    # Handle both torch tensors and numpy arrays
    if hasattr(pts1, 'cpu'):
        pts1 = pts1.cpu().numpy()
        pts2 = pts2.cpu().numpy()

    if len(pts1) == 0:
        return np.array([]).reshape(0, 2).astype("int32"), np.array([]).reshape(0, 2).astype("int32")

    # Scale keypoints back to original image coordinates
    scale_x0 = w_org / w0_loaded
    scale_y0 = h_org / h0_loaded
    scale_x1 = w_sim / w1_loaded
    scale_y1 = h_sim / h1_loaded

    pts1[:, 0] *= scale_x0
    pts1[:, 1] *= scale_y0
    pts2[:, 0] *= scale_x1
    pts2[:, 1] *= scale_y1

    pts1 = pts1.astype("int32")
    pts2 = pts2.astype("int32")

    return pts1, pts2


def _filter_geometric(pts1, pts2, method, focal_length, principal_point,
                      threshold, image_size, ransac_method):
    """
    Internal geometric outlier filtering using Essential or Fundamental Matrix.

    Parameters
    ----------
    pts1 : numpy.ndarray
        Points from image 1, shape (N, 2).
    pts2 : numpy.ndarray
        Points from image 2, shape (N, 2).
    method : str
        Filtering method: "essential", "fundamental", or "none".
    focal_length : float or None
        Focal length in pixels.
    principal_point : tuple or None
        Principal point (cx, cy) in pixels.
    threshold : float
        Maximum allowed error in pixels.
    image_size : tuple or None
        Image size (width, height).
    ransac_method : str
        RANSAC variant.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array where True indicates inliers.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    if len(pts1) == 0:
        return np.array([], dtype=bool)

    method_lower = method.lower()

    if method_lower == "none":
        return np.ones(len(pts1), dtype=bool)

    elif method_lower == "essential":
        # Essential matrix requires at least 5 points
        if len(pts1) < 5:
            return np.ones(len(pts1), dtype=bool)

        # Estimate focal length if not provided
        if focal_length is None:
            if image_size is not None:
                focal_length = float(image_size[0])
            else:
                focal_length = max(
                    pts1[:, 0].max() - pts1[:, 0].min(),
                    pts1[:, 1].max() - pts1[:, 1].min()
                )
            warnings.warn(
                f"focal_length not provided for Essential Matrix filtering. "
                f"Estimated as {focal_length:.0f} pixels. "
                f"For better results, provide the actual focal length.",
                UserWarning,
                stacklevel=3
            )

        # Estimate principal point if not provided
        if principal_point is None:
            if image_size is not None:
                principal_point = (image_size[0] / 2, image_size[1] / 2)
            else:
                cx = (pts1[:, 0].max() + pts1[:, 0].min()) / 2
                cy = (pts1[:, 1].max() + pts1[:, 1].min()) / 2
                principal_point = (cx, cy)

        # Build camera matrix
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Map method string to OpenCV flag
        method_map = {
            "RANSAC": cv2.RANSAC,
            "LMEDS": cv2.LMEDS,
            "USAC_MAGSAC": cv2.USAC_MAGSAC,
            "USAC_DEFAULT": cv2.USAC_DEFAULT,
        }
        flag = method_map.get(ransac_method.upper(), cv2.USAC_MAGSAC)

        _, mask = cv2.findEssentialMat(pts1, pts2, K, method=flag, threshold=threshold)

        if mask is None:
            return np.ones(len(pts1), dtype=bool)

        return mask.ravel().astype(bool)

    elif method_lower == "fundamental":
        # Fundamental matrix requires at least 8 points
        if len(pts1) < 8:
            return np.ones(len(pts1), dtype=bool)

        # Map method string to OpenCV flag
        method_map = {
            "RANSAC": cv2.FM_RANSAC,
            "LMEDS": cv2.FM_LMEDS,
            "USAC_MAGSAC": cv2.USAC_MAGSAC,
            "USAC_DEFAULT": cv2.USAC_DEFAULT,
        }
        flag = method_map.get(ransac_method.upper(), cv2.USAC_MAGSAC)

        _, mask = cv2.findFundamentalMat(pts1, pts2, flag, threshold)

        if mask is None:
            return np.ones(len(pts1), dtype=bool)

        return mask.ravel().astype(bool)

    else:
        raise ValueError(
            f"Unknown outlier_filter '{method}'. "
            "Available: 'essential', 'fundamental', 'none'"
        )


def _filter_spatial(pts, grid_size, image_size, selection="first", random_state=None):
    """
    Spatial thinning of matched points using grid-based sampling.

    Parameters
    ----------
    pts : numpy.ndarray
        Points to thin, shape (N, 2). Typically pts1 (original image coords).
    grid_size : int
        Grid cell size in pixels (e.g., 100 for 100x100 px cells).
        Must be positive integer.
    image_size : tuple of int
        Image size as (width, height) in pixels. Required for proper grid coverage.
    selection : str, default "first"
        Point selection method within each grid cell:

        - "first": First point by input order (deterministic, fastest)
        - "random": Random selection (requires random_state for reproducibility)
        - "center": Point closest to cell center (ties: first by index)

    random_state : int or None, default None
        Seed for random selection. Required when selection="random".

    Returns
    -------
    mask : numpy.ndarray
        Boolean array where True indicates selected points, shape (N,).

    Raises
    ------
    ValueError
        If grid_size <= 0 or selection is unknown.
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    if len(pts) == 0:
        return np.array([], dtype=bool)

    # Calculate cell indices (vectorized)
    cell_col = (pts[:, 0] // grid_size).astype(int)
    cell_row = (pts[:, 1] // grid_size).astype(int)
    n_cols = int(np.ceil(image_size[0] / grid_size))
    cell_id = cell_row * n_cols + cell_col

    # Use pandas for efficient groupby
    df = pd.DataFrame({'idx': np.arange(len(pts)), 'cell': cell_id})

    if selection == "first":
        selected = df.groupby('cell')['idx'].first().values
    elif selection == "random":
        rng = np.random.default_rng(random_state)
        selected = df.groupby('cell')['idx'].apply(
            lambda x: rng.choice(x.values)
        ).values
    elif selection == "center":
        # Compute cell centers
        cell_center_x = (cell_col + 0.5) * grid_size
        cell_center_y = (cell_row + 0.5) * grid_size
        # Compute distance to cell center
        dist_to_center = np.sqrt(
            (pts[:, 0] - cell_center_x) ** 2 +
            (pts[:, 1] - cell_center_y) ** 2
        )
        df['dist'] = dist_to_center
        # Select point with minimum distance (ties: first by index)
        selected = df.loc[df.groupby('cell')['dist'].idxmin(), 'idx'].values
    else:
        raise ValueError(
            f"Unknown selection '{selection}'. "
            "Available: 'first', 'random', 'center'"
        )

    mask = np.zeros(len(pts), dtype=bool)
    mask[selected] = True
    return mask


def image_match(path_org, path_sim, method="akaze", outlier_filter="fundamental",
                params=None, threshold=10.0, ransac_method="USAC_MAGSAC",
                spatial_thin_grid=None, spatial_thin_selection="first",
                spatial_thin_random_state=None,
                plot_result=False, device="cpu", resize=None, **kwargs):
    """
    Feature matching between the original (real) photograph and a simulated landscape image.

    The workflow is:

    - Local feature detection (method-dependent)
    - Feature matching with Lowe's ratio test
    - Geometric outlier filtering (Essential/Fundamental Matrix)
    - Optional spatial thinning (grid-based sampling)

    Parameters
    ----------
    path_org : str
        Path for original photograph.
    path_sim : str
        Path for simulated landscape image.
    method : str, default "akaze"
        Matching method to use. Built-in methods: "akaze", "sift".
        With imm package installed: "sift-lightglue", "superpoint-lightglue",
        "minima-superpoint-lightglue", "roma", "tiny-roma", "minima-roma",
        "loftr", "minima-loftr", "ufm", "rdd", "master".
    outlier_filter : str, default "fundamental"
        Outlier filtering method:

        - "fundamental": Fundamental Matrix filtering (default). Use when camera
          intrinsics are unknown. Requires at least 8 matches.
        - "essential": Essential Matrix filtering. Recommended when params
          with fov is provided. Requires at least 5 matches.
        - "none": No geometric filtering. Use this when you plan to
          apply custom filtering later.

    params : dict, optional
        Camera parameters dict containing fov, w, h, and optionally cx, cy.
        Used to compute focal_length and principal_point for "essential" filter.
        focal_length is calculated as: (w / 2) / tan(fov * pi / 180 / 2)
    threshold : float, default 10.0
        Outlier threshold in pixels for Essential/Fundamental filtering.
        Higher values are more permissive (allow more matches through).
    ransac_method : str, default "USAC_MAGSAC"
        RANSAC variant for geometric filtering:

        - "RANSAC": Standard RANSAC
        - "LMEDS": Least-Median of Squares
        - "USAC_MAGSAC": MAGSAC++ (most robust, recommended)
        - "USAC_DEFAULT": Default USAC configuration

    spatial_thin_grid : int, optional
        Grid cell size in pixels for spatial thinning. If specified, keeps at most
        one match per grid cell. Example: 100 keeps ~1 point per 100x100 px region.
        Applied AFTER geometric outlier filtering.
    spatial_thin_selection : str, default "first"
        Selection method for spatial thinning:

        - "first": First point by input order (deterministic, fastest)
        - "random": Random selection (use spatial_thin_random_state for reproducibility)
        - "center": Point closest to cell center

    spatial_thin_random_state : int, optional
        Random seed when spatial_thin_selection="random".
    plot_result : bool, default False
        Whether to return a result plot.
    device : str, default "cpu"
        Device to use for imm methods ("cpu" or "cuda"). Ignored for built-in methods.
    resize : int, optional
        Resize images to this size for imm methods.
        Keypoints are automatically scaled back to original coordinates.

        .. note::
            For memory-intensive methods (roma, tiny-roma, minima-roma, loftr,
            minima-loftr, ufm, rdd, master), if resize is None, images are
            automatically resized to 640px to prevent out-of-memory errors.
            LightGlue-based methods (sift-lightglue, superpoint-lightglue,
            minima-superpoint-lightglue) can handle full-resolution images
            and do not auto-resize.

    **kwargs
        Additional keyword arguments passed to imm's get_matcher (e.g., max_num_keypoints).

    Returns
    -------
    points : pd.DataFrame
        The coordinates of matched points. (Left-Top origin)
    plot : np.array or None
        An OpenCV image of result plot if plot_result=True, otherwise None.

    Examples
    --------
    Basic matching with Fundamental Matrix filtering (default):

    >>> match, _ = image_match(path_org, path_sim, method="akaze")

    Matching with Essential Matrix filtering using params:

    >>> match, _ = image_match(path_org, path_sim, method="minima-roma",
    ...                        outlier_filter="essential", params=params,
    ...                        device="cuda")
    """
    import math

    # Compute focal_length and principal_point from params if provided
    focal_length = None
    principal_point = None
    if params is not None:
        if "fov" in params and "w" in params:
            focal_length = (params["w"] / 2) / math.tan(params["fov"] * math.pi / 180 / 2)
        if "cx" in params and "cy" in params:
            principal_point = (params["cx"], params["cy"])
        elif "w" in params and "h" in params:
            principal_point = (params["w"] / 2, params["h"] / 2)

    method_lower = method.lower()
    im_org = None

    if method_lower == "akaze":
        im_org = cv2.imread(path_org)
        im_sim = cv2.imread(path_sim)
        pts1, pts2 = _opencv_match(im_org, im_sim, detector_type="akaze")
    elif method_lower == "sift":
        im_org = cv2.imread(path_org)
        im_sim = cv2.imread(path_sim)
        pts1, pts2 = _opencv_match(im_org, im_sim, detector_type="sift")
    elif method_lower in [m.lower() for m in IMM_METHODS]:
        # Auto-resize for memory-intensive methods (non-LightGlue)
        effective_resize = resize
        if method_lower not in [m.lower() for m in _LIGHTGLUE_METHODS]:
            if resize is None:
                effective_resize = _DEFAULT_RESIZE_HEAVY
                warnings.warn(
                    f"Method '{method}' is memory-intensive. "
                    f"Automatically resizing images to {_DEFAULT_RESIZE_HEAVY}px "
                    f"to prevent out-of-memory errors. "
                    f"Set 'resize' explicitly to override this behavior.",
                    UserWarning,
                    stacklevel=2
                )
        pts1, pts2 = _imm_match(path_org, path_sim, method, device, resize=effective_resize, **kwargs)
        im_org = cv2.imread(path_org)
    else:
        available = ["akaze", "sift"] + IMM_METHODS
        raise ValueError(
            f"Unknown method '{method}'. Available methods: {available}"
        )

    # Get image size for filtering
    image_size = (im_org.shape[1], im_org.shape[0]) if im_org is not None else None

    # Apply geometric outlier filtering if requested
    if outlier_filter != "none" and len(pts1) > 0:
        mask = _filter_geometric(
            pts1, pts2,
            method=outlier_filter,
            focal_length=focal_length,
            principal_point=principal_point,
            threshold=threshold,
            image_size=image_size,
            ransac_method=ransac_method
        )
        pts1 = pts1[mask]
        pts2 = pts2[mask]

    # Apply spatial thinning AFTER geometric filtering
    if spatial_thin_grid is not None and len(pts1) > 0:
        if image_size is None:
            raise ValueError(
                "image_size could not be determined. "
                "Ensure path_org is a valid image file."
            )
        mask = _filter_spatial(
            pts1,
            grid_size=spatial_thin_grid,
            image_size=image_size,
            selection=spatial_thin_selection,
            random_state=spatial_thin_random_state
        )
        pts1 = pts1[mask]
        pts2 = pts2[mask]

    # Ensure pts1/pts2 are 2D arrays with shape (N, 2)
    if len(pts1) == 0:
        pts = pd.DataFrame(columns=["u_org", "v_org", "u_sim", "v_sim"])
    else:
        pts1 = np.atleast_2d(pts1)
        pts2 = np.atleast_2d(pts2)
        pts = pd.DataFrame(np.hstack((pts1, pts2)), columns=["u_org", "v_org", "u_sim", "v_sim"])

    if plot_result:
        if im_org is None:
            im_org = cv2.imread(path_org)
        plot_img = plot_matches(im_org, pts)
        return pts, plot_img
    else:
        return pts, None


def plot_matches(image, matches, color=(180, 105, 255), thickness=20, tip_length=0.3):
    """
    Plot feature matches on an image.

    Parameters
    ----------
    image : numpy.ndarray
        Image to draw on (BGR format).
    matches : pd.DataFrame
        Match coordinates with columns u_org, v_org, u_sim, v_sim.
    color : tuple, default (180, 105, 255)
        Arrow color in BGR.
    thickness : int, default 20
        Arrow thickness.
    tip_length : float, default 0.3
        Arrow tip length as fraction of arrow length.

    Returns
    -------
    plot : numpy.ndarray
        Image with arrows drawn.
    """
    plot_img = image.copy()

    if len(matches) == 0:
        return plot_img

    pts1 = matches[["u_org", "v_org"]].to_numpy().astype(np.int32)
    pts2 = matches[["u_sim", "v_sim"]].to_numpy().astype(np.int32)

    for i in range(len(pts1)):
        plot_img = cv2.arrowedLine(
            plot_img, tuple(pts1[i]), tuple(pts2[i]),
            color=color, thickness=thickness, tipLength=tip_length
        )

    plot_img = cv2.putText(
        plot_img, f"simulated <- original ({len(pts1)} matches)",
        (int(plot_img.shape[1] * 0.15), int(plot_img.shape[0] * 0.05)),
        cv2.FONT_HERSHEY_TRIPLEX, 5, (0, 0, 0), 5, cv2.LINE_AA
    )

    return plot_img


def set_gcp(match, rev_proj):
    """
    Adds geographic coordinates to the matched point pairs.
    The result of this function will be used as the Ground Control Points (GCPs)
    during camera parameter estimation.

    Parameters
    ----------
    match : pd.DataFrame
        Result of alproj.gcp.image_match()
    rev_proj : pd.DataFrame
        Result of alproj.project.reverse_proj

    Returns
    -------
    gcp : pd.DataFrame
        A dataframe with 5 columns

        u : int
            x_axis coordinates of the Ground Control Points on the original photograph. Left-Top origin.
        v : int
            y_axis coordinates of the GCPs.
        x : float
            X coordinates of GCPs in a projected coordinate system (e.g., UTM in meters).
        y : float
            Y coordinates of GCPs in the same projected coordinate system.
        z : float
            Z coordinates (elevation) of GCPs in the same unit (e.g., meters).
    """
    gcp = pd.merge(match, rev_proj, how="left",\
         left_on=["u_sim", "v_sim"], right_on=["u", "v"]) \
             [["u_org","v_org","x","y","z"]] \
                 .rename(columns={"u_org":"u", "v_org":"v"})
    return gcp.dropna(how="any", axis=0)


def filter_gcp_distance(gcp, params, min_distance=None, max_distance=None):
    """
    Filter GCPs based on 3D distance from the camera position.

    Parameters
    ----------
    gcp : pd.DataFrame
        GCP data with columns u, v, x, y, z (geographic coordinates in
        projected CRS, e.g., meters). Typically the result of set_gcp().
    params : dict
        Camera parameters containing 'x', 'y', 'z' keys (camera position
        in the same projected CRS as gcp).
    min_distance : float, optional
        Minimum distance threshold in coordinate units (e.g., meters).
        Points closer than this are excluded. Must be non-negative if specified.
    max_distance : float, optional
        Maximum distance threshold. Points farther than this are excluded.
        Must be >= min_distance if both are specified.

    Returns
    -------
    gcp_filtered : pd.DataFrame
        Filtered GCP data with reset index.

    Raises
    ------
    KeyError
        If params lacks 'x', 'y', or 'z' keys.
    ValueError
        If min_distance < 0 or max_distance < min_distance.

    Notes
    -----
    - Coordinates must be in a projected CRS (e.g., UTM) for accurate
      Euclidean distance. Using lat/lon directly will produce incorrect results.
    - Rows with NaN in x, y, or z are excluded from the result.

    Examples
    --------
    >>> gcps = set_gcp(match, df)
    >>> gcps_filtered = filter_gcp_distance(gcps, params, min_distance=100)
    """
    # Validate params keys
    for key in ('x', 'y', 'z'):
        if key not in params:
            raise KeyError(f"params must contain '{key}' key")

    # Validate distance parameters
    if min_distance is not None and min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    if min_distance is not None and max_distance is not None:
        if max_distance < min_distance:
            raise ValueError("max_distance must be >= min_distance")

    if len(gcp) == 0:
        return gcp.copy()

    if min_distance is None and max_distance is None:
        return gcp.copy()

    # Drop rows with NaN coordinates
    gcp_valid = gcp.dropna(subset=['x', 'y', 'z'])

    # Calculate 3D Euclidean distance (vectorized)
    dx = gcp_valid["x"].values - params["x"]
    dy = gcp_valid["y"].values - params["y"]
    dz = gcp_valid["z"].values - params["z"]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    # Apply filters
    mask = np.ones(len(gcp_valid), dtype=bool)
    if min_distance is not None:
        mask &= (distance >= min_distance)
    if max_distance is not None:
        mask &= (distance <= max_distance)

    return gcp_valid[mask].reset_index(drop=True)
