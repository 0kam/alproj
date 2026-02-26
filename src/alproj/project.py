import numpy as np
import moderngl as gl
import math
import cv2
import pandas as pd
import warnings
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import generic_filter
from alproj.optimize import _distort

def projection_mat(fov_x_deg, w, h, near=-1, far=1, cx=None, cy=None):
    """
    Makes an OpenGL-style projection matrix from Field of View, width, and height of an image.
    See https://learnopengl.com/Getting-started/Coordinate-Systems .

    Parameters
    ----------
    fov_x_deg : float
        Field of View in degrees.
    w : int
        Width in pixels
    h : int
        Height in pixels
    near : float, default -1
        Z-axis coordinate of near plane.
    far : float, default 1
        Z-axis coordinate of far plane.
    cx : float, default None
        X-axis coordinate of principal point. If None, w/2. 
    cy : float, default None
        Y-axis coordinate of principal point. If None, h/2.
    
    Returns
    -------
    projection_mat : numpy.ndarray
        A projection matrix.
    """
    if cx == None:
        cx = w/2
    if cy == None:
        cy = h/2
    fov_x = fov_x_deg * math.pi / 180
    fov_y = 2 * math.atan(math.tan(fov_x / 2) * h / w)
    fx = 1 / math.tan(fov_x/2)
    fy = 1 / math.tan(fov_y/2)
    # Build in row-major (math convention), then convert to column-major
    # for OpenGL via transpose.  The off-diagonal entries (w-2*cx)/w and
    # -(h-2*cy)/h shift the principal point in NDC space.  Row 3 gives
    # w_clip = Z (positive for objects in front after the modelview N
    # negation), while row 2 provides a simple depth value for the
    # depth test.
    mat = np.array([
        [fx, 0,  (w-2*cx)/w,   0],
        [0,  fy, -(h-2*cy)/h,  0],
        [0,  0,  0,            -1],
        [0,  0,  1,             0]
    ])
    return mat.transpose().flatten()

def modelview_mat(pan_deg, tilt_deg, roll_deg, t_x, t_y, t_z):
    """
    Makes an OpenGL-style modelview matrix from euler angles and camera location in world coordinate system.
    See https://learnopengl.com/Getting-started/Coordinate-Systems .

    Derived from :func:`alproj.optimize.extrinsic_mat` with coordinate
    permutation so that the OpenGL renderer and the forward projection use
    identical rotation logic.

    OpenGL vertices are stored as ``(x_geo, z_geo, y_geo)`` while
    ``extrinsic_mat`` works in geographic ``(x, y, z)`` order.  The
    conversion is::

        M_gl = N @ E @ P

    where *E* is the 4x4 extrinsic matrix, *P* swaps indices 1 and 2
    (y <-> z) to convert vertex order to geographic order, and *N*
    negates the camera-space Z row so that objects in front of the camera
    have *positive* Z -- matching the projection matrix convention used
    by :func:`projection_mat` (``w_clip = Z``).

    Parameters
    ----------
    pan_deg : float
        Pan angle in degrees
    tilt_deg : float
        Tilt angle in degrees
    roll_deg : float
        Roll angle in degrees
    t_x : float
        X-axis (latitudinal) coordinate of the camera location in a (planar) geographic coordinate system.
    t_y : float
        Y-axis (longitudinal) coordinate of the camera location in a (planar) geographic coordinate system.
    t_z : float
        Z-axis (elevational) coordinate of the camera location in a (planar) geographic coordinate system.

    Returns
    -------
    modelview_mat : numpy.ndarray
        A modelview matrix (flattened, column-major for OpenGL).
    """
    from alproj.optimize import extrinsic_mat

    E = extrinsic_mat(pan_deg, tilt_deg, roll_deg, t_x, t_y, t_z)

    # Permutation: swap y and z to convert (x_geo, z_geo, y_geo) -> (x_geo, y_geo, z_geo)
    P = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # Negate camera-space Z so that objects in front have Z > 0,
    # matching the projection convention where w_clip = Z.
    N = np.diag([1.0, 1.0, -1.0, 1.0])

    M = N @ E @ P
    return M.transpose().flatten()

def _pinhole_remap(raw_rect, params_rect, params_target, n_iter=20):
    """
    Remap a wide rectilinear rendered image to the target pinhole image
    with distortion, using inverse mapping with cv2.remap.

    For each output (distorted) pixel (u_d, v_d) in the target image,
    computes the corresponding source rectilinear pixel by:
      distorted pixel -> undistort -> 3D ray -> rectilinear pixel

    Parameters
    ----------
    raw_rect : numpy.ndarray
        Wide rectilinear rendered image (h_rect, w_rect, channels).
    params_rect : dict
        Camera parameters used to render raw_rect (rectilinear, no distortion).
    params_target : dict
        Target pinhole camera parameters (with distortion).
    n_iter : int, default 20
        Number of fixed-point iterations for inverting the distortion.

    Returns
    -------
    result : numpy.ndarray
        Distorted image at target resolution (h_target, w_target, channels).
    """
    h_rect, w_rect, n_ch = raw_rect.shape
    w_out = int(params_target["w"])
    h_out = int(params_target["h"])

    # ---- Step 1: Build output grid (target distorted pixels) ----
    map_x, map_y = np.meshgrid(np.arange(w_out), np.arange(h_out))
    grid = np.stack([map_x.flatten(), map_y.flatten()]).T.astype(np.float64)

    # ---- Step 2: Invert distortion in target image space ----
    d = np.array([
        params_target["a1"], params_target["a2"],
        params_target["k1"], params_target["k2"], params_target["k3"],
        params_target["k4"], params_target["k5"], params_target["k6"],
        params_target["p1"], params_target["p2"],
        params_target["s1"], params_target["s2"], params_target["s3"], params_target["s4"]])
    cx_t = params_target.get("cx")
    cy_t = params_target.get("cy")

    undistorted = grid.copy()
    for _ in range(n_iter):
        re_distorted = _distort(
            undistorted, w_out, h_out,
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
            d[8], d[9], d[10], d[11], d[12], d[13],
            cx=cx_t, cy=cy_t
        )
        undistorted = undistorted + (grid - re_distorted)

    # ---- Step 3: Undistorted target pixel -> 3D ray -> rendered pixel ----
    # Target intrinsics
    fov_t = params_target["fov"] * math.pi / 180
    fov_ty = 2 * math.atan(math.tan(fov_t / 2) * h_out / w_out)
    fx_t = w_out / (2 * math.tan(fov_t / 2))
    fy_t = h_out / (2 * math.tan(fov_ty / 2))
    cx_t_val = cx_t if cx_t is not None else w_out / 2
    cy_t_val = cy_t if cy_t is not None else h_out / 2

    # Rendered image intrinsics
    fov_r = params_rect["fov"] * math.pi / 180
    fov_ry = 2 * math.atan(math.tan(fov_r / 2) * h_rect / w_rect)
    fx_r = w_rect / (2 * math.tan(fov_r / 2))
    fy_r = h_rect / (2 * math.tan(fov_ry / 2))
    cx_r = params_rect.get("cx", w_rect / 2)
    cy_r = params_rect.get("cy", h_rect / 2)

    u_undist = undistorted[:, 0]
    v_undist = undistorted[:, 1]

    # Pinhole model: u = w - (fx*X/Z + cx), v = fy*Y/Z + cy
    # => X/Z = (w - u - cx) / fx, Y/Z = (v - cy) / fy
    XZ = (w_out - u_undist - cx_t_val) / fx_t
    YZ = (v_undist - cy_t_val) / fy_t

    # Ray -> rendered image pixel
    u_rect = (w_rect - (fx_r * XZ + cx_r)).astype(np.float32)
    v_rect = (fy_r * YZ + cy_r).astype(np.float32)

    map_uv = np.stack([u_rect, v_rect]).reshape([2, h_out, w_out])

    # ---- Step 4: Remap ----
    is_color_data = (n_ch == 3 and raw_rect.dtype == np.float32
                     and raw_rect.max() <= 1.0 + 1e-6)
    interp = cv2.INTER_LINEAR if is_color_data else cv2.INTER_NEAREST

    result = cv2.remap(raw_rect, map_uv[0, :, :], map_uv[1, :, :],
                       interpolation=interp,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)
    return result

def _opengl_render(vert, value, ind, params, min_distance=None):
    """
    Render a rectilinear (undistorted) image using OpenGL.

    This is the core OpenGL rendering used by persp_proj(). It does NOT apply
    lens distortion — the caller is responsible for post-processing.

    Parameters
    ----------
    vert : numpy.ndarray
        Coordinates of vertices.
    value : numpy.ndarray
        Values of vertices (e.g., colors or geographic coordinates).
    ind : numpy.ndarray
        Index data of vertices.
    params : dict
        Camera parameters (must already have offsets applied).
    min_distance : float, default None
        Minimum distance from camera for masking.

    Returns
    -------
    raw : numpy.ndarray
        Rendered image (h, w, 3), float32, undistorted.
    """
    ctx = gl.create_standalone_context()
    ctx.enable(gl.DEPTH_TEST)
    ctx.enable(gl.CULL_FACE)
    vbo = ctx.buffer(vert.astype("f4").tobytes())
    cbo = ctx.buffer(value.astype("f4").tobytes())
    ibo = ctx.buffer(ind.astype("i4").tobytes())
    prog = ctx.program(
        vertex_shader='''
            #version 330
            precision highp float;
            in vec3 in_vert;
            in vec3 in_color;
            out vec3 v_color;
            out float v_distance;
            uniform mat4 proj;
            uniform mat4 view;

            void main() {
                vec4 local_pos = vec4(in_vert, 1.0);
                vec4 view_pos = vec4(view * local_pos);
                gl_Position = vec4(proj * view_pos);
                v_color = in_color;
                v_distance = length(view_pos.xyz);
            }
        ''',
        fragment_shader='''
            #version 330
            precision highp float;
            in vec3 v_color;
            in float v_distance;
            uniform float min_dist;

            layout(location=0)out vec4 f_color;
            void main() {
                if (min_dist > 0.0 && v_distance < min_dist) {
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    f_color = vec4(v_color, 1.0);
                }
            }
        '''
    )

    # Adjust cy for np.flipud: the vertical flip maps row i -> h-1-i,
    # which shifts the effective principal point. Using cy_adj = h - 1 - cy
    # in the projection matrix compensates exactly so that the rendered v
    # coordinates match _pinhole_project's v = fy*Y/Z + cy.
    cy_raw = params.get("cy")
    cy_adj = (params["h"] - 1 - cy_raw) if cy_raw is not None else None
    proj_mat = projection_mat(params["fov"], params["w"], params["h"],
                              cx=params.get("cx"), cy=cy_adj)
    view_mat = modelview_mat(
        params["pan"], params["tilt"], params["roll"],
        params["x"], params["y"], params["z"])

    prog['proj'].value = tuple(proj_mat)
    prog['view'].value = tuple(view_mat)
    prog['min_dist'].value = float(min_distance) if min_distance is not None else 0.0

    vao_content = [(vbo, "3f", "in_vert"), (cbo, "3f", "in_color")]
    vao = ctx.vertex_array(program=prog, content=vao_content, index_buffer=ibo)
    rbo = ctx.renderbuffer((params["w"], params["h"]), dtype="f4")
    drbo = ctx.depth_renderbuffer((params["w"], params["h"]))
    fbo = ctx.framebuffer(rbo, drbo)

    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render()

    raw = np.frombuffer(fbo.read(dtype="f4"), dtype="float32")
    raw = raw.reshape(params["h"], params["w"], 3)
    raw = np.flipud(raw)

    vao.release()
    fbo.release()
    ctx.release()
    vbo.release()
    cbo.release()
    ibo.release()
    prog.release()
    del vao_content

    return raw


def _undistort_theta(theta_d, k1, k2, k3, k4, n_iter=10):
    """
    Invert the equidistant fisheye distortion model via Newton's method.

    Given distorted angle theta_d, find theta such that:
        theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)

    Parameters
    ----------
    theta_d : numpy.ndarray
        Distorted angle values (negative for visible points in this convention).
    k1, k2, k3, k4 : float
        Fisheye distortion coefficients.
    n_iter : int, default 10
        Number of Newton iterations.

    Returns
    -------
    theta : numpy.ndarray
        Undistorted angle values, same shape as theta_d.
    """
    theta = theta_d.copy()
    for _ in range(n_iter):
        t2 = theta ** 2
        # f(theta) = theta * (1 + k1*t2 + k2*t2^2 + k3*t2^3 + k4*t2^4) - theta_d
        poly = 1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4
        f = theta * poly - theta_d
        # f'(theta) = 1 + 3*k1*t2 + 5*k2*t2^2 + 7*k3*t2^3 + 9*k4*t2^4
        fp = 1.0 + 3*k1 * t2 + 5*k2 * t2**2 + 7*k3 * t2**3 + 9*k4 * t2**4
        # Guard against zero derivative (shouldn't happen for small distortion)
        fp = np.where(np.abs(fp) > 1e-12, fp, 1.0)
        theta = theta - f / fp
    return theta


def _fisheye_remap(raw_rect, params_rect, params_fisheye):
    """
    Remap a rectilinear rendered image to equidistant fisheye projection
    using inverse mapping with cv2.remap.

    For each output fisheye pixel (u_out, v_out), computes the corresponding
    source rectilinear pixel (u_rect, v_rect) by inverting the forward
    projection chain:
      fisheye pixel -> distorted angle -> undistorted angle -> 3D ray -> rectilinear pixel

    Parameters
    ----------
    raw_rect : numpy.ndarray
        Rectilinear rendered image (h_rect, w_rect, channels).
    params_rect : dict
        Camera parameters used to render raw_rect (rectilinear, no distortion).
    params_fisheye : dict
        Target fisheye camera parameters.

    Returns
    -------
    result : numpy.ndarray
        Fisheye-remapped image (h_out, w_out, channels).
    """
    h_rect, w_rect, n_ch = raw_rect.shape

    if params_fisheye["fov"] <= 0:
        raise ValueError(f"fov must be positive, got {params_fisheye['fov']}")

    # ---- Rectilinear intrinsics ----
    fov_r = params_rect["fov"] * math.pi / 180
    fov_ry = 2 * math.atan(math.tan(fov_r / 2) * h_rect / w_rect)
    fx_r = w_rect / (2 * math.tan(fov_r / 2))
    fy_r = h_rect / (2 * math.tan(fov_ry / 2))
    cx_r, cy_r = w_rect / 2, h_rect / 2

    # ---- Fisheye intrinsics ----
    w_out = int(params_fisheye["w"])
    h_out = int(params_fisheye["h"])
    fov_rad = params_fisheye["fov"] * math.pi / 180
    f_fish = w_out / fov_rad
    cx_f = params_fisheye["cx"]
    cy_f = params_fisheye["cy"]

    k1 = params_fisheye.get("k1", 0)
    k2 = params_fisheye.get("k2", 0)
    k3 = params_fisheye.get("k3", 0)
    k4 = params_fisheye.get("k4", 0)

    # ---- Aspect ratio and tangential distortion parameters ----
    a1 = params_fisheye.get("a1", 1.0)
    a2 = params_fisheye.get("a2", 1.0)
    p1 = params_fisheye.get("p1", 0.0)
    p2 = params_fisheye.get("p2", 0.0)

    # ---- Inverse mapping: fisheye pixel -> rectilinear pixel ----
    # Build a grid over every output fisheye pixel.
    u_out, v_out = np.meshgrid(
        np.arange(w_out, dtype=np.float64),
        np.arange(h_out, dtype=np.float64))

    # Step 0: Undo tangential distortion (iterative fixed-point).
    # Forward: u_final = u_fish + du(u_fish), so u_fish = u_final - du(u_fish)
    u_fish = u_out.copy()
    v_fish = v_out.copy()
    if p1 != 0.0 or p2 != 0.0:
        w_half = w_out / 2
        h_half = h_out / 2
        for _ in range(5):
            x_n = (u_fish - (w_out - cx_f)) / w_half
            y_n = (v_fish - cy_f) / h_half
            r2 = x_n**2 + y_n**2
            du = 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
            dv = p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n
            u_fish = u_out - du * w_half
            v_fish = v_out - dv * h_half

    # Step 1: Fisheye pixel -> image-space displacement.
    # Forward: u = w - (r_img * X/r * a1 + cx), v = r_img * Y/r * a2 + cy
    # Undo aspect ratio to get isotropic displacements.
    if abs(a1) < 1e-10 or abs(a2) < 1e-10:
        raise ValueError(f"a1 and a2 must be non-zero, got a1={a1}, a2={a2}")
    dx = (w_out - u_fish - cx_f) / a1
    dy = (v_fish - cy_f) / a2

    # r_img = f_fish * theta_d, and r_img <= 0 in this convention,
    # so |r_img| = sqrt(dx^2 + dy^2).
    r_img_abs = np.sqrt(dx**2 + dy**2)

    # Step 2: Recover distorted angle theta_d (negative).
    theta_d = -r_img_abs / f_fish

    # Step 3: Invert distortion to get undistorted angle theta.
    has_distortion = (k1 != 0 or k2 != 0 or k3 != 0 or k4 != 0)
    if has_distortion:
        theta = _undistort_theta(theta_d, k1, k2, k3, k4)
    else:
        theta = theta_d

    # Step 4: Compute r_cam = -tan(theta).
    # theta <= 0 for visible points, so -tan(theta) >= 0.
    # Clamp to avoid tan() blowup beyond the rectilinear FOV limit.
    max_theta = -math.pi / 2 + 1e-6
    theta_clamped = np.clip(theta, max_theta, 0.0)
    r_cam = -np.tan(theta_clamped)

    # Step 5: Recover the 3D ray direction (X, Y) from the unit direction.
    # mx = X/r_cam = dx / r_img  where  r_img = -r_img_abs
    # So mx = dx / (-r_img_abs) = -dx / r_img_abs
    # Similarly my = -dy / r_img_abs
    safe_r_img = np.where(r_img_abs > 1e-10, r_img_abs, 1.0)
    mx = -dx / safe_r_img
    my = -dy / safe_r_img

    X = r_cam * mx
    Y = r_cam * my

    # Step 6: 3D ray -> rectilinear pixel.
    # From the forward mapping:
    #   X = -(w_rect - uu_r - cx_r) / fx_r  =>  uu_r = w_rect - cx_r + X * fx_r
    #   Y = -(vv_r - cy_r) / fy_r           =>  vv_r = cy_r - Y * fy_r
    map_x = (w_rect - cx_r + X * fx_r).astype(np.float32)
    map_y = (cy_r - Y * fy_r).astype(np.float32)

    # On optical axis (r_img_abs ~ 0), the ray points straight ahead.
    # X = Y = 0, so the rectilinear source pixel is the image center.
    on_axis = r_img_abs < 1e-10
    map_x = np.where(on_axis, np.float32(w_rect - cx_r), map_x)
    map_y = np.where(on_axis, np.float32(cy_r), map_y)

    # ---- Apply cv2.remap ----
    # Use INTER_LINEAR for color data, INTER_NEAREST for coordinate data
    # to avoid interpolating geographic coordinates.
    is_color_data = (n_ch == 3 and raw_rect.dtype == np.float32
                     and raw_rect.max() <= 1.0 + 1e-6)
    interp = cv2.INTER_LINEAR if is_color_data else cv2.INTER_NEAREST
    result = cv2.remap(raw_rect, map_x, map_y,
                       interpolation=interp,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)

    return result


def _render_pinhole(vert, value, ind, params, min_distance):
    """Pinhole model: wide OpenGL render + distortion remap."""
    rect_fov = min(params["fov"] + 20, 140)
    if params["fov"] > 120:
        warnings.warn(
            f"Pinhole FOV ({params['fov']:.0f}\u00b0) exceeds 120\u00b0. "
            f"The rectilinear source (capped at {rect_fov:.0f}\u00b0) may not fully "
            f"cover the distorted field. For best results, keep FOV \u2264 120\u00b0.")
    w_rect = int(params["w"] * 1.5)
    h_rect = int(params["h"] * 1.5)

    params_rect = {
        "x": params["x"], "y": params["y"], "z": params["z"],
        "fov": rect_fov,
        "pan": params["pan"], "tilt": params["tilt"],
        "roll": params["roll"],
        "w": w_rect, "h": h_rect,
        "cx": w_rect / 2, "cy": h_rect / 2,
    }

    raw_rect = _opengl_render(vert, value, ind, params_rect, min_distance)
    return _pinhole_remap(raw_rect, params_rect, params)


def _render_fisheye(vert, value, ind, params, min_distance):
    """Fisheye model: wide rectilinear render + fisheye remap."""
    rect_fov = min(params["fov"] + 20, 140)
    if params["fov"] > 120:
        warnings.warn(
            f"Fisheye FOV ({params['fov']:.0f}°) exceeds 120°. "
            f"The rectilinear source (capped at {rect_fov:.0f}°) cannot fully "
            f"cover the fisheye field. Edge regions will use interpolated values. "
            f"For best results, keep fisheye FOV ≤ 120°.")
    w_rect = int(params["w"] * 1.5)
    h_rect = int(params["h"] * 1.5)

    params_rect = {
        "x": params["x"], "y": params["y"], "z": params["z"],
        "fov": rect_fov,
        "pan": params["pan"], "tilt": params["tilt"],
        "roll": params["roll"],
        "w": w_rect, "h": h_rect,
        "cx": w_rect / 2, "cy": h_rect / 2,
    }

    raw_rect = _opengl_render(vert, value, ind, params_rect, min_distance)
    return _fisheye_remap(raw_rect, params_rect, params)


def persp_proj(vert, value, ind, params, offsets=None, min_distance=None):
    """
    3D to 2D perspective projection of vertices, with given camera parameters.

    Parameters
    ----------
    vert : numpy.ndarray
        Coordinates of vertices, in X(latitudinal), Z(vertical), Y(longitudinal) order.
    value : numpy.ndarray
        Values of vertices. e.g. colors, geographic coordinates.
    ind : numpy.ndarray
        Index data of vertices. See http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-9-vbo-indexing/ .
    params : dict
        Camera parameters.

        x : float
            The latitudinal coordinate of the shooting point in planar (e.g. UTM) coordinate reference systems.
        y : float
            The longitudinal coordinate of the shooting point.
        z : float
            The vertical coordinate of the shooting point, the unit of z must be the same as x and y (e.g. m).
        fov : float
            A Field of View in degree.
        pan : float
            A pan angle in degree. North is 0 degree and East is 90 degree. The rotation angles (pan, tilt, roll) follows the OpenCV's left-handed coordinate system.
        tilt : float
            A tilt angle in degree. 0 indicates that the camera is horizontal. A positive value indicates that the camera looks up.
        roll : float
            A roll angle in degree. A positive value indicates that camera leans to the right.
        w : int
            An image width in pixel.
        h : int
            An image height in pixel
        cx : float
            The X coordinate of the principle point
        cy : float
            The Y coordinate of the principle point
        model : str, default None
            Camera model. If ``"fisheye"``, uses equidistant fisheye projection
            via forward splatting from a wide rectilinear render.
            If not specified, uses the default pinhole model with distortion.
        a1, a2 : float
            Distortion coefficients that calibrate non-equal aspect ratio of each pixel.
            Used by both pinhole and fisheye models.
        k1, k2, k3, k4, k5, k6 : float
            Radial distortion coefficients.
            For pinhole: image-space polynomial distortion.
            For fisheye: angle-space distortion (only k1-k4 used).
        p1, p2 : float
            Tangential (decentering) distortion coefficients.
            Used by both pinhole and fisheye models.
        s1, s2, s3, s4 : float
            Prism distortion coefficients. (pinhole model only)
    offsets : numpy.ndarray, default None
        Offset for vertex coordinates. Usually returned by alproj.surface.get_colored_surface().
    min_distance : float, default None
        Minimum distance from camera in coordinate units (e.g., meters).
        Pixels closer than this distance will be rendered as black.
        Useful for masking near-field objects that may cause matching errors.

    Returns
    -------
    raw : numpy.ndarray
        Projected result.

    """
    params = params.copy()
    if offsets is not None:
        params["x"] = params["x"] - offsets[0]
        params["y"] = params["y"] - offsets[2]
        params["z"] = params["z"] - offsets[1]

    if params.get("model") == "fisheye":
        return _render_fisheye(vert, value, ind, params, min_distance)
    return _render_pinhole(vert, value, ind, params, min_distance)

def sim_image(vert, color, ind, params, offsets=None, min_distance=None):
    """
    Renders a simulated image of landscape with given surface and camera parameters.

    Parameters
    ----------
    vert : numpy.ndarray
        Vertex coordinates of the surface returned by alproj.surface.get_colored_surface().
    color : numpy.ndarray
        Vertex colors in RGB, returned by alproj.surface.get_colored_surface().
    ind : numpy.ndarray
        Index data of vertices, returned by alproj.surface.get_colored_surface().
    params : dict
        Camera parameters. See alproj.project.persp_proj().
    offsets : numpy.ndarray, default None
        Offset for vertex coordinates. Usually returned by alproj.surface.get_colored_surface().
    min_distance : float, default None
        Minimum distance from camera in coordinate units (e.g., meters).
        Pixels closer than this distance will be rendered as black.
        Useful for masking near-field objects that may cause matching errors.

    Returns
    -------
    img : numpy.ndarray
        Rendered image in OpenCV's BGR format.
    """
    raw = persp_proj(vert, color, ind, params, offsets, min_distance=min_distance) * 255
    raw = raw.astype(np.uint8)
    img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    return img

def reverse_proj(array, vert, ind, params, offsets=None, chnames=["B", "G", "R"]):
    """
    2D to 3D reverse-projection (geo-rectification) of given array, onto given surface, with given camera parameters.
    Reverse-projected array will be returned as pandas.DataFrame with channel names, coordinates in the original array,
    and coordinates in the geographic coordinate system.

    Parameters
    ----------
    array : numpy.ndarray
        Target array, such as landscape photograph. The shape of the array must be (height, width, channels).
    vert : numpy.ndarray
        Vertex coordinates of the surface.
    ind : numpy.ndarray
        Index array that shows which three points shape a triangle. See http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-9-vbo-indexing/ .
    params : dict
        Camera parameters. See alproj.project.persp_proj().
    offsets : numpy.ndarray, default None
        Offset for vertex coordinates. Usually returned by alproj.surface.get_colored_surface().
    chnames : list of str, default ["B", "G", "R"]
        Channel names of the target array. Default value is ["B","G","R"] because channel order is BGR in OpenCV.
    
    Returns
    -------
    df : pandas.DataFrame
        Reverse-projected result with column
        - u , v : The x and y axis coordinates in the original array.
        - x, y, z : The latitudinal, longitudinal, and vertical coordinates in the reverse-projected coordinate system. 
        - [chnames] : The channel names passed by chnames, such as B, G, R.

    """
    if array.shape[2] != len(chnames):
        raise ValueError("The array has {} channels but chnames has length of {}. Please set chnames correctly."\
            .format(array.shape[2], len(chnames)))
    coord = persp_proj(vert, vert, ind, params, offsets)
    coord = coord[:, :, [0,2,1]] # channel: x, z, y -> x, y, z
    uv = np.meshgrid(np.arange(0,array.shape[1]), np.arange(0,array.shape[0]))
    uv = np.stack(uv, axis = 2)
    concat = np.concatenate([uv, coord, array], 2).reshape(-1, 5+array.shape[2])
    columns = ["u", "v", "x", "y", "z"]
    columns.extend(chnames)
    df = pd.DataFrame(concat, columns=columns)
    df[["u", "v"]] = df[["u", "v"]].astype("int16")
    df = df[df["x"] > 0]
    if offsets is not None:
        df["x"] += offsets[0]
        df["y"] += offsets[2]
        df["z"] += offsets[1]
    return df

def to_geotiff(df, output_path, resolution=1.0, *, crs,
               bands=["R", "G", "B"], interpolate=True, max_dist=1.0, agg_func="mean",
               nodata=255):
    """
    Convert reverse_proj output DataFrame to GeoTIFF.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from reverse_proj() containing x, y coordinates and band values.
    output_path : str
        Output file path for the GeoTIFF.
    resolution : float, default 1.0
        Pixel resolution in the same unit as the coordinate system.
    crs : str
        Coordinate Reference System (e.g., "EPSG:6690"). Must match the CRS
        of the DSM and aerial photograph used for surface generation.
    bands : list of str, default ["R", "G", "B"]
        Column names in df to use as raster bands.
    interpolate : bool, default True
        Whether to interpolate missing values using focal statistics.
    max_dist : float, default 1.0
        Maximum distance (in coordinate units) for interpolation.
    agg_func : str, default "mean"
        Aggregation function for rasterization and interpolation.
        Options: "mean", "median", "max", "min".
    nodata : int, default 255
        NoData value for occluded/missing pixels.

    Returns
    -------
    None
        Writes GeoTIFF to output_path.

    Examples
    --------
    >>> georectified = reverse_proj(original, vert, ind, params_optim, offsets)
    >>> to_geotiff(georectified, "output.tif", resolution=1.0, crs="EPSG:6690")
    """
    # Validate bands exist in DataFrame
    for band in bands:
        if band not in df.columns:
            raise ValueError(f"Band '{band}' not found in DataFrame columns: {list(df.columns)}")

    # Calculate raster extent
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    # Calculate raster dimensions
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster dimensions: width={width}, height={height}")

    # Create transform (note: y is inverted for raster coordinates)
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Convert coordinates to pixel indices
    df = df.copy()
    df["col"] = ((df["x"] - x_min) / resolution).astype(int).clip(0, width - 1)
    df["row"] = ((y_max - df["y"]) / resolution).astype(int).clip(0, height - 1)  # y inverted

    # Aggregation function mapping
    agg_funcs = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
    }
    if agg_func not in agg_funcs:
        raise ValueError(f"agg_func must be one of {list(agg_funcs.keys())}")
    func = agg_funcs[agg_func]

    # Initialize raster arrays
    raster_data = np.full((len(bands), height, width), np.nan, dtype=np.float32)

    # Rasterize each band
    for band_idx, band in enumerate(bands):
        # Group by pixel and aggregate
        grouped = df.groupby(["row", "col"])[band].agg(agg_func).reset_index()
        rows = grouped["row"].values
        cols = grouped["col"].values
        values = grouped[band].values
        raster_data[band_idx, rows, cols] = values

    # Interpolate missing values if requested
    if interpolate and max_dist > 0:
        iterations = int(np.ceil(max_dist / resolution))
        for band_idx in range(len(bands)):
            for _ in range(iterations):
                band_data = raster_data[band_idx]
                # Only fill NaN values
                mask = np.isnan(band_data)
                if not mask.any():
                    break
                filled = generic_filter(
                    band_data,
                    lambda x: func(x) if not np.all(np.isnan(x)) else np.nan,
                    size=3,
                    mode='constant',
                    cval=np.nan
                )
                band_data[mask] = filled[mask]
                raster_data[band_idx] = band_data

    # Convert to uint8 for typical RGB output
    # Replace NaN with nodata value
    nan_mask = np.isnan(raster_data)
    raster_data = np.clip(np.nan_to_num(raster_data, nan=0), 0, 255).astype(np.uint8)
    raster_data[nan_mask] = nodata

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=len(bands),
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for band_idx in range(len(bands)):
            dst.write(raster_data[band_idx], band_idx + 1)

    print(f"GeoTIFF saved to {output_path} ({width}x{height} pixels, {len(bands)} bands, nodata={nodata})")
