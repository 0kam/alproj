import numpy as np
import moderngl as gl
import math
import cv2
import pandas as pd
import warnings
import copy
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
    fov_y = fov_x * h / w
    fx = 1 / math.tan(fov_x/2)
    fy = 1 / math.tan(fov_y/2)
    mat = np.array([
        fx, 0, (w-2*cx)/w, 0,
        0, fy, -(h-2*cy)/h, 0,
        0, 0, -(far+near)/(far-near), -2*far*near/(far-near),
        0, 0, -1, 0
    ])
    return mat

def modelview_mat(pan_deg, tilt_deg, roll_deg, t_x, t_y, t_z):
    """
    Makes an OpenGL-style modelview matrix from euler angles and camera location in world coordinate system.
    See https://learnopengl.com/Getting-started/Coordinate-Systems .

    Parameters
    ----------
    pan_deg : float
        Pan angle in degrees
    tilt_deg : float
        Tilt angle n degrees
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
        A modelview matrix.
    """
    pan = (360-pan_deg) * math.pi / 180
    tilt = tilt_deg * math.pi / 180
    roll = roll_deg * math.pi / 180
    rmat_x = np.array([
        [1, 0, 0, 0],
        [0, math.cos(tilt), -math.sin(tilt), 0],
        [0, math.sin(tilt), math.cos(tilt), 0],
        [0, 0, 0, 1]
    ])
    rmat_y = np.array([
        [math.cos(pan), 0, math.sin(pan), 0],
        [0, 1, 0, 0],
        [-math.sin(pan), 0, math.cos(pan), 0],
        [0, 0, 0, 1]
    ])
    rmat_z = np.array([
        [math.cos(roll), -math.sin(roll), 0, 0],
        [math.sin(roll), math.cos(roll), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rmat = np.dot(np.dot(rmat_z, rmat_x), rmat_y)
    tmat = np.array([
        [1, 0, 0, -t_x],
        [0, 1, 0, -t_z],
        [0, 0, 1, -t_y],
        [0, 0, 0, 1]
    ])
    return np.dot(rmat, tmat).transpose().flatten()

def distort(img: np.array, distort_coeffs: np.array):
    """
    Distorts an image with given distortion coefficients.

    Parameters
    ----------
    img : numpy.ndarray, shape (height, width, channels)
        An image to be distorted.
    distort_coeffs : np.array, shape (14,)
        Distortion coefficients. The order of the coefficients must be:
        a1, a2, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4.
    
    Returns
    -------
    img_distorted : numpy.ndarray
        Distorted image.
    """
    width = img.shape[1]
    height = img.shape[0]
    map_x, map_y  = np.meshgrid(np.arange(width), np.arange(height))
    grid = np.stack([map_x.flatten(), map_y.flatten()]).T
    distort_coeffs = distort_coeffs
    d = copy.copy(distort_coeffs)
    grid_d = _distort(
        grid, width, height, 
        1/d[0], 1/d[1], -d[2], -d[3], -d[4], -d[5], -d[6], -d[7],
        -d[8], -d[9], -d[10], -d[11], -d[12], -d[13]
    )

    map_d = grid_d.T.reshape([2, height, width]).astype('float32')
    img_distorted = cv2.remap(img, map_d[0,:,:], map_d[1,:,:], interpolation = cv2.INTER_NEAREST)
    
    return img_distorted

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
        a1, a2 : float
            Distortion coefficients that calibrates non-equal aspect ratio of each pixels.
        k1, k2, k3, k4, k5, k6 : float
            Radial distortion coefficients.
        p1, p2 : float
            Tangental distortion coefficients.
        s1, s2, s3, s4 : float
            Prism distortion coefficients.
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
    if params["fov"] > 90:
        warnings.warn("Wider FoV may cause redering fault. Please check the output image carefuly.")
    ctx = gl.create_standalone_context()
    ctx.enable(gl.DEPTH_TEST) # enable depth testing
    ctx.enable(gl.CULL_FACE)
    vbo = ctx.buffer(vert.astype("f4").tobytes())
    cbo = ctx.buffer(value.astype("f4").tobytes())
    ibo = ctx.buffer(ind.astype("i4").tobytes()) #vertex indecies of each triangles
    prog = ctx.program(
        vertex_shader='''
        // Vertex shader DONOT consider lens distortion
            #version 330
            precision highp float;
            in vec3 in_vert;
            in vec3 in_color;
            out vec3 v_color;
            out float v_distance;  // distance from camera
            // decrare some values used inside GPU by "uniform"
            // the real values will be later set by CPU
            uniform mat4 proj; // projection matrix
            uniform mat4 view; // model view matrix

            void main() {
                vec4 local_pos = vec4(in_vert, 1.0);
                vec4 view_pos = vec4(view * local_pos);
                gl_Position = vec4(proj * view_pos);
                v_color = in_color;
                v_distance = length(view_pos.xyz);  // Euclidean distance from camera
            }
        ''',
        fragment_shader='''
            #version 330
            precision highp float;
            in vec3 v_color;
            in float v_distance;
            uniform float min_dist;  // minimum distance threshold

            layout(location=0)out vec4 f_color;
            void main() {
                if (min_dist > 0.0 && v_distance < min_dist) {
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);  // render as black
                } else {
                    f_color = vec4(v_color, 1.0); // 1,0 added is alpha
                }
            }
        '''
        )
    
    # set some "uniform" values in prog
    proj_mat = projection_mat(params["fov"], params["w"], params["h"])
    view_mat = modelview_mat(params["pan"], params["tilt"], params["roll"], params["x"], params["y"], params["z"])
    dist_coeff = np.array([params["a1"], params["a2"], params["k1"], params["k2"], params["k3"], params["k4"], params["k5"], params["k6"], \
        params["p1"], params["p2"], params["s1"], params["s2"], params["s3"], params["s4"]])
    
    prog['proj'].value = tuple(proj_mat)
    prog['view'].value = tuple(view_mat)
    prog['min_dist'].value = float(min_distance) if min_distance is not None else 0.0
    #  pass the vertex, color, index info to the shader
    vao_content = [(vbo, "3f", "in_vert"), (cbo, "3f", "in_color")]
    vao = ctx.vertex_array(program = prog, content = vao_content, index_buffer = ibo)
    # create 2D frame
    rbo = ctx.renderbuffer((params["w"], params["h"]), dtype = "f4")
    drbo = ctx.depth_renderbuffer((params["w"], params["h"]))
    fbo = ctx.framebuffer(rbo, drbo)
    
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    # render the rgb image
    vao.render()
    # convert RAW image to ndarray, raw is a 1-dimentional array (rgbrgbrgbrgb......) 
    # array starts from right-bottom of the image, you should flip it in l-r and u-b side
    raw = np.frombuffer((fbo.read(dtype="f4")), dtype = "float32")
    raw = raw.reshape(params["h"], params["w"], 3)
    raw = np.flipud(raw)
    vao.release()
    # rbo.release()
    # drbo.release()
    fbo.release()
    ctx.release()
    vbo.release()
    cbo.release()
    ibo.release()
    prog.release()
    del(vao_content, vert, value, ind)
    raw_distorted = distort(raw, dist_coeff)

    return raw_distorted

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

def to_geotiff(df, output_path, resolution=1.0, crs="EPSG:6690",
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
    crs : str, default "EPSG:6690"
        Coordinate Reference System. Default is JGD2011 / Japan Plane Rectangular CS II.
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
