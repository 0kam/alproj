import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.fill import fillnodata
from rasterio.io import MemoryFile
import numpy as np
import math
import warnings

def _get_window(raster: rasterio.DatasetReader, shooting_point: dict, distance: float):
    """
    Get window of a rectangle centered at shooting point.
    
    Parameters
    ----------
    raster : rasterio.DatasetReader
        A raster opend by rasterio.open()
    shooting_point : dict
        Shooting point. Must contain keys "x" and "y". Should be in the same coordinate reference system as the DSM.
    distance : float
        Distance from shooting point to the edge of the rectangle.
    """
    raster_directions = (raster.transform[0] > 0, raster.transform[4] > 0)
    if raster_directions[0]:
        left = shooting_point["x"] - distance
        right = shooting_point["x"] + distance
    else:
        left = shooting_point["x"] + distance
        right = shooting_point["x"] - distance
    if raster_directions[1]:
        bottom = shooting_point["y"] - distance
        top = shooting_point["y"] + distance
    else:
        bottom = shooting_point["y"] + distance
        top = shooting_point["y"] - distance
    lb = raster.index(left, bottom)
    rt = raster.index(right, top)
    min_value = min(lb[0], lb[1], rt[0], rt[1])
    if min_value < 0:
        too_large = math.ceil(abs(min_value) * raster.res[0] + 1)
        raise ValueError(f"Distance is too large. Consider using a smaller distance less than {distance - too_large} m.")
    return Window.from_slices((lb[0], rt[0]), (lb[1], rt[1]))

def _get_bounds(shooting_point: dict, distance: float):
    """
    Get bounds of a rectangle centered at shooting point.
    
    Parameters
    ----------
    shooting_point : dict
        Shooting point. Must contain keys "x" and "y". Should be in the same coordinate reference system as the DSM.
    distance : float
        Distance from shooting point to the edge of the rectangle.
    """
    left = shooting_point["x"] - distance
    right = shooting_point["x"] + distance
    bottom = shooting_point["y"] - distance
    top = shooting_point["y"] + distance
    return (left, bottom, right, top)

def _merge_rasters(aerial, dsm, bounds=None, res=1.0, resampling=Resampling.cubic_spline):
    """
    Merge two rasters in the same coordinate reference system.
    
    Parameters
    ----------
    aerial : rasterio.DatasetReader
        An aerial photograph opend by rasterio.open()
    dsm : rasterio.DatasetReader
        A Digital SurfaceModel opend by rasterio.open()
    bounds : tuple
        (left, bottom, right, top) in the same coordinate reference system.
    res : float
        Resolution of the output raster in m.
    resampling : rasterio.enums.Resampling
        Resampling method. See https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    
    Returns
    -------
    merged : numpy.ndarray
        Merged raster.
    transform : affine.Affine
        Affine transformation matrix.
    """
    if bounds is None:
        bounds = aerial.bounds
    aerial2, transform_a = merge([aerial], bounds=bounds, res=res, resampling=resampling)
    dsm2, transform_d = merge([dsm], bounds=bounds, res=res, resampling=resampling)
    aerial2[np.isnan(aerial2)] = 0
    dsm2[np.isnan(dsm2)] = 0
    if transform_a == transform_d:
        transform = transform_a
    else:
        print("error in merging aerial photo and DSM")
    return aerial2, dsm2, transform

def get_colored_surface(aerial, dsm, shooting_point, distance=2000, res=1.0, resampling=Resampling.cubic_spline, fill_dsm_dist=300):
    """
    Get colored surface.
    
    Parameters
    ----------
    aerial : rasterio.DatasetReader
        An aerial photograph opend by rasterio.open(), should be in a CRS that has units of meters, and have values of 0-255.
    dsm : rasterio.DatasetReader
        A Digital SurfaceModel opend by rasterio.open(), should be in the same CRS as the aerial.
    shooting_point : dict
        Shooting point. Must contain keys "x" and "y". Should be in the same coordinate reference system as the DSM.
    distance : float default 3000
        Distance from shooting point to the edge of the rectangle.
    res : float default 1.0
        Resolution of the output raster in m.
    resampling : rasterio.enums.Resampling default Resampling.cubic_spline
        Resampling method. See https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    fill_dsm_dist : float default 300
        Distance in m for filling nodata in DSM.

    Returns
    -------
    vert : numpy.ndarray
        Vertex coordinates.
    col : numpy.ndarray
        Vertex color.
    ind : numpy.ndarray
        Index for each triangle.
    offset : numpy.ndarray
        Offset for vertex coordinates. You need to add this to vert to get the correct coordinates.
    """
    bounds = _get_bounds(shooting_point, distance=distance)
    window_dsm = _get_window(dsm, shooting_point, distance=distance)
    window_aerial = _get_window(aerial, shooting_point, distance=distance)

    with MemoryFile() as dsm_memfile, MemoryFile() as aerial_memfile:
        with dsm_memfile.open(driver="GTiff", width=window_dsm.width, height=window_dsm.height, count=1, dtype=dsm.dtypes[0], crs=dsm.crs, transform=dsm.window_transform(window_dsm)) as dst:
            dst.write(dsm.read(window=window_dsm))
        dsm2 = dsm_memfile.open()
        dsm_max_height = dsm2.read().max()
        with aerial_memfile.open(driver="GTiff", width=window_aerial.width, height=window_aerial.height, count=3, dtype=aerial.dtypes[0], crs=aerial.crs, transform=aerial.window_transform(window_aerial)) as dst:
            dst.write(aerial.read(window=window_aerial)[:3,:,:])
        aerial2 = aerial_memfile.open()
        aerial2, dsm2, transform = _merge_rasters(aerial2, dsm2, res=res, resampling=resampling)
    
    dsm_mask = dsm2 > 0
    dsm2 = fillnodata(dsm2, dsm_mask, max_search_distance=math.ceil(fill_dsm_dist*res))
    if dsm2.min() < 0:
        warnings.warn("DSM still has negative elevation values. Consider using a larger fill_dsm_dist. Negative values will be filled with 0.")
    dsm2[dsm2 < 0] = 0
    dsm2[dsm2 > dsm_max_height] = dsm_max_height
    # Get colored surface
    # Coordinates
    x = np.arange(0, dsm2.shape[2])  * transform[0] + transform[2]
    y = np.arange(0, dsm2.shape[1])  * transform[4] + transform[5]
    xx, yy = np.meshgrid(x, y)
    w = xx.shape[0]
    h = xx.shape[1]
    zz = np.squeeze(dsm2)
    # RGB
    R = aerial2[0,:,:]
    G = aerial2[1,:,:]
    B = aerial2[2,:,:]
    vert = np.vstack((xx,zz,yy)).reshape([3, -1])
    vert = np.transpose(vert)
    col = np.vstack((R,G,B)).reshape([3, -1]) / 255
    col[col>1] = 1
    col[col<0] = 0
    col = np.transpose(col)
    # Vertex index for each triangle
    ai = np.arange(0, w-1)
    aj = np.arange(0, h-1)
    aii, ajj = np.meshgrid(ai, aj)
    a = aii + ajj * h
    a = a.flatten()
    ind = np.vstack((a, a + h, a + h + 1, a, a + h + 1, a + 1))
    ind = np.transpose(ind).reshape([-1, 3])
    if ind.max() <= (vert.shape[0] - 1):
        warnings.warn("Some triangles are outside the bounds of the raster. Consider using a smaller distance.")
    assert vert.shape == col.shape
    offsets = vert.min(axis=0)
    return vert-offsets, col, ind, offsets # vertex coordinates, vertex color, index for each triangle, offset