import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
import numpy as np
import math
import warnings

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

def _normalize_aerial(data, source_dtype, color_max=None):
    """
    Normalize aerial photo color values to [0, 1] range.

    Parameters
    ----------
    data : numpy.ndarray
        Color data array (float, after merge).
    source_dtype : numpy.dtype
        Original dtype of the aerial raster before merging.
    color_max : float or None
        Explicit maximum value for normalization. If None, determined automatically from source_dtype.

    Returns
    -------
    numpy.ndarray
        Normalized array with values clipped to [0, 1].
    """
    data = data.astype(np.float64)
    if color_max is not None:
        data /= color_max
    elif np.issubdtype(source_dtype, np.unsignedinteger):
        data /= np.iinfo(source_dtype).max
    elif np.issubdtype(source_dtype, np.signedinteger):
        data /= np.iinfo(source_dtype).max
    elif np.issubdtype(source_dtype, np.floating):
        max_val = data.max()
        if max_val <= 1.0:
            pass  # Already normalized
        elif max_val <= 255.0:
            data /= 255.0
        else:
            warnings.warn(
                f"Float aerial photo has max value {max_val:.1f} (> 255). "
                "Dividing by 255; consider passing color_max explicitly."
            )
            data /= 255.0
    else:
        data /= 255.0
    np.clip(data, 0, 1, out=data)
    return data


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
    aerial2 : numpy.ndarray
        Merged aerial raster.
    dsm2 : numpy.ndarray
        Merged DSM raster.
    transform : affine.Affine
        Affine transformation matrix.
    nodata_mask : numpy.ndarray
        2D boolean mask where True indicates nodata pixels in the DSM.
    """
    if bounds is None:
        bounds = aerial.bounds
    aerial2, transform_a = merge([aerial], bounds=bounds, res=res, resampling=resampling)
    dsm2, transform_d = merge([dsm], bounds=bounds, res=res, resampling=resampling)
    # Handle nodata for aerial (integer dtypes don't support NaN)
    if np.issubdtype(aerial2.dtype, np.floating):
        aerial2[np.isnan(aerial2)] = 0
    else:
        if aerial.nodata is not None:
            aerial2[aerial2 == aerial.nodata] = 0

    # Handle nodata for DSM
    if np.issubdtype(dsm2.dtype, np.floating):
        nodata_mask = np.isnan(dsm2[0])
        dsm2[np.isnan(dsm2)] = 0
    else:
        if dsm.nodata is not None:
            nodata_mask = (dsm2[0] == dsm.nodata)
            dsm2[dsm2 == dsm.nodata] = 0
        else:
            nodata_mask = np.zeros(dsm2.shape[1:], dtype=bool)
    if transform_a != transform_d:
        raise ValueError("Transform mismatch between aerial photo and DSM after merging.")
    transform = transform_a
    return aerial2, dsm2, transform, nodata_mask

def get_colored_surface(aerial, dsm, shooting_point, distance=2000, res=1.0, resampling=Resampling.cubic_spline, fill_dsm_dist=300, color_max=None):
    """
    Get colored surface.

    Parameters
    ----------
    aerial : rasterio.DatasetReader
        An aerial photograph opend by rasterio.open(), should be in a CRS that has units of meters.
        Supports uint8, uint16, and float32 dtypes (normalization is automatic).
    dsm : rasterio.DatasetReader
        A Digital SurfaceModel opend by rasterio.open(), should be in the same CRS as the aerial.
    shooting_point : dict
        Shooting point. Must contain keys "x" and "y". Should be in the same coordinate reference system as the DSM.
    distance : float default 2000
        Distance from shooting point to the edge of the rectangle.
    res : float default 1.0
        Resolution of the output raster in m.
    resampling : rasterio.enums.Resampling default Resampling.cubic_spline
        Resampling method. See https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    fill_dsm_dist : float default 300
        Distance in m for filling nodata in DSM.
    color_max : float or None default None
        Explicit maximum color value for normalization. If None, determined automatically from the aerial dtype.

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
    source_dtype = aerial.dtypes[0]
    bounds = _get_bounds(shooting_point, distance=distance)
    total_pixels = (2 * distance / res) ** 2
    if total_pixels > 100_000_000:
        warnings.warn(
            f"Requested area is very large ({total_pixels:.0f} pixels). "
            "Consider using a larger res or smaller distance."
        )
    aerial2, dsm2, transform, nodata_mask = _merge_rasters(
        aerial, dsm, bounds=bounds, res=res, resampling=resampling)
    aerial2 = aerial2[:3, :, :]
    dsm_max_height = dsm2[0][~nodata_mask].max() if (~nodata_mask).any() else 0

    dsm2 = fillnodata(dsm2[0], ~nodata_mask, max_search_distance=math.ceil(fill_dsm_dist / res))
    dsm2 = dsm2[np.newaxis, :, :]
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
    col = np.vstack((R,G,B)).reshape([3, -1])
    col = _normalize_aerial(col, np.dtype(source_dtype), color_max=color_max)
    col = np.transpose(col)
    # Vertex index for each triangle
    ai = np.arange(0, w-1)
    aj = np.arange(0, h-1)
    aii, ajj = np.meshgrid(ai, aj)
    a = aii + ajj * h
    a = a.flatten()
    ind = np.vstack((a, a + h, a + h + 1, a, a + h + 1, a + 1))
    ind = np.transpose(ind).reshape([-1, 3])
    # Filter out triangles that reference nodata vertices
    valid_vertex = ~nodata_mask.flatten()
    valid_tri = valid_vertex[ind].all(axis=1)
    ind = ind[valid_tri]
    if ind.size == 0:
        warnings.warn("All triangles were filtered out (all vertices are nodata).")
    elif ind.max() > (vert.shape[0] - 1):
        warnings.warn("Some triangles are outside the bounds of the raster. Consider using a smaller distance.")
    assert vert.shape == col.shape
    offsets = vert.min(axis=0)
    return vert-offsets, col, ind, offsets # vertex coordinates, vertex color, index for each triangle, offset