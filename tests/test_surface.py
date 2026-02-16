"""Unit tests for alproj.surface module."""

import numpy as np
import pytest
import warnings
from affine import Affine
from rasterio.io import MemoryFile

from alproj.surface import _get_bounds, _normalize_aerial, _merge_rasters, get_colored_surface


def _make_raster(data, transform, crs="EPSG:32654", nodata=None):
    """Create an in-memory rasterio DatasetReader from a numpy array.

    Parameters
    ----------
    data : numpy.ndarray
        3D array (bands, height, width).
    transform : affine.Affine
        Affine transformation.
    crs : str
        Coordinate reference system.
    nodata : float or int or None
        Nodata value for the raster.

    Returns
    -------
    memfile : rasterio.io.MemoryFile
        Keep a reference to prevent garbage collection.
    dataset : rasterio.DatasetReader
        Open dataset reader.
    """
    bands, height, width = data.shape
    kwargs = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    )
    if nodata is not None:
        kwargs["nodata"] = nodata
    memfile = MemoryFile()
    dataset = memfile.open(**kwargs)
    dataset.write(data)
    return memfile, dataset


def _make_test_rasters(
    origin_x=500.0, origin_y=600.0, size=10, res=1.0, elevation=100.0,
    rgb_value=128, aerial_dtype=np.uint8,
):
    """Create a pair of aerial + DSM rasters for testing.

    Returns memfiles (to keep alive) and dataset readers.
    """
    transform = Affine(res, 0, origin_x, 0, -res, origin_y + size * res)
    dsm_data = np.full((1, size, size), elevation, dtype=np.float32)
    aerial_data = np.full((3, size, size), rgb_value, dtype=aerial_dtype)
    mf_aerial, aerial = _make_raster(aerial_data, transform)
    mf_dsm, dsm = _make_raster(dsm_data, transform)
    return (mf_aerial, aerial), (mf_dsm, dsm)


class TestGetBounds:
    """Tests for _get_bounds."""

    def test_basic(self):
        sp = {"x": 100.0, "y": 200.0}
        bounds = _get_bounds(sp, distance=50.0)
        assert bounds == (50.0, 150.0, 150.0, 250.0)

    def test_zero_distance(self):
        sp = {"x": 0.0, "y": 0.0}
        bounds = _get_bounds(sp, distance=0.0)
        assert bounds == (0.0, 0.0, 0.0, 0.0)


class TestMergeRasters:
    """Tests for _merge_rasters."""

    def test_returns_nodata_mask(self):
        """nodata_mask should be True where DSM is NaN."""
        size = 10
        transform = Affine(1, 0, 500, 0, -1, 510)
        dsm_data = np.full((1, size, size), 100.0, dtype=np.float32)
        dsm_data[0, 0:3, 0:3] = np.nan
        aerial_data = np.full((3, size, size), 128, dtype=np.uint8)
        mf_a, aerial = _make_raster(aerial_data.astype(np.float32), transform)
        mf_d, dsm = _make_raster(dsm_data, transform)

        bounds = (500, 500, 510, 510)
        _, dsm2, _, nodata_mask = _merge_rasters(aerial, dsm, bounds=bounds, res=1.0)

        assert nodata_mask.shape == dsm2.shape[1:]
        assert nodata_mask[:3, :3].all()
        assert not nodata_mask[5, 5]
        # After merge, NaN should be replaced with 0
        assert not np.isnan(dsm2).any()

        mf_a.close()
        mf_d.close()


class TestGetColoredSurface:
    """Tests for get_colored_surface."""

    def test_output_shapes(self):
        """vert, col should have same shape; ind should be Nx3."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0
        )
        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )

        assert vert.ndim == 2 and vert.shape[1] == 3
        assert col.shape == vert.shape
        assert ind.ndim == 2 and ind.shape[1] == 3
        assert offsets.shape == (3,)
        # All colors should be in [0, 1]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

        mf_a.close()
        mf_d.close()

    def test_output_dtypes(self):
        """vert and col should be float; ind should be integer."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0
        )
        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )

        assert np.issubdtype(vert.dtype, np.floating)
        assert np.issubdtype(col.dtype, np.floating)
        assert np.issubdtype(ind.dtype, np.integer)

        mf_a.close()
        mf_d.close()

    def test_shooting_point_at_edge(self):
        """Shooting point near raster edge should not raise ValueError."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0
        )
        # Shooting point at the very corner of the raster
        sp = {"x": 500.0, "y": 500.0}
        # distance extends beyond raster - should work, nodata areas are filled/filtered
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=8, res=1.0
        )

        assert vert.shape[0] > 0
        assert ind.shape[0] > 0

        mf_a.close()
        mf_d.close()

    def test_elevation_zero_not_filtered(self):
        """Vertices at elevation 0 should not be treated as nodata."""
        size = 10
        transform = Affine(1, 0, 500, 0, -1, 510)
        dsm_data = np.zeros((1, size, size), dtype=np.float32)  # All elevation = 0
        aerial_data = np.full((3, size, size), 128, dtype=np.uint8)
        mf_a, aerial = _make_raster(aerial_data, transform)
        mf_d, dsm = _make_raster(dsm_data, transform)

        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )

        # With zero elevation and no nodata, all triangles should be present
        assert ind.shape[0] > 0
        # No triangles should have been filtered
        expected_w = 8  # 2*4/1.0 = 8 pixels, shape[0]
        expected_h = 8
        expected_tris = 2 * (expected_w - 1) * (expected_h - 1)
        assert ind.shape[0] == expected_tris

        mf_a.close()
        mf_d.close()

    def test_all_nodata_warns(self):
        """All-nodata DSM should warn about filtered triangles."""
        size = 10
        transform = Affine(1, 0, 500, 0, -1, 510)
        dsm_data = np.full((1, size, size), np.nan, dtype=np.float32)
        aerial_data = np.full((3, size, size), 128, dtype=np.uint8)
        mf_a, aerial = _make_raster(aerial_data, transform)
        mf_d, dsm = _make_raster(dsm_data, transform)

        sp = {"x": 505.0, "y": 505.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vert, col, ind, offsets = get_colored_surface(
                aerial, dsm, sp, distance=4, res=1.0
            )
            warning_messages = [str(x.message) for x in w]
            assert any("All triangles were filtered out" in msg for msg in warning_messages)

        assert ind.size == 0

        mf_a.close()
        mf_d.close()

    def test_partial_nodata_filters_correctly(self):
        """Nodata region should have its triangles removed."""
        size = 10
        transform = Affine(1, 0, 500, 0, -1, 510)
        dsm_data = np.full((1, size, size), 100.0, dtype=np.float32)
        dsm_data[0, :5, :5] = np.nan  # Top-left quadrant is nodata
        aerial_data = np.full((3, size, size), 128, dtype=np.uint8)
        mf_a, aerial = _make_raster(aerial_data, transform)
        mf_d, dsm = _make_raster(dsm_data, transform)

        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )

        # Should have some triangles, but fewer than if all data were valid
        full_w = 8
        full_h = 8
        max_tris = 2 * (full_w - 1) * (full_h - 1)
        assert 0 < ind.shape[0] < max_tris

        mf_a.close()
        mf_d.close()

    def test_integer_nodata_aerial(self):
        """Integer aerial with nodata value should have nodata replaced with 0."""
        size = 10
        transform = Affine(1, 0, 500, 0, -1, 510)
        dsm_data = np.full((1, size, size), 100.0, dtype=np.float32)
        aerial_data = np.full((3, size, size), 128, dtype=np.uint8)
        nodata_val = 0
        aerial_data[:, 0:2, 0:2] = nodata_val
        mf_a, aerial = _make_raster(aerial_data, transform, nodata=nodata_val)
        mf_d, dsm = _make_raster(dsm_data, transform)

        bounds = (500, 500, 510, 510)
        aerial2, _, _, _ = _merge_rasters(aerial, dsm, bounds=bounds, res=1.0)
        # nodata pixels should be 0
        assert (aerial2[:, 0:2, 0:2] == 0).all()

        mf_a.close()
        mf_d.close()

    def test_memory_guard_warning(self):
        """Very large area should trigger memory warning."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0
        )
        sp = {"x": 505.0, "y": 505.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # distance=10000, res=0.1 -> (2*10000/0.1)^2 = 4e10 pixels
            # Don't actually run the merge, just check the warning is raised
            # before merge happens. We'll use a smaller but still >100M case.
            try:
                get_colored_surface(aerial, dsm, sp, distance=6000, res=1.0)
            except Exception:
                pass  # May fail due to raster size, but warning should fire first
            warning_messages = [str(x.message) for x in w]
            assert any("very large" in msg for msg in warning_messages)

        mf_a.close()
        mf_d.close()


class TestNormalizeAerial:
    """Tests for _normalize_aerial."""

    def test_uint8(self):
        data = np.array([[128, 255, 0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.uint8))
        np.testing.assert_allclose(result, [[128 / 255, 1.0, 0.0]])

    def test_uint16(self):
        data = np.array([[32768, 65535, 0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.uint16))
        np.testing.assert_allclose(result, [[32768 / 65535, 1.0, 0.0]])

    def test_float_already_normalized(self):
        data = np.array([[0.5, 1.0, 0.0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.float32))
        np.testing.assert_allclose(result, [[0.5, 1.0, 0.0]])

    def test_float_0_255_range(self):
        data = np.array([[128.0, 255.0, 0.0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.float32))
        np.testing.assert_allclose(result, [[128 / 255, 1.0, 0.0]])

    def test_float_above_255_warns(self):
        data = np.array([[300.0, 500.0]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _normalize_aerial(data, np.dtype(np.float32))
            assert any("max value" in str(x.message) for x in w)
        # 300/255 > 1.0 so it gets clipped to 1.0; 500/255 > 1.0 also clipped
        np.testing.assert_allclose(result, [[1.0, 1.0]])

    def test_negative_values_clipped(self):
        data = np.array([[-10.0, 128.0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.uint8))
        assert result[0, 0] == 0.0

    def test_color_max_override(self):
        data = np.array([[500.0, 1000.0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.float32), color_max=1000.0)
        np.testing.assert_allclose(result, [[0.5, 1.0]])

    def test_signed_int(self):
        data = np.array([[16384, 32767, 0]], dtype=np.float64)
        result = _normalize_aerial(data, np.dtype(np.int16))
        np.testing.assert_allclose(result, [[16384 / 32767, 1.0, 0.0]], atol=1e-6)


class TestGetColoredSurfaceDtypes:
    """Tests for get_colored_surface with various aerial dtypes."""

    def test_uint8_backward_compat(self):
        """uint8 aerial should produce same results as before."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0,
            rgb_value=128, aerial_dtype=np.uint8,
        )
        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )
        assert col.min() >= 0.0
        assert col.max() <= 1.0
        np.testing.assert_allclose(col.max(), 128 / 255, atol=0.02)

        mf_a.close()
        mf_d.close()

    def test_uint16(self):
        """uint16 aerial should be normalized by 65535."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0,
            rgb_value=32768, aerial_dtype=np.uint16,
        )
        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )
        assert col.min() >= 0.0
        assert col.max() <= 1.0
        np.testing.assert_allclose(col.max(), 32768 / 65535, atol=0.02)

        mf_a.close()
        mf_d.close()

    def test_float32_01_range(self):
        """float32 aerial in [0, 1] should be kept as-is."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0,
            rgb_value=0.6, aerial_dtype=np.float32,
        )
        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0
        )
        assert col.min() >= 0.0
        assert col.max() <= 1.0
        np.testing.assert_allclose(col.max(), 0.6, atol=0.02)

        mf_a.close()
        mf_d.close()

    def test_color_max_parameter(self):
        """Explicit color_max should override auto-detection."""
        (mf_a, aerial), (mf_d, dsm) = _make_test_rasters(
            origin_x=500, origin_y=500, size=10, res=1.0,
            rgb_value=500, aerial_dtype=np.uint16,
        )
        sp = {"x": 505.0, "y": 505.0}
        vert, col, ind, offsets = get_colored_surface(
            aerial, dsm, sp, distance=4, res=1.0, color_max=1000.0,
        )
        assert col.min() >= 0.0
        assert col.max() <= 1.0
        np.testing.assert_allclose(col.max(), 0.5, atol=0.02)

        mf_a.close()
        mf_d.close()
