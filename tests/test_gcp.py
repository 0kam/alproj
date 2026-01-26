"""Unit tests for alproj.gcp module."""

import numpy as np
import pandas as pd
import pytest

from alproj.gcp import _filter_spatial, filter_gcp_distance


class TestFilterSpatial:
    """Tests for _filter_spatial function."""

    def test_basic_first_selection(self):
        """Test basic grid-based thinning with 'first' selection."""
        pts = np.array([
            [10, 10],   # cell (0, 0)
            [20, 20],   # cell (0, 0) - should be filtered out
            [110, 10],  # cell (1, 0)
            [210, 110], # cell (2, 1)
        ])
        mask = _filter_spatial(pts, grid_size=100, image_size=(300, 200), selection="first")

        assert mask.sum() == 3  # 3 unique cells
        assert mask[0]  # First point in cell (0,0) is kept
        assert not mask[1]  # Second point in same cell is filtered
        assert mask[2]
        assert mask[3]

    def test_center_selection(self):
        """Test 'center' selection picks point closest to cell center."""
        pts = np.array([
            [10, 10],   # cell (0, 0), distance to center (50, 50) = ~56.57
            [45, 55],   # cell (0, 0), distance to center (50, 50) = ~7.07 - closest
            [90, 90],   # cell (0, 0), distance to center (50, 50) = ~56.57
        ])
        mask = _filter_spatial(pts, grid_size=100, image_size=(100, 100), selection="center")

        assert mask.sum() == 1
        assert mask[1]  # Point closest to center

    def test_random_selection_reproducibility(self):
        """Test 'random' selection with random_state is reproducible."""
        pts = np.array([
            [10, 10],
            [20, 20],
            [30, 30],
        ])

        mask1 = _filter_spatial(pts, grid_size=100, image_size=(100, 100),
                                selection="random", random_state=42)
        mask2 = _filter_spatial(pts, grid_size=100, image_size=(100, 100),
                                selection="random", random_state=42)

        np.testing.assert_array_equal(mask1, mask2)

    def test_empty_input(self):
        """Test with empty input array."""
        pts = np.array([]).reshape(0, 2)
        mask = _filter_spatial(pts, grid_size=100, image_size=(100, 100))

        assert len(mask) == 0

    def test_invalid_grid_size(self):
        """Test that grid_size <= 0 raises ValueError."""
        pts = np.array([[10, 10]])

        with pytest.raises(ValueError, match="grid_size must be positive"):
            _filter_spatial(pts, grid_size=0, image_size=(100, 100))

        with pytest.raises(ValueError, match="grid_size must be positive"):
            _filter_spatial(pts, grid_size=-10, image_size=(100, 100))

    def test_unknown_selection(self):
        """Test that unknown selection method raises ValueError."""
        pts = np.array([[10, 10]])

        with pytest.raises(ValueError, match="Unknown selection"):
            _filter_spatial(pts, grid_size=100, image_size=(100, 100), selection="unknown")

    def test_single_point(self):
        """Test with single point."""
        pts = np.array([[50, 50]])
        mask = _filter_spatial(pts, grid_size=100, image_size=(100, 100))

        assert mask.sum() == 1
        assert mask[0]

    def test_center_selection_tie_breaks_by_index(self):
        """Test that center selection uses first index for ties."""
        # Two points equidistant from center (50, 50)
        pts = np.array([
            [40, 50],  # distance = 10, first index
            [60, 50],  # distance = 10, second index
        ])
        mask = _filter_spatial(pts, grid_size=100, image_size=(100, 100), selection="center")

        assert mask.sum() == 1
        assert mask[0]  # First index wins on tie

    def test_multiple_cells(self):
        """Test with points spread across multiple cells."""
        pts = np.array([
            [50, 50],    # cell (0, 0)
            [150, 50],   # cell (1, 0)
            [250, 50],   # cell (2, 0)
            [50, 150],   # cell (0, 1)
            [150, 150],  # cell (1, 1)
        ])
        mask = _filter_spatial(pts, grid_size=100, image_size=(300, 200))

        assert mask.sum() == 5  # All in different cells
        assert np.all(mask)


class TestFilterGcpDistance:
    """Tests for filter_gcp_distance function."""

    def test_basic_min_distance(self):
        """Test basic min_distance filtering."""
        gcp = pd.DataFrame({
            'u': [100, 200, 300],
            'v': [100, 200, 300],
            'x': [100, 200, 300],
            'y': [0, 0, 0],
            'z': [0, 0, 0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params, min_distance=150)

        assert len(result) == 2
        assert result['x'].tolist() == [200, 300]

    def test_basic_max_distance(self):
        """Test basic max_distance filtering."""
        gcp = pd.DataFrame({
            'u': [100, 200, 300],
            'v': [100, 200, 300],
            'x': [100, 200, 300],
            'y': [0, 0, 0],
            'z': [0, 0, 0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params, max_distance=250)

        assert len(result) == 2
        assert result['x'].tolist() == [100, 200]

    def test_min_and_max_distance(self):
        """Test combined min and max distance filtering."""
        gcp = pd.DataFrame({
            'u': [100, 200, 300, 400],
            'v': [100, 200, 300, 400],
            'x': [100, 200, 300, 400],
            'y': [0, 0, 0, 0],
            'z': [0, 0, 0, 0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params, min_distance=150, max_distance=350)

        assert len(result) == 2
        assert result['x'].tolist() == [200, 300]

    def test_3d_distance_calculation(self):
        """Test that 3D Euclidean distance is correctly calculated."""
        gcp = pd.DataFrame({
            'u': [100],
            'v': [100],
            'x': [3],  # distance = sqrt(3^2 + 4^2 + 0^2) = 5
            'y': [4],
            'z': [0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        # Point at distance exactly 5
        result = filter_gcp_distance(gcp, params, min_distance=5)
        assert len(result) == 1  # >= 5, so included

        result = filter_gcp_distance(gcp, params, min_distance=5.1)
        assert len(result) == 0  # < 5.1, so excluded

    def test_empty_input(self):
        """Test with empty GCP DataFrame."""
        gcp = pd.DataFrame(columns=['u', 'v', 'x', 'y', 'z'])
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params, min_distance=100)

        assert len(result) == 0
        assert list(result.columns) == ['u', 'v', 'x', 'y', 'z']

    def test_no_filter_returns_copy(self):
        """Test that no filtering returns a copy."""
        gcp = pd.DataFrame({
            'u': [100],
            'v': [100],
            'x': [100],
            'y': [0],
            'z': [0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params)

        assert len(result) == 1
        # Verify it's a copy, not the same object
        result.iloc[0, 0] = 999
        assert gcp.iloc[0, 0] == 100

    def test_nan_handling(self):
        """Test that rows with NaN coordinates are excluded."""
        gcp = pd.DataFrame({
            'u': [100, 200, 300],
            'v': [100, 200, 300],
            'x': [100, np.nan, 300],
            'y': [0, 0, 0],
            'z': [0, 0, 0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params, min_distance=0)

        assert len(result) == 2
        assert result['x'].tolist() == [100, 300]

    def test_missing_params_key(self):
        """Test that missing params key raises KeyError."""
        gcp = pd.DataFrame({
            'u': [100],
            'v': [100],
            'x': [100],
            'y': [0],
            'z': [0],
        })

        with pytest.raises(KeyError, match="params must contain 'x' key"):
            filter_gcp_distance(gcp, {'y': 0, 'z': 0}, min_distance=100)

        with pytest.raises(KeyError, match="params must contain 'y' key"):
            filter_gcp_distance(gcp, {'x': 0, 'z': 0}, min_distance=100)

        with pytest.raises(KeyError, match="params must contain 'z' key"):
            filter_gcp_distance(gcp, {'x': 0, 'y': 0}, min_distance=100)

    def test_negative_min_distance(self):
        """Test that negative min_distance raises ValueError."""
        gcp = pd.DataFrame({
            'u': [100],
            'v': [100],
            'x': [100],
            'y': [0],
            'z': [0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        with pytest.raises(ValueError, match="min_distance must be non-negative"):
            filter_gcp_distance(gcp, params, min_distance=-10)

    def test_max_less_than_min(self):
        """Test that max_distance < min_distance raises ValueError."""
        gcp = pd.DataFrame({
            'u': [100],
            'v': [100],
            'x': [100],
            'y': [0],
            'z': [0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        with pytest.raises(ValueError, match="max_distance must be >= min_distance"):
            filter_gcp_distance(gcp, params, min_distance=200, max_distance=100)

    def test_index_reset(self):
        """Test that result has reset index."""
        gcp = pd.DataFrame({
            'u': [100, 200, 300],
            'v': [100, 200, 300],
            'x': [100, 200, 300],
            'y': [0, 0, 0],
            'z': [0, 0, 0],
        })
        params = {'x': 0, 'y': 0, 'z': 0}

        result = filter_gcp_distance(gcp, params, min_distance=150)

        assert list(result.index) == [0, 1]  # Reset index
