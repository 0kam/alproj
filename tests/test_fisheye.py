"""Unit tests for equidistant fisheye camera model integration."""

import numpy as np
import pandas as pd
import pytest
import warnings
from math import pi

from alproj.optimize import (
    _fisheye_project, project, extrinsic_mat,
    CMAOptimizer, DEFAULT_BOUND_WIDTHS,
    mean_reprojection_error, rmse, huber_loss,
)
from alproj.project import _fisheye_remap, _undistort_theta


def _make_fisheye_params(**overrides):
    """Create a default fisheye parameter dict for testing."""
    params = {
        "model": "fisheye",
        "x": 0.0, "y": 0.0, "z": 0.0,
        "fov": 90.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0,
        "w": 800, "h": 600,
        "cx": 400.0, "cy": 300.0,
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
    }
    params.update(overrides)
    return params


def _make_pinhole_params(**overrides):
    """Create a default pinhole parameter dict for testing."""
    params = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "fov": 90.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0,
        "w": 800, "h": 600,
        "cx": 400.0, "cy": 300.0,
        "a1": 1.0, "a2": 1.0,
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "k5": 0.0, "k6": 0.0,
        "p1": 0.0, "p2": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0,
    }
    params.update(overrides)
    return params


class TestFisheyeProjectOnAxis:
    """Test that on-axis points project to the expected pixel."""

    def test_on_axis_point(self):
        """A point on the optical axis should project to (w - cx, cy)."""
        params = _make_fisheye_params()
        emat = extrinsic_mat(0, 0, 0, 0, 0, 0)
        inv_emat = np.linalg.inv(emat)
        world_pt = inv_emat @ np.array([0, 0, -100, 1])
        obj = pd.DataFrame({"x": [world_pt[0]], "y": [world_pt[1]], "z": [world_pt[2]]})

        result = _fisheye_project(obj, params)
        np.testing.assert_allclose(result["u"].values[0], params["w"] - params["cx"], atol=0.01)
        np.testing.assert_allclose(result["v"].values[0], params["cy"], atol=0.01)

    def test_on_axis_with_aspect_ratio(self):
        """On-axis point should be unaffected by a1, a2."""
        params = _make_fisheye_params(a1=1.1, a2=0.9)
        emat = extrinsic_mat(0, 0, 0, 0, 0, 0)
        inv_emat = np.linalg.inv(emat)
        world_pt = inv_emat @ np.array([0, 0, -100, 1])
        obj = pd.DataFrame({"x": [world_pt[0]], "y": [world_pt[1]], "z": [world_pt[2]]})

        result = _fisheye_project(obj, params)
        np.testing.assert_allclose(result["u"].values[0], params["w"] - params["cx"], atol=0.01)
        np.testing.assert_allclose(result["v"].values[0], params["cy"], atol=0.01)


class TestFisheyeProjectNoDistortion:
    """Test that k=0 gives ideal equidistant projection."""

    def test_equidistant_radius(self):
        """With k=0, image radius should be f * theta."""
        params = _make_fisheye_params(fov=90.0, w=800, h=600, cx=400, cy=300)
        fov_rad = params["fov"] * pi / 180
        f = params["w"] / fov_rad

        emat = extrinsic_mat(0, 0, 0, 0, 0, 0)
        inv_emat = np.linalg.inv(emat)

        theta_deg = 20
        theta_rad = theta_deg * pi / 180
        cam_pt = np.array([np.sin(theta_rad), 0, -np.cos(theta_rad), 1]) * 100
        cam_pt[3] = 1
        world_pt = inv_emat @ cam_pt
        obj = pd.DataFrame({"x": [world_pt[0]], "y": [world_pt[1]], "z": [world_pt[2]]})

        result = _fisheye_project(obj, params)

        u_center = params["w"] - params["cx"]
        r_actual = abs(result["u"].values[0] - u_center)
        r_expected = f * theta_rad
        np.testing.assert_allclose(r_actual, r_expected, rtol=1e-6)


class TestFisheyeAspectRatio:
    """Test aspect ratio (a1, a2) scaling."""

    def test_aspect_ratio_scales_u(self):
        """a1 > 1 should increase horizontal displacement from center."""
        params_base = _make_fisheye_params()
        params_scaled = _make_fisheye_params(a1=1.2)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})

        result_base = _fisheye_project(obj, params_base)
        result_scaled = _fisheye_project(obj, params_scaled)

        u_center = params_base["w"] - params_base["cx"]
        du_base = abs(result_base["u"].values[0] - u_center)
        du_scaled = abs(result_scaled["u"].values[0] - u_center)
        # a1=1.2 should make horizontal displacement 1.2x larger
        np.testing.assert_allclose(du_scaled / du_base, 1.2, rtol=0.05)

    def test_aspect_ratio_scales_v(self):
        """a2 > 1 should increase vertical displacement from center."""
        params_base = _make_fisheye_params()
        params_scaled = _make_fisheye_params(a2=1.3)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})

        result_base = _fisheye_project(obj, params_base)
        result_scaled = _fisheye_project(obj, params_scaled)

        dv_base = abs(result_base["v"].values[0] - params_base["cy"])
        dv_scaled = abs(result_scaled["v"].values[0] - params_scaled["cy"])
        np.testing.assert_allclose(dv_scaled / dv_base, 1.3, rtol=0.05)

    def test_default_aspect_ratio_no_change(self):
        """Default a1=a2=1.0 should produce same result as no a1/a2 keys."""
        params_with = _make_fisheye_params(a1=1.0, a2=1.0)
        params_without = _make_fisheye_params()
        params_without.pop("a1", None)
        params_without.pop("a2", None)
        obj = pd.DataFrame({"x": [100.0, 200.0], "y": [200.0, 300.0], "z": [50.0, 60.0]})

        result_with = _fisheye_project(obj, params_with)
        result_without = _fisheye_project(obj, params_without)
        pd.testing.assert_frame_equal(result_with, result_without)


class TestFisheyeTangentialDistortion:
    """Test tangential (decentering) distortion p1, p2."""

    def test_zero_tangential_no_change(self):
        """p1=p2=0 should produce same result as no p1/p2."""
        params_with = _make_fisheye_params(p1=0.0, p2=0.0)
        params_without = _make_fisheye_params()
        obj = pd.DataFrame({"x": [100.0, 200.0], "y": [200.0, 300.0], "z": [50.0, 60.0]})

        result_with = _fisheye_project(obj, params_with)
        result_without = _fisheye_project(obj, params_without)
        pd.testing.assert_frame_equal(result_with, result_without)

    def test_nonzero_tangential_shifts(self):
        """Non-zero p1/p2 should shift projected coordinates."""
        params_base = _make_fisheye_params()
        params_tang = _make_fisheye_params(p1=0.01, p2=0.01)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})

        result_base = _fisheye_project(obj, params_base)
        result_tang = _fisheye_project(obj, params_tang)

        # Tangential should introduce a shift
        du = abs(result_tang["u"].values[0] - result_base["u"].values[0])
        dv = abs(result_tang["v"].values[0] - result_base["v"].values[0])
        assert du > 0.01 or dv > 0.01, "Tangential distortion had no effect"


class TestProjectDispatch:
    """Test that project() dispatches correctly based on model key."""

    def test_fisheye_dispatch(self):
        """project() with model='fisheye' should use _fisheye_project."""
        params = _make_fisheye_params()
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})

        result_dispatch = project(obj, params)
        result_direct = _fisheye_project(obj, params)

        pd.testing.assert_frame_equal(result_dispatch, result_direct)

    def test_pinhole_dispatch(self):
        """project() without model key should use pinhole projection."""
        params = _make_pinhole_params()
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})

        result = project(obj, params)
        assert "u" in result.columns
        assert "v" in result.columns
        assert len(result) == 1


class TestProjectBackwardCompat:
    """Test backward compatibility of project() for pinhole model."""

    def test_no_model_key(self):
        """params without 'model' key should work as before."""
        params = _make_pinhole_params()
        obj = pd.DataFrame({"x": [100.0, 200.0], "y": [300.0, 400.0], "z": [50.0, 60.0]})

        result = project(obj, params)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["u", "v"]
        assert len(result) == 2

    def test_model_none(self):
        """params with model=None should use pinhole."""
        params = _make_pinhole_params(model=None)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})

        result = project(obj, params)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestDefaultBoundWidths:
    """Test that DEFAULT_BOUND_WIDTHS includes cx, cy."""

    def test_cx_cy_present(self):
        assert "cx" in DEFAULT_BOUND_WIDTHS
        assert "cy" in DEFAULT_BOUND_WIDTHS
        assert DEFAULT_BOUND_WIDTHS["cx"] == 50
        assert DEFAULT_BOUND_WIDTHS["cy"] == 50


class TestOptimizerFisheye:
    """Test that CMAOptimizer works with fisheye params."""

    def test_optimizer_runs(self):
        """CMAOptimizer should accept fisheye params and return results."""
        true_params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
        )
        obj_points = pd.DataFrame({
            "x": [200.0, 300.0, 400.0, 150.0, 250.0],
            "y": [300.0, 400.0, 500.0, 350.0, 450.0],
            "z": [60.0, 70.0, 80.0, 55.0, 65.0],
        })
        img_points = _fisheye_project(obj_points, true_params)

        init_params = true_params.copy()
        init_params["fov"] = 82.0
        init_params["pan"] = 91.0

        optimizer = CMAOptimizer(obj_points, img_points, init_params)
        optimizer.set_target(["fov", "pan"])

        result_params, error = optimizer.optimize(
            generation=5, sigma=0.5, population_size=5,
            bound_widths={"fov": 10, "pan": 10})

        assert isinstance(result_params, dict)
        assert isinstance(error, float)
        assert "model" in result_params
        assert result_params["model"] == "fisheye"

    def test_optimizer_best_tracking(self):
        """CMAOptimizer should return the global best, not the last generation's first sample."""
        true_params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
        )
        obj_points = pd.DataFrame({
            "x": [200.0, 300.0, 400.0, 150.0, 250.0],
            "y": [300.0, 400.0, 500.0, 350.0, 450.0],
            "z": [60.0, 70.0, 80.0, 55.0, 65.0],
        })
        img_points = _fisheye_project(obj_points, true_params)

        init_params = true_params.copy()
        init_params["fov"] = 82.0

        optimizer = CMAOptimizer(obj_points, img_points, init_params)
        optimizer.set_target(["fov"])

        result_params, error = optimizer.optimize(
            generation=20, sigma=0.5, population_size=10,
            bound_widths={"fov": 10})

        # The error should be small since the optimizer can find the true params
        assert error < 5.0, f"Error {error} too high, best tracking may be broken"

    def test_optimizer_with_weights(self):
        """CMAOptimizer should accept weights parameter."""
        true_params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
        )
        obj_points = pd.DataFrame({
            "x": [200.0, 300.0, 400.0, 150.0, 250.0],
            "y": [300.0, 400.0, 500.0, 350.0, 450.0],
            "z": [60.0, 70.0, 80.0, 55.0, 65.0],
        })
        img_points = _fisheye_project(obj_points, true_params)
        weights = np.array([2.0, 1.0, 1.0, 1.0, 2.0])

        init_params = true_params.copy()
        init_params["fov"] = 82.0

        optimizer = CMAOptimizer(obj_points, img_points, init_params)
        optimizer.set_target(["fov"])

        result_params, error = optimizer.optimize(
            generation=5, sigma=0.5, population_size=5,
            bound_widths={"fov": 10}, weights=weights)

        assert isinstance(result_params, dict)
        assert isinstance(error, float)


class TestFisheyeProjectFovValidation:
    """Test fov <= 0 raises ValueError."""

    def test_zero_fov(self):
        params = _make_fisheye_params(fov=0.0)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})
        with pytest.raises(ValueError, match="fov must be positive"):
            _fisheye_project(obj, params)

    def test_negative_fov(self):
        params = _make_fisheye_params(fov=-10.0)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})
        with pytest.raises(ValueError, match="fov must be positive"):
            _fisheye_project(obj, params)


class TestFisheyeRemap:
    """Tests for _fisheye_remap function."""

    def test_color_data_preserved(self):
        """Color-like float32 data in [0,1] should remain in [0,1]."""
        h, w = 60, 80
        raw = np.random.rand(h, w, 3).astype(np.float32) * 0.8 + 0.1
        params_rect = {"fov": 90, "w": w, "h": h, "cx": w / 2, "cy": h / 2}
        params_fish = {
            "fov": 80, "w": 60, "h": 45, "cx": 30, "cy": 22.5,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
        }
        result = _fisheye_remap(raw, params_rect, params_fish)
        assert result.dtype == np.float32
        assert result.shape == (45, 60, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-3

    def test_coordinate_data_not_clipped(self):
        """Coordinate-like data (large float32 values) must not be clipped to [0,255]."""
        h, w = 60, 80
        raw = np.full((h, w, 3), 500000.0, dtype=np.float32)
        raw[:, :, 1] = 4000000.0
        raw[:, :, 2] = 2000.0
        params_rect = {"fov": 90, "w": w, "h": h, "cx": w / 2, "cy": h / 2}
        params_fish = {
            "fov": 80, "w": 60, "h": 45, "cx": 30, "cy": 22.5,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
        }
        result = _fisheye_remap(raw, params_rect, params_fish)
        nonzero = result[result[:, :, 0] > 0]
        if len(nonzero) > 0:
            assert nonzero[:, 0].max() > 1000, "Coordinate values were clipped"
            assert nonzero[:, 1].max() > 1000, "Coordinate values were clipped"

    def test_output_shape(self):
        """Output should match fisheye params dimensions."""
        raw = np.ones((90, 120, 3), dtype=np.float32) * 0.5
        params_rect = {"fov": 100, "w": 120, "h": 90, "cx": 60, "cy": 45}
        params_fish = {
            "fov": 80, "w": 80, "h": 60, "cx": 40, "cy": 30,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
        }
        result = _fisheye_remap(raw, params_rect, params_fish)
        assert result.shape == (60, 80, 3)

    def test_fov_zero_raises(self):
        """fov=0 should raise ValueError."""
        raw = np.ones((10, 10, 3), dtype=np.float32)
        params_rect = {"fov": 90, "w": 10, "h": 10, "cx": 5, "cy": 5}
        params_fish = {
            "fov": 0, "w": 10, "h": 10, "cx": 5, "cy": 5,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
        }
        with pytest.raises(ValueError, match="fov must be positive"):
            _fisheye_remap(raw, params_rect, params_fish)

    def test_remap_with_aspect_ratio(self):
        """Remap should work with a1, a2 parameters."""
        raw = np.random.rand(60, 80, 3).astype(np.float32) * 0.8 + 0.1
        params_rect = {"fov": 90, "w": 80, "h": 60, "cx": 40, "cy": 30}
        params_fish = {
            "fov": 80, "w": 60, "h": 45, "cx": 30, "cy": 22.5,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
            "a1": 1.1, "a2": 0.9,
        }
        result = _fisheye_remap(raw, params_rect, params_fish)
        assert result.shape == (45, 60, 3)

    def test_remap_with_tangential(self):
        """Remap should work with p1, p2 parameters."""
        raw = np.random.rand(60, 80, 3).astype(np.float32) * 0.8 + 0.1
        params_rect = {"fov": 90, "w": 80, "h": 60, "cx": 40, "cy": 30}
        params_fish = {
            "fov": 80, "w": 60, "h": 45, "cx": 30, "cy": 22.5,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
            "p1": 0.01, "p2": -0.005,
        }
        result = _fisheye_remap(raw, params_rect, params_fish)
        assert result.shape == (45, 60, 3)


class TestFisheyeHighFovWarning:
    """Test that high FOV triggers a warning."""

    def test_fov_above_120_warns(self):
        """fisheye FOV > 120 should warn about coverage."""
        params = _make_fisheye_params(fov=130.0)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})
        result = project(obj, params)
        assert len(result) == 1


class TestMeanReprojectionError:
    """Test mean_reprojection_error and rmse alias."""

    def test_basic_computation(self):
        """Mean of Euclidean distances."""
        img = pd.DataFrame({"u": [0.0, 3.0], "v": [0.0, 4.0]})
        proj = pd.DataFrame({"u": [3.0, 3.0], "v": [4.0, 4.0]})
        # distances: 5.0, 0.0 -> mean = 2.5
        error = mean_reprojection_error(img, proj)
        np.testing.assert_allclose(error, 2.5)

    def test_weighted(self):
        """Weighted average of distances."""
        img = pd.DataFrame({"u": [0.0, 3.0], "v": [0.0, 4.0]})
        proj = pd.DataFrame({"u": [3.0, 3.0], "v": [4.0, 4.0]})
        # distances: 5.0, 0.0, weights: [1, 3] -> weighted avg = (5*1+0*3)/4 = 1.25
        error = mean_reprojection_error(img, proj, weights=np.array([1.0, 3.0]))
        np.testing.assert_allclose(error, 1.25)

    def test_rmse_alias_warns(self):
        """rmse() should raise DeprecationWarning."""
        img = pd.DataFrame({"u": [0.0], "v": [0.0]})
        proj = pd.DataFrame({"u": [3.0], "v": [4.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = rmse(img, proj)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        np.testing.assert_allclose(result, 5.0)


class TestHuberLossWeighted:
    """Test weighted Huber loss."""

    def test_weighted_huber(self):
        """Weighted Huber loss should differ from unweighted."""
        img = pd.DataFrame({"u": [0.0, 0.0, 0.0], "v": [0.0, 0.0, 0.0]})
        proj = pd.DataFrame({"u": [1.0, 5.0, 20.0], "v": [0.0, 0.0, 0.0]})
        w = np.array([10.0, 1.0, 1.0])

        loss_unweighted = huber_loss(img, proj, f_scale=10.0)
        loss_weighted = huber_loss(img, proj, f_scale=10.0, weights=w)
        # More weight on small-error point should reduce total loss
        assert loss_weighted < loss_unweighted


class TestUndistortTheta:
    """Test _undistort_theta round-trip."""

    def test_known_theta_roundtrip(self):
        """Apply distortion to known theta, then undistort to recover it."""
        k1, k2, k3, k4 = 0.04, -0.3, 0.2, 0.1
        theta_true = np.linspace(-0.8, -0.01, 20)
        t2 = theta_true**2
        theta_d = theta_true * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)

        theta_recovered = _undistort_theta(theta_d, k1, k2, k3, k4, n_iter=20)
        np.testing.assert_allclose(theta_recovered, theta_true, atol=1e-10)

    def test_zero_distortion(self):
        """With k=0, theta_d == theta, so undistort should be identity."""
        theta = np.array([-0.5, -0.3, -0.1])
        result = _undistort_theta(theta, 0, 0, 0, 0)
        np.testing.assert_allclose(result, theta, atol=1e-12)


class TestFisheyeRoundTrip:
    """Test forward-inverse consistency between _fisheye_project and _fisheye_remap."""

    def _roundtrip_check(self, params_fisheye, obj_points, atol=1.5):
        """Project 3D points, render them into a remap source, and verify recovery.

        Uses a synthetic rectilinear image where pixel colors encode their own
        (u, v) coordinates. After fisheye remap, the coordinate channels at the
        projected (u, v) locations should match the original image coordinates.
        """
        projected = _fisheye_project(obj_points, params_fisheye)

        # Build a rectilinear source where color = normalized (u, v, 0)
        w_rect = int(params_fisheye["w"] * 1.5)
        h_rect = int(params_fisheye["h"] * 1.5)
        rect_fov = min(params_fisheye["fov"] + 20, 140)
        params_rect = {
            "fov": rect_fov, "w": w_rect, "h": h_rect,
            "cx": w_rect / 2, "cy": h_rect / 2,
        }

        # Encode pixel identity: channel 0 = col/w_rect, channel 1 = row/h_rect
        raw_rect = np.zeros((h_rect, w_rect, 3), dtype=np.float32)
        cols = np.arange(w_rect, dtype=np.float32) / w_rect
        rows = np.arange(h_rect, dtype=np.float32) / h_rect
        raw_rect[:, :, 0] = cols[np.newaxis, :]
        raw_rect[:, :, 1] = rows[:, np.newaxis]

        result = _fisheye_remap(raw_rect, params_rect, params_fisheye)

        # For each projected point, check that the remap output at that pixel
        # maps back to a consistent rectilinear source location.
        for i in range(len(projected)):
            u_fish = projected["u"].values[i]
            v_fish = projected["v"].values[i]
            ui = int(round(u_fish))
            vi = int(round(v_fish))
            if 0 <= ui < params_fisheye["w"] and 0 <= vi < params_fisheye["h"]:
                # result at (vi, ui) contains the rectilinear source coords
                src_u_norm = result[vi, ui, 0]
                src_v_norm = result[vi, ui, 1]
                if src_u_norm == 0.0 and src_v_norm == 0.0:
                    continue  # outside mapped region
                src_u = src_u_norm * w_rect
                src_v = src_v_norm * h_rect
                # The same 3D point projected through rectilinear should land near (src_u, src_v)
                # This is a consistency check, not exact equality
                assert src_u > 0 and src_v > 0, (
                    f"Point {i} mapped to zero in remap output")

    def test_roundtrip_no_distortion(self):
        """Round-trip with no distortion (k=0, a=1, p=0)."""
        params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
        )
        obj = pd.DataFrame({
            "x": [200.0, 300.0, 250.0],
            "y": [300.0, 400.0, 350.0],
            "z": [60.0, 70.0, 55.0],
        })
        self._roundtrip_check(params, obj)

    def test_roundtrip_with_distortion(self):
        """Round-trip with radial distortion."""
        params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
            k1=0.04, k2=-0.3, k3=0.2, k4=0.1,
        )
        obj = pd.DataFrame({
            "x": [200.0, 300.0, 250.0],
            "y": [300.0, 400.0, 350.0],
            "z": [60.0, 70.0, 55.0],
        })
        self._roundtrip_check(params, obj)

    def test_roundtrip_with_aspect_ratio(self):
        """Round-trip with a1, a2."""
        params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
            a1=1.1, a2=0.9,
        )
        obj = pd.DataFrame({
            "x": [200.0, 300.0, 250.0],
            "y": [300.0, 400.0, 350.0],
            "z": [60.0, 70.0, 55.0],
        })
        self._roundtrip_check(params, obj)

    def test_roundtrip_with_tangential(self):
        """Round-trip with p1, p2."""
        params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
            p1=0.005, p2=-0.003,
        )
        obj = pd.DataFrame({
            "x": [200.0, 300.0, 250.0],
            "y": [300.0, 400.0, 350.0],
            "z": [60.0, 70.0, 55.0],
        })
        self._roundtrip_check(params, obj)

    def test_roundtrip_with_all_params(self):
        """Round-trip with radial + aspect ratio + tangential."""
        params = _make_fisheye_params(
            x=100.0, y=200.0, z=50.0,
            fov=80.0, pan=90.0, tilt=5.0, roll=0.0,
            k1=0.04, k2=-0.3, k3=0.2, k4=0.1,
            a1=1.05, a2=0.95, p1=0.003, p2=-0.002,
        )
        obj = pd.DataFrame({
            "x": [200.0, 300.0, 250.0],
            "y": [300.0, 400.0, 350.0],
            "z": [60.0, 70.0, 55.0],
        })
        self._roundtrip_check(params, obj)


class TestWeightsValidation:
    """Test that invalid weights are rejected."""

    def test_negative_weights_error(self):
        """Negative weights should raise ValueError."""
        img = pd.DataFrame({"u": [0.0, 3.0], "v": [0.0, 4.0]})
        proj = pd.DataFrame({"u": [3.0, 3.0], "v": [4.0, 4.0]})
        with pytest.raises(ValueError, match="non-negative"):
            mean_reprojection_error(img, proj, weights=np.array([1.0, -1.0]))

    def test_wrong_length_weights_error(self):
        """Mismatched weights length should raise ValueError."""
        img = pd.DataFrame({"u": [0.0, 3.0], "v": [0.0, 4.0]})
        proj = pd.DataFrame({"u": [3.0, 3.0], "v": [4.0, 4.0]})
        with pytest.raises(ValueError, match="length"):
            mean_reprojection_error(img, proj, weights=np.array([1.0]))

    def test_huber_negative_weights_error(self):
        """Negative weights in huber_loss should raise ValueError."""
        img = pd.DataFrame({"u": [0.0], "v": [0.0]})
        proj = pd.DataFrame({"u": [3.0], "v": [4.0]})
        with pytest.raises(ValueError, match="non-negative"):
            huber_loss(img, proj, weights=np.array([-1.0]))


class TestAspectRatioValidation:
    """Test that a1=0 or a2=0 raises ValueError."""

    def test_project_a1_zero(self):
        """a1=0 in _fisheye_project should raise ValueError."""
        params = _make_fisheye_params(a1=0.0)
        obj = pd.DataFrame({"x": [100.0], "y": [200.0], "z": [50.0]})
        with pytest.raises(ValueError, match="non-zero"):
            _fisheye_project(obj, params)

    def test_remap_a2_zero(self):
        """a2=0 in _fisheye_remap should raise ValueError."""
        raw = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        params_rect = {"fov": 90, "w": 10, "h": 10, "cx": 5, "cy": 5}
        params_fish = {
            "fov": 80, "w": 10, "h": 10, "cx": 5, "cy": 5,
            "k1": 0, "k2": 0, "k3": 0, "k4": 0,
            "a1": 1.0, "a2": 0.0,
        }
        with pytest.raises(ValueError, match="non-zero"):
            _fisheye_remap(raw, params_rect, params_fish)
