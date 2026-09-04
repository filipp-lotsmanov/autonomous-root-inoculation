"""Tests for the pixel <-> robot coordinate transformation.

The transform is the seam between the CV pipeline and the robot, so an error
here silently moves every target. Round-trip consistency is the main property
under test.
"""
import numpy as np
import pytest

from spatial_transform import SpatialTransformationEngine


@pytest.fixture
def engine(geometry):
    return SpatialTransformationEngine(geometry)


class TestRoundTrip:
    def test_single_point_survives_round_trip(self, engine):
        pixels = np.array([[2101.0, 1503.0]])
        recovered = engine.map_robot_to_pixel_space(
            engine.map_pixels_to_robot_space(pixels)
        )
        assert recovered == pytest.approx(pixels, abs=1e-6)

    def test_corners_and_centre_survive_round_trip(self, engine, geometry):
        height, width = geometry.pixel_dimensions
        pixels = np.array([
            [0.0, 0.0],
            [width, 0.0],
            [0.0, height],
            [width, height],
            [width / 2, height / 2],
        ])
        recovered = engine.map_robot_to_pixel_space(
            engine.map_pixels_to_robot_space(pixels)
        )
        assert recovered == pytest.approx(pixels, abs=1e-6)

    def test_builtin_validation_reports_success(self, engine):
        stats = engine.verify_transformation_accuracy(test_samples=50)
        assert stats['passed']
        assert stats['max_error_px'] < 0.01

    def test_validation_is_reproducible(self, engine):
        first = engine.verify_transformation_accuracy(test_samples=20)
        second = engine.verify_transformation_accuracy(test_samples=20)
        assert first['mean_error_px'] == second['mean_error_px']

    def test_seed_changes_the_sample(self, engine):
        a = engine.verify_transformation_accuracy(test_samples=20, seed=1)
        b = engine.verify_transformation_accuracy(test_samples=20, seed=2)
        # Both must pass, but they are different point sets.
        assert a['passed'] and b['passed']
        assert a['mean_error_px'] != b['mean_error_px']


class TestScaling:
    def test_dish_centre_maps_to_workspace_centre(self, engine, geometry):
        height, width = geometry.pixel_dimensions
        centre = np.array([[width / 2, height / 2]])
        robot = engine.map_pixels_to_robot_space(centre)
        assert robot[0, 0] == pytest.approx(geometry.robot_workspace_center[0], abs=1e-6)
        assert robot[0, 1] == pytest.approx(geometry.robot_workspace_center[1], abs=1e-6)

    def test_drop_altitude_is_applied_to_every_target(self, engine, geometry):
        pixels = np.array([[100.0, 200.0], [3000.0, 2000.0]])
        robot = engine.map_pixels_to_robot_space(pixels)
        assert robot[:, 2] == pytest.approx(geometry.drop_altitude)

    def test_pixel_distance_scales_by_documented_factor(self, engine, geometry):
        """Scale is min(pixel_dims) * coverage / physical diameter."""
        expected_px_per_m = (min(geometry.pixel_dimensions)
                             * geometry.dish_coverage_ratio
                             / geometry.physical_dish_diameter)
        a = np.array([[2101.0, 1503.0]])
        b = np.array([[2101.0 + expected_px_per_m, 1503.0]])
        ra = engine.map_pixels_to_robot_space(a)
        rb = engine.map_pixels_to_robot_space(b)
        moved = np.linalg.norm(rb[0, :2] - ra[0, :2])
        assert moved == pytest.approx(1.0, rel=1e-6)

    def test_transform_is_affine(self, engine):
        """Midpoint of two pixels maps to midpoint of their robot positions."""
        p1, p2 = np.array([[500.0, 400.0]]), np.array([[2500.0, 2400.0]])
        mid = (p1 + p2) / 2
        r1 = engine.map_pixels_to_robot_space(p1)
        r2 = engine.map_pixels_to_robot_space(p2)
        r_mid = engine.map_pixels_to_robot_space(mid)
        assert r_mid[0, :2] == pytest.approx(((r1 + r2) / 2)[0, :2], abs=1e-9)


class TestCsvBatch:
    def _write_csv(self, path, rows):
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_batch_adds_robot_columns(self, engine, tmp_path):
        csv = self._write_csv(tmp_path / "tips.csv", [
            {'Image': 'p.png', 'Plant ID': 'p_plant_1',
             'Tip X (pixels)': 1500, 'Tip Y (pixels)': 1800, 'Detected': True},
            {'Image': 'p.png', 'Plant ID': 'p_plant_2',
             'Tip X (pixels)': 2000, 'Tip Y (pixels)': 1900, 'Detected': True},
        ])
        result = engine.process_csv_batch(csv)
        for column in ('Robot X (m)', 'Robot Y (m)', 'Robot Z (m)'):
            assert column in result.columns
        assert len(result) == 2

    def test_batch_preserves_detection_flag(self, engine, tmp_path):
        csv = self._write_csv(tmp_path / "tips.csv", [
            {'Image': 'p.png', 'Plant ID': 'p_plant_1',
             'Tip X (pixels)': 1500, 'Tip Y (pixels)': 1800, 'Detected': True},
            {'Image': 'p.png', 'Plant ID': 'p_plant_2',
             'Tip X (pixels)': 0, 'Tip Y (pixels)': 0, 'Detected': False},
        ])
        result = engine.process_csv_batch(csv)
        assert result['Detected'].tolist() == [True, False]

    def test_batch_matches_direct_transform(self, engine, tmp_path):
        csv = self._write_csv(tmp_path / "tips.csv", [
            {'Image': 'p.png', 'Plant ID': 'p_plant_1',
             'Tip X (pixels)': 1234, 'Tip Y (pixels)': 2345, 'Detected': True},
        ])
        result = engine.process_csv_batch(csv)
        direct = engine.map_pixels_to_robot_space(np.array([[1234.0, 2345.0]]))
        assert result.loc[0, 'Robot X (m)'] == pytest.approx(direct[0, 0])
        assert result.loc[0, 'Robot Y (m)'] == pytest.approx(direct[0, 1])
