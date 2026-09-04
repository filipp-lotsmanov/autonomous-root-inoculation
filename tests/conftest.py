"""Shared fixtures and import-path setup.

The project is a collection of scripts rather than an installed package, and
its modules import each other by bare name (``import config``), so each
source directory has to be on sys.path for the tests to import them.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for source_dir in ("segmentation", "controllers/pid", "integration"):
    path = str(PROJECT_ROOT / source_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def geometry():
    """GeometricParameters matching the values in integration/system_config.py."""
    from spatial_transform import GeometricParameters

    return GeometricParameters(
        robot_workspace_center=(0.1827, 0.1370, 0.0870),
        physical_dish_diameter=0.15,
        pixel_dimensions=(3006, 4202),
        dish_coverage_ratio=0.91,
        optical_axis_angle=-3.141592653589793 / 2,
        texture_flip_enabled=True,
        drop_altitude=0.17,
    )
