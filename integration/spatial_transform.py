import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GeometricParameters:
    """Container for geometric transformation parameters."""
    robot_workspace_center: Tuple[float, float, float]
    physical_dish_diameter: float
    pixel_dimensions: Tuple[int, int]
    dish_coverage_ratio: float
    optical_axis_angle: float
    texture_flip_enabled: bool
    drop_altitude: float
    
    @property
    def scaling_coefficient(self) -> float:
        """Calculate pixel-to-meter conversion factor."""
        effective_pixels = min(self.pixel_dimensions) * self.dish_coverage_ratio
        return effective_pixels / self.physical_dish_diameter
    
    @property
    def image_midpoint(self) -> Tuple[float, float]:
        """Calculate pixel coordinates of image center."""
        height, width = self.pixel_dimensions
        return (width / 2.0, height / 2.0)


class SpatialTransformationEngine:
    """
    Bidirectional coordinate mapping between image and robot reference frames.
    
    Implements affine transformation with rotation, scaling, and translation
    to map between 2D pixel space and 3D robot workspace.
    """
    
    def __init__(self, parameters: GeometricParameters):
        """
        Initialize transformation engine with geometric parameters.
        Args:
            parameters: GeometricParameters instance with calibration data
        """
        self.params = parameters
        self._setup_transformation_matrices()
        self._print_initialization_summary()


    def _setup_transformation_matrices(self):
        """Pre-compute transformation matrices for efficiency."""
        # Rotation matrix for optical axis alignment
        theta = self.params.optical_axis_angle
        c, s = np.cos(theta), np.sin(theta)
        self.rotation_2d = np.array([[c, -s], [s, c]])
        self.rotation_2d_inverse = self.rotation_2d.T


    def _print_initialization_summary(self):
        """Display transformation configuration."""
        print(f"  Robot workspace center: {self.params.robot_workspace_center}")
        print(f"  Dish diameter: {self.params.physical_dish_diameter} m")
        print(f"  Pixel dimensions: {self.params.pixel_dimensions[1]}x{self.params.pixel_dimensions[0]}")
        print(f"  Scaling: {self.params.scaling_coefficient:.1f} pixels/meter")
        print(f"  Optical angle: {np.degrees(self.params.optical_axis_angle):.1f}°")
        print(f"  Texture inversion: {self.params.texture_flip_enabled}")
        print(f"  Drop altitude: {self.params.drop_altitude} m")
    
    def _normalize_to_centered_coordinates(self, pixels: np.ndarray) -> np.ndarray:
        """
        Convert from top-left origin to center-based coordinate system.
        Args:
            pixels: Nx2 array of [x, y] pixel coordinates
            
        Returns:
            Centered coordinates with flipped Y-axis
        """
        mid_x, mid_y = self.params.image_midpoint
        
        # Translate to center and flip Y (image convention to Cartesian)
        centered_x = pixels[:, 0] - mid_x
        centered_y = mid_y - pixels[:, 1]  # Flip: image Y down → world Y up
        
        return np.column_stack([centered_x, centered_y])
    
    def _apply_metric_scaling(self, centered_coords: np.ndarray) -> np.ndarray:
        """Convert pixel distances to metric distances."""
        return centered_coords / self.params.scaling_coefficient
    
    def _apply_texture_correction(self, metric_coords: np.ndarray) -> np.ndarray:
        """Apply simulation-specific texture orientation correction."""
        if self.params.texture_flip_enabled:
            return -metric_coords
        return metric_coords
    
    def _apply_optical_alignment(self, corrected_coords: np.ndarray) -> np.ndarray:
        """Rotate coordinates to align with robot's optical axis orientation."""
        return corrected_coords @ self.rotation_2d.T
    
    def _translate_to_workspace(self, aligned_coords: np.ndarray) -> np.ndarray:
        """
        Map from specimen-centered to robot workspace coordinates.
        
        Returns:
            Nx3 array with [x, y, z] in robot frame
        """
        workspace_x = self.params.robot_workspace_center[0] + aligned_coords[:, 0]
        workspace_y = self.params.robot_workspace_center[1] + aligned_coords[:, 1]
        workspace_z = np.full(len(aligned_coords), self.params.drop_altitude)
        
        return np.column_stack([workspace_x, workspace_y, workspace_z])
    
    def map_pixels_to_robot_space(self, pixel_positions: np.ndarray) -> np.ndarray:
        """
        Complete forward transformation: pixel coordinates → robot coordinates.
        
        Pipeline: pixel → centered → scaled → corrected → rotated → translated
        
        Args:
            pixel_positions: Nx2 array of [x, y] pixel coordinates
            
        Returns:
            Nx3 array of [x, y, z] robot coordinates (meters)
        """
        # Ensure proper input shape
        pixels = np.atleast_2d(np.asarray(pixel_positions))
        
        # Transformation pipeline
        centered = self._normalize_to_centered_coordinates(pixels)
        metric = self._apply_metric_scaling(centered)
        corrected = self._apply_texture_correction(metric)
        rotated = self._apply_optical_alignment(corrected)
        robot_coords = self._translate_to_workspace(rotated)
        
        return robot_coords
    
    def map_robot_to_pixel_space(self, robot_positions: np.ndarray) -> np.ndarray:
        """
        Inverse transformation: robot coordinates → pixel coordinates.
        Args:
            robot_positions: Nx3 array of [x, y, z] robot coordinates
            
        Returns:
            Nx2 array of [x, y] pixel coordinates
        """
        robots = np.atleast_2d(np.asarray(robot_positions))
        
        # Inverse pipeline (reverse order)
        # 1. Inverse translation
        centered_x = robots[:, 0] - self.params.robot_workspace_center[0]
        centered_y = robots[:, 1] - self.params.robot_workspace_center[1]
        workspace_centered = np.column_stack([centered_x, centered_y])
        
        # 2. Inverse rotation
        unrotated = workspace_centered @ self.rotation_2d_inverse.T
        
        # 3. Inverse texture correction
        if self.params.texture_flip_enabled:
            uncorrected = -unrotated
        else:
            uncorrected = unrotated
        
        # 4. Inverse scaling
        pixel_centered = uncorrected * self.params.scaling_coefficient
        
        # 5. Inverse centering and Y-flip
        mid_x, mid_y = self.params.image_midpoint
        pixel_x = pixel_centered[:, 0] + mid_x
        pixel_y = mid_y - pixel_centered[:, 1]  # Flip back
        
        return np.column_stack([pixel_x, pixel_y])
    
    def process_csv_batch(self, csv_filepath: str, 
                          save_output: Optional[str] = None,
                          x_column: str = 'Tip X (pixels)',
                          y_column: str = 'Tip Y (pixels)') -> pd.DataFrame:
        """
        Batch process CSV file with pixel coordinates.
        
        Args:
            csv_filepath: Path to input CSV
            save_output: Optional path to save transformed data
            x_column: Name of X coordinate column
            y_column: Name of Y coordinate column
            
        Returns:
            DataFrame with added robot coordinate columns
        """
        print(f" Processing coordinate batch from: {csv_filepath}")
        
        data = pd.read_csv(csv_filepath)
        print(f"  Loaded {len(data)} entries")
        
        if 'Detected' in data.columns:
            detected_count = data['Detected'].sum()
            print(f"  Valid detections: {detected_count}/{len(data)}")
        
        # Extract pixel arrays
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError(f"Missing required columns: {x_column}, {y_column}")
        
        pixel_array = data[[x_column, y_column]].values
        
        # Apply transformation
        robot_array = self.map_pixels_to_robot_space(pixel_array)
        
        # Append to dataframe
        data['Robot X (m)'] = robot_array[:, 0]
        data['Robot Y (m)'] = robot_array[:, 1]
        data['Robot Z (m)'] = robot_array[:, 2]
        
        # Calculate radial distances for validation
        radial_dist = np.sqrt(
            (robot_array[:, 0] - self.params.robot_workspace_center[0])**2 +
            (robot_array[:, 1] - self.params.robot_workspace_center[1])**2
        )
        data['Radial Distance (m)'] = radial_dist
        
        # Report statistics
        print(f"  X span: [{robot_array[:, 0].min():.4f}, {robot_array[:, 0].max():.4f}] m")
        print(f"  Y span: [{robot_array[:, 1].min():.4f}, {robot_array[:, 1].max():.4f}] m")
        print(f"  Max radial: {radial_dist.max():.4f} m (limit: {self.params.physical_dish_diameter/2:.4f} m)")
        
        # Flag outliers
        max_allowed = self.params.physical_dish_diameter / 2
        outliers = (radial_dist > max_allowed).sum()
        if outliers > 0:
            print(f"{outliers} coordinates exceed dish boundary")
        
        # Save if requested
        if save_output:
            data.to_csv(save_output, index=False)
            print(f"\nOutput saved: {save_output}")
        
        return data
    
    def verify_transformation_accuracy(self, test_samples: int = 15,
                                       seed: int = 0) -> Dict[str, float]:
        """
        Validate transformation via round-trip consistency check.
        
        Generates random pixel coordinates, transforms to robot and back,
        measures deviation from original.
        Args:
            test_samples: Number of random points to test
            seed: RNG seed. Fixed by default so the reported accuracy is
                reproducible — this is a validation check, not a sample.

        Returns:
            Dictionary with error statistics
        """
        print(f" Running round-trip validation ({test_samples} samples)")

        # Generate test points across entire image
        rng = np.random.default_rng(seed)
        test_data = rng.random((test_samples, 2))
        test_data[:, 0] *= self.params.pixel_dimensions[1]  # Width
        test_data[:, 1] *= self.params.pixel_dimensions[0]  # Height
        
        # Forward and backward transformation
        robot_space = self.map_pixels_to_robot_space(test_data)
        recovered_pixels = self.map_robot_to_pixel_space(robot_space)
        
        # Measure discrepancy
        deviations = np.linalg.norm(test_data - recovered_pixels, axis=1)
        
        stats = {
            'mean_error_px': deviations.mean(),
            'max_error_px': deviations.max(),
            'std_error_px': deviations.std(),
            'passed': deviations.max() < 0.01
        }
        
        print(f"  Round-trip deviation (pixels):")
        print(f"    Mean: {stats['mean_error_px']:.6f}")
        print(f"    Max:  {stats['max_error_px']:.6f}")
        print(f"    Std:  {stats['std_error_px']:.6f}")
        print(f"  Status: {'✓ PASS' if stats['passed'] else '✗ FAIL'}")
        
        return stats


def create_transformer_from_config(config_module) -> SpatialTransformationEngine:
    """
    Factory function to build transformer from configuration.
    Args:
        config_module: Configuration module with required parameters
        
    Returns:
        Configured SpatialTransformationEngine instance
    """
    params = GeometricParameters(
        robot_workspace_center=config_module.SPECIMEN_POSITION,
        physical_dish_diameter=config_module.SPECIMEN_DIAMETER,
        pixel_dimensions=config_module.IMAGE_SHAPE,
        dish_coverage_ratio=config_module.FILL_RATIO,
        optical_axis_angle=config_module.CAMERA_ROTATION,
        texture_flip_enabled=config_module.APPLY_180_CORRECTION,
        drop_altitude=config_module.DISPENSING_HEIGHT
    )
    
    return SpatialTransformationEngine(params)


def demonstration():
    """Demonstrate transformation with example data."""

    # Example parameters
    params = GeometricParameters(
        robot_workspace_center=(0.18, 0.14, 0.09),
        physical_dish_diameter=0.15,
        pixel_dimensions=(3006, 4202),
        dish_coverage_ratio=0.91,
        optical_axis_angle=-np.pi/2,
        texture_flip_enabled=True,
        drop_altitude=0.17
    )
    
    engine = SpatialTransformationEngine(params)
    
    # Test coordinates
    test_pixels = np.array([
        [2101, 1503],  # Image center
        [1500, 1000],  # Upper region
        [2700, 2000],  # Lower-right
    ])

    robot_coords = engine.map_pixels_to_robot_space(test_pixels)
    
    for idx, (px, rb) in enumerate(zip(test_pixels, robot_coords), 1):
        print(f"  Sample {idx}:")
        print(f"    Pixel: [{px[0]:.0f}, {px[1]:.0f}]")
        print(f"    Robot: [{rb[0]:.4f}, {rb[1]:.4f}, {rb[2]:.4f}] m")
    
    # Validation
    engine.verify_transformation_accuracy()


if __name__ == "__main__":
    demonstration()
