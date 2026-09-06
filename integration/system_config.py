from pathlib import Path
import numpy as np


# ============================================================================
# FILE SYSTEM PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
TEXTURE_REPOSITORY = BASE_DIR / "textures" / "_plates"
INTEGRATION_OUTPUT_DIR = BASE_DIR / "Integration_pipeline" / "integration_results"

# run_pipeline.py writes its outputs relative to the directory it is launched
# from: under segmentation/ when run as the README documents, at the repo root
# when run from there. Accept either rather than hard-coding one, so results
# produced by a valid invocation are never silently invisible to the robot
# side. When both exist the most recently written wins, which makes
# regenerating the CSV take effect regardless of where it was run from.
CANDIDATE_PIPELINE_OUTPUT_DIRS = (
    BASE_DIR / "segmentation" / "pipeline_results",
    BASE_DIR / "pipeline_results",
)


def _resolve_pipeline_output_dir() -> Path:
    present = [d for d in CANDIDATE_PIPELINE_OUTPUT_DIRS
               if (d / "root_tips_pixels.csv").is_file()]
    if not present:
        return CANDIDATE_PIPELINE_OUTPUT_DIRS[0]
    return max(present, key=lambda d: (d / "root_tips_pixels.csv").stat().st_mtime)


PIPELINE_OUTPUT_DIR = _resolve_pipeline_output_dir()

# Data sources
ROOT_TIP_COORDINATES_CSV = PIPELINE_OUTPUT_DIR / "root_tips_pixels.csv"
# Same weights file segmentation/config.py loads; download from GitHub Releases.
NEURAL_NETWORK_WEIGHTS = BASE_DIR / "segmentation" / "unet_root_segmentation_256px.h5"


# ============================================================================
# PHYSICAL WORKSPACE PARAMETERS
# ============================================================================

# OT-2 platform workspace boundaries (meters)
WORKSPACE_BOUNDS = {
    'x': (0.0, 0.45),
    'y': (0.0, 0.30),
    'z': (0.0, 0.30)
}

# Specimen configuration
DISH_CENTER_POSITION = (0.1827, 0.1370, 0.0870)
DISH_PHYSICAL_DIAMETER = 0.15
INOCULATION_ALTITUDE = 0.17


# ============================================================================
# IMAGING SYSTEM PARAMETERS
# ============================================================================

# Camera configuration
CAPTURED_IMAGE_DIMENSIONS = (3006, 4202)
DISH_FILL_PROPORTION = 0.91  # Fraction of image occupied by specimen
OPTICAL_AXIS_ORIENTATION = -np.pi / 2
TEXTURE_ORIENTATION_CORRECTION = True  # Simulation-specific adjustment


# ============================================================================
# COMPUTER VISION PARAMETERS
#
# The plant geometry (positions, spacing, ROI width) lives in
# segmentation/config.py, which is the single source of truth. It was
# duplicated here under different names and read by nothing; the copies are
# removed so the two cannot drift apart.
# ============================================================================


# ============================================================================
# CONTROL SYSTEM PARAMETERS
# ============================================================================

# PID gains tuned against the PyBullet simulator (see docs/pid_controller.md)
CONTROL_GAINS = {
    'xy_proportional': 4.0,
    'xy_integral': 3.0,
    'xy_derivative': 0.8,
    'z_proportional': 6.0,
    'z_integral': 4.0,
    'z_derivative': 1.2
}

# Actuation limits
VELOCITY_CONSTRAINTS = {
    'xy_maximum': 0.25,  # m/s
    'z_maximum': 0.20    # m/s
}

# Anti-windup protection
INTEGRAL_ACCUMULATION_LIMITS = {
    'xy_limit': 0.50,
    'z_limit': 0.80
}

# Convergence criteria
POSITION_TOLERANCE = 0.001  # meters (1mm)
CONVERGENCE_CONFIRMATION_STEPS = 5  # consecutive steps required

# Simulation parameters
CONTROL_LOOP_FREQUENCY = 240.0  # Hz
CONTROL_TIMESTEP = 1.0 / CONTROL_LOOP_FREQUENCY

# ============================================================================
# COMPATIBILITY ALIASES
# ============================================================================


DT = CONTROL_TIMESTEP
TOLERANCE = POSITION_TOLERANCE
CONSECUTIVE_REQUIRED = CONVERGENCE_CONFIRMATION_STEPS

# Derived from WORKSPACE_BOUNDS rather than repeated, so the bounds cannot be
# changed in one place and silently ignored in the other.
ROBOT_WORKSPACE_X_MIN, ROBOT_WORKSPACE_X_MAX = WORKSPACE_BOUNDS['x']
ROBOT_WORKSPACE_Y_MIN, ROBOT_WORKSPACE_Y_MAX = WORKSPACE_BOUNDS['y']


# ============================================================================
# MOTION PLANNING PARAMETERS
# ============================================================================

# Navigation strategy
ENABLE_WAYPOINT_NAVIGATION = True
VERTICAL_CLEARANCE_MARGIN = 0.05  # meters

# Iteration limits
MAXIMUM_NAVIGATION_ITERATIONS = 2000
POST_ARRIVAL_SETTLING_DURATION = 30  # simulation steps
POST_DISPENSE_SETTLING_DURATION = 50  # simulation steps


# ============================================================================
# SAFETY AND VALIDATION
# ============================================================================

# Boundary checking
ENFORCE_WORKSPACE_BOUNDS = True
REQUIRE_CV_DETECTION_FLAG = True


# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

# Output configuration
ARCHIVE_MOTION_TRAJECTORIES = True
CREATE_VISUALIZATION_PLOTS = True

# Verbosity
DETAILED_LOGGING = True


# ============================================================================
# ADDITIONAL COMPATIBILITY ALIASES
# ============================================================================

SETTLE_STEPS = POST_ARRIVAL_SETTLING_DURATION
DROP_SETTLE_STEPS = POST_DISPENSE_SETTLING_DURATION
MAX_STEPS_PER_TARGET = MAXIMUM_NAVIGATION_ITERATIONS
SAFE_HEIGHT_OFFSET = VERTICAL_CLEARANCE_MARGIN
USE_WAYPOINTS = ENABLE_WAYPOINT_NAVIGATION
SAVE_TRAJECTORY_LOG = ARCHIVE_MOTION_TRAJECTORIES
REQUIRE_DETECTED_FLAG = REQUIRE_CV_DETECTION_FLAG
CHECK_WORKSPACE_BOUNDS = ENFORCE_WORKSPACE_BOUNDS
VERBOSE = DETAILED_LOGGING
OUTPUT_DIR = INTEGRATION_OUTPUT_DIR


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def verify_configuration():
    """Validate configuration parameters."""
    issues = []
    warnings = []
    
    # Path validation
    if not TEXTURE_REPOSITORY.exists():
        issues.append(f"Texture repository not found: {TEXTURE_REPOSITORY}")
    
    if not PIPELINE_OUTPUT_DIR.exists():
        issues.append(f"Pipeline output directory missing: {PIPELINE_OUTPUT_DIR}")
    
    if not ROOT_TIP_COORDINATES_CSV.exists():
        searched = "; ".join(str(d / "root_tips_pixels.csv")
                             for d in CANDIDATE_PIPELINE_OUTPUT_DIRS)
        issues.append(f"Root tip CSV not found. Searched: {searched}")
    
    # Workspace validation
    if not (WORKSPACE_BOUNDS['x'][0] <= DISH_CENTER_POSITION[0] <= WORKSPACE_BOUNDS['x'][1]):
        issues.append(f"Dish X position {DISH_CENTER_POSITION[0]} outside workspace")
    
    if not (WORKSPACE_BOUNDS['y'][0] <= DISH_CENTER_POSITION[1] <= WORKSPACE_BOUNDS['y'][1]):
        issues.append(f"Dish Y position {DISH_CENTER_POSITION[1]} outside workspace")
    
    # Control parameter validation
    all_gains = [CONTROL_GAINS[k] for k in CONTROL_GAINS]
    if any(g <= 0 for g in all_gains):
        issues.append("All control gains must be positive")
    
    if any(v <= 0 for v in VELOCITY_CONSTRAINTS.values()):
        issues.append("Velocity constraints must be positive")
    
    # Display results
    if issues:
        print(" Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print(" Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print(" Configuration validated")
    return len(issues) == 0


def display_configuration():
    """Print current configuration summary."""
    print(f"  Texture source: {TEXTURE_REPOSITORY}")
    print(f"  CSV source: {ROOT_TIP_COORDINATES_CSV.name}")
    print(f"  Output destination: {INTEGRATION_OUTPUT_DIR}")
    
    print(f"  Workspace: X={WORKSPACE_BOUNDS['x']}, Y={WORKSPACE_BOUNDS['y']}, Z={WORKSPACE_BOUNDS['z']}")
    
    print(f" Dish center: {DISH_CENTER_POSITION} m")
    print(f"  Dish diameter: {DISH_PHYSICAL_DIAMETER} m")
    print(f"  Drop altitude: {INOCULATION_ALTITUDE} m")
    
    print(f"  Image: {CAPTURED_IMAGE_DIMENSIONS[1]}x{CAPTURED_IMAGE_DIMENSIONS[0]} pixels")
    print(f"  Fill proportion: {DISH_FILL_PROPORTION:.2f}")
    print(f"  Optical angle: {np.degrees(OPTICAL_AXIS_ORIENTATION):.0f}°")
    
    print(f"  XY gains: Kp={CONTROL_GAINS['xy_proportional']}, Ki={CONTROL_GAINS['xy_integral']}, Kd={CONTROL_GAINS['xy_derivative']}")
    print(f"  Z gains: Kp={CONTROL_GAINS['z_proportional']}, Ki={CONTROL_GAINS['z_integral']}, Kd={CONTROL_GAINS['z_derivative']}")
    print(f"  Velocity limits: XY={VELOCITY_CONSTRAINTS['xy_maximum']}, Z={VELOCITY_CONSTRAINTS['z_maximum']} m/s")
    print(f"  Tolerance: {POSITION_TOLERANCE*1000:.0f} mm")


if __name__ == "__main__":
    display_configuration()
    verify_configuration()
