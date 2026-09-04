import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Path configuration for modular architecture
SCRIPT_LOCATION = Path(__file__).resolve().parent
PROJECT_BASE = SCRIPT_LOCATION.parent

# Import path setup
sys.path.extend([
    str(SCRIPT_LOCATION),
    str(PROJECT_BASE / "controllers" / "pid"),
])

# Module imports
import system_config as cfg
from spatial_transform import SpatialTransformationEngine, GeometricParameters
from inoculation_orchestrator import build_orchestrator
from pid_controller import ThreeAxisPIDController


def initialize_robotic_platform():
    """
    Establish connection to OT-2 simulation environment.
    Returns:
        Tuple of (simulation_instance, robot_id, specimen_location, texture_name)
    """
    
    try:
        from sim_class import Simulation
        import pybullet as p
        import os
        
        # Simulation expects execution from project root
        working_directory = Path.cwd()
        os.chdir(PROJECT_BASE)
        print(f" Working directory: {PROJECT_BASE}")
        
        # Initialize simulation
        simulation_instance = Simulation(num_agents=1, render=True)
        
        # Restore working directory
        os.chdir(working_directory)
        
        # Extract simulation state
        active_robot = simulation_instance.robotIds[0]
        specimen_object = simulation_instance.specimenIds[0]
        
        # Query specimen physical location
        specimen_pose, _ = p.getBasePositionAndOrientation(specimen_object)
        specimen_location = np.array(specimen_pose)
        
        # Determine loaded texture
        texture_identifier = None
        if hasattr(simulation_instance, 'get_plate_image'):
            texture_identifier = simulation_instance.get_plate_image()
            print(f" Loaded texture: {texture_identifier}")

        print(f"  Robot: {active_robot}")
        print(f"  Specimen: {specimen_object}")
        print(f"  Location: [{specimen_location[0]:.4f}, {specimen_location[1]:.4f}, {specimen_location[2]:.4f}] m")
        
        return simulation_instance, active_robot, specimen_location, texture_identifier
        
    except ImportError as error:
        print(f" Import failed: {error}")
        print("  Required: sim_class.py")
        return None, None, None, None
    except Exception as error:
        print(f"Initialization failed: {error}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def construct_control_system(simulation, config_module):
    """
    Build PID control system
    Args:
        simulation: Simulation instance
        config_module: Configuration module
        
    Returns:
        Configured ThreeAxisPIDController
    """
    controller = ThreeAxisPIDController(
        gains_x=(cfg.CONTROL_GAINS['xy_proportional'], 
                cfg.CONTROL_GAINS['xy_integral'], 
                cfg.CONTROL_GAINS['xy_derivative']),
        gains_y=(cfg.CONTROL_GAINS['xy_proportional'],
                cfg.CONTROL_GAINS['xy_integral'],
                cfg.CONTROL_GAINS['xy_derivative']),
        gains_z=(cfg.CONTROL_GAINS['z_proportional'],
                cfg.CONTROL_GAINS['z_integral'],
                cfg.CONTROL_GAINS['z_derivative']),
        output_limits={
            'x': (-cfg.VELOCITY_CONSTRAINTS['xy_maximum'], cfg.VELOCITY_CONSTRAINTS['xy_maximum']),
            'y': (-cfg.VELOCITY_CONSTRAINTS['xy_maximum'], cfg.VELOCITY_CONSTRAINTS['xy_maximum']),
            'z': (-cfg.VELOCITY_CONSTRAINTS['z_maximum'], cfg.VELOCITY_CONSTRAINTS['z_maximum'])
        },
        integral_limits={
            'x': (-cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit'], cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit']),
            'y': (-cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit'], cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit']),
            'z': (-cfg.INTEGRAL_ACCUMULATION_LIMITS['z_limit'], cfg.INTEGRAL_ACCUMULATION_LIMITS['z_limit'])
        }
    )

    print(f"  XY: Kp={cfg.CONTROL_GAINS['xy_proportional']}, "
          f"Ki={cfg.CONTROL_GAINS['xy_integral']}, "
          f"Kd={cfg.CONTROL_GAINS['xy_derivative']}")
    print(f"  Z:  Kp={cfg.CONTROL_GAINS['z_proportional']}, "
          f"Ki={cfg.CONTROL_GAINS['z_integral']}, "
          f"Kd={cfg.CONTROL_GAINS['z_derivative']}")
    
    return controller


def execute_workflow():
    """Main execution workflow."""
    
    #Configuration validation
    cfg.display_configuration()
    
    if not cfg.verify_configuration():
        print(" Configuration failed")
        return
    
    #Platform initialization
    sim, robot_id, specimen_pos, loaded_texture = initialize_robotic_platform()
    
    if sim is None:
        print(" Platform initialization failed")
        return
    
    #Determine processing target
    if loaded_texture:
        target_plate_name = Path(loaded_texture).name
        print(f" Processing plate: {target_plate_name}")
    else:
        print(" Cannot determine loaded plate")
        sim.close()
        return

    # Use actual specimen position from simulation
    actual_position = tuple(specimen_pos) if specimen_pos is not None else cfg.DISH_CENTER_POSITION
    print(f"\nSpecimen position: {actual_position}")
    
    # Build geometric parameters
    geo_params = GeometricParameters(
        robot_workspace_center=actual_position,
        physical_dish_diameter=cfg.DISH_PHYSICAL_DIAMETER,
        pixel_dimensions=cfg.CAPTURED_IMAGE_DIMENSIONS,
        dish_coverage_ratio=cfg.DISH_FILL_PROPORTION,
        optical_axis_angle=cfg.OPTICAL_AXIS_ORIENTATION,
        texture_flip_enabled=cfg.TEXTURE_ORIENTATION_CORRECTION,
        drop_altitude=cfg.INOCULATION_ALTITUDE
    )
    
    transformer = SpatialTransformationEngine(geo_params)
    
    # Validate transformation
    print("\nValidating transformation accuracy")
    transformer.verify_transformation_accuracy(test_samples=10)
    
    #Control system assembly
    pid_controller = construct_control_system(sim, cfg)
    orchestrator = build_orchestrator(transformer, sim, pid_controller, cfg)


    # Filter CSV to current plate
    print(f"\nFiltering CSV for: {target_plate_name}")
    full_dataset = pd.read_csv(cfg.ROOT_TIP_COORDINATES_CSV)
    plate_specific_data = full_dataset[full_dataset['Image'] == target_plate_name].copy()
    
    # Create temporary filtered CSV
    temp_csv_location = cfg.INTEGRATION_OUTPUT_DIR / 'active_targets.csv'
    cfg.INTEGRATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plate_specific_data.to_csv(temp_csv_location, index=False)
    
    print(f"  Full dataset: {len(full_dataset)} entries")
    print(f"  Current plate: {len(plate_specific_data)} entries")
    print(f"  Valid targets: {plate_specific_data['Detected'].sum()}")
    
    try:
        # Execute autonomous sequence
        execution_results = orchestrator.execute_inoculation_sequence(
            csv_input=str(temp_csv_location),
            robot_id=robot_id
        )
        
        print(" Execution completed")
        
        # Save plate-specific results
        result_filename = f'inoculation_{Path(target_plate_name).stem}.csv'
        result_path = cfg.INTEGRATION_OUTPUT_DIR / result_filename
        execution_results.to_csv(result_path, index=False)
        print(f" Results: {result_path}")
        
        # Generate visualization if enabled
        if cfg.CREATE_VISUALIZATION_PLOTS and len(execution_results) > 0:
            generate_performance_visualization(execution_results, transformer, target_plate_name)
        
    except KeyboardInterrupt:
        print(" Execution interrupted")
    except Exception as error:
        print(f" Execution failed: {error}")
        import traceback
        traceback.print_exc()
    finally:
        # Hold the render window open only when a human is actually watching.
        # This runs on every exit path, KeyboardInterrupt included, so an
        # unconditional input() would hang an unattended run indefinitely.
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                input("Enter to terminate")
            except (EOFError, KeyboardInterrupt):
                pass

        if sim:
            sim.close()
            print(" Platform disconnected")
        
        if temp_csv_location.exists():
            temp_csv_location.unlink()
            print(" Temporary files cleaned")


def generate_performance_visualization(results: pd.DataFrame, 
                                      transformer, plate_id: str):
    """Create performance analysis plots."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, (ax_spatial, ax_errors) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Spatial distribution plot
        ax_spatial.add_patch(Circle(
            (cfg.DISH_CENTER_POSITION[0], cfg.DISH_CENTER_POSITION[1]),
            cfg.DISH_PHYSICAL_DIAMETER / 2,
            fill=False, edgecolor='gray', linestyle='--', linewidth=2
        ))
        
        # Plot outcomes
        successes = results[results['Success']]
        failures = results[~results['Success']]
        
        if len(successes) > 0:
            ax_spatial.scatter(successes['Target X (m)'], successes['Target Y (m)'],
                             c='green', s=100, marker='o', label='Success', alpha=0.7)
        
        if len(failures) > 0:
            ax_spatial.scatter(failures['Target X (m)'], failures['Target Y (m)'],
                             c='red', s=100, marker='x', label='Failed', linewidths=3)
        
        ax_spatial.set_xlabel('X (m)')
        ax_spatial.set_ylabel('Y (m)')
        ax_spatial.set_title(f'Spatial Distribution - {Path(plate_id).stem[:30]}')
        ax_spatial.legend()
        ax_spatial.grid(True, alpha=0.3)
        ax_spatial.set_aspect('equal')
        
        # Error histogram
        if len(successes) > 0:
            error_values = successes['Final Error (mm)']
            
            ax_errors.hist(error_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax_errors.axvline(error_values.mean(), color='red', linestyle='--', 
                            label=f'Mean: {error_values.mean():.3f}mm')
            ax_errors.axvline(1.0, color='orange', linestyle=':', 
                            label='Requirement: 1.0mm')
            
            ax_errors.set_xlabel('Positioning Error (mm)')
            ax_errors.set_ylabel('Frequency')
            ax_errors.set_title('Error Distribution')
            ax_errors.legend()
            ax_errors.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        viz_filename = f'performance_{Path(plate_id).stem}.png'
        viz_path = cfg.INTEGRATION_OUTPUT_DIR / viz_filename
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f" Visualization: {viz_path}")
        
        plt.close()
        
    except ImportError:
        print("Matplotlib unavailable - skipping plots")
    except Exception as error:
        print(f"Visualization failed: {error}")


if __name__ == "__main__":
    execute_workflow()
