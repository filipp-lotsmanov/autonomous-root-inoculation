import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import time
from datetime import datetime


class MotionController:
    """
    Handles robot motion control using PID feedback.
    """
    
    def __init__(self, pid_controller, simulation_interface, config):
        """
        Initialize motion controller.
        Args:
            pid_controller: Tuned ThreeAxisPIDController instance
            simulation_interface: Robot simulation instance
            config: Configuration module
        """
        self.pid = pid_controller
        self.sim = simulation_interface
        self.cfg = config
        self.timestep = config.CONTROL_TIMESTEP
        
        # Motion state
        self.current_trajectory = []
        self.convergence_streak = 0
    
    def reset_control_state(self):
        """Clear controller state between targets."""
        self.pid.reset()
        self.current_trajectory = []
        self.convergence_streak = 0
    
    def execute_navigation_sequence(self, destination: np.ndarray, 
                                    robot_identifier: int,
                                    use_safe_approach: bool = True) -> Tuple[bool, float, List]:
        """
        Navigate robot through waypoints to final destination.
        Args:
            destination: Target position [x, y, z]
            robot_identifier: Robot ID
            use_safe_approach: Use elevated approach path
            
        Returns:
            (success_flag, final_positioning_error, position_history)
        """
        self.reset_control_state()
        
        # Build navigation waypoints
        waypoint_sequence = self._plan_approach_path(
            destination, robot_identifier, use_safe_approach
        )
        
        # Execute each waypoint
        total_iterations = 0
        for wp_index, waypoint_target in enumerate(waypoint_sequence):
            reached, iterations = self._navigate_to_waypoint(
                waypoint_target, robot_identifier, wp_index, len(waypoint_sequence)
            )
            
            total_iterations += iterations
            
            if not reached:
                return False, self._calculate_position_error(waypoint_target, robot_identifier), self.current_trajectory
        
        # Post-navigation settling
        if self.cfg.SETTLE_STEPS > 0:
            self._execute_settling_pause(robot_identifier, self.cfg.SETTLE_STEPS)
            total_iterations += self.cfg.SETTLE_STEPS
        
        # Final error measurement
        final_error = self._calculate_position_error(destination, robot_identifier)
        
        return True, final_error, self.current_trajectory
    
    def _plan_approach_path(self, target: np.ndarray, robot_id: int, 
                           safe_approach: bool) -> List[np.ndarray]:
        """Generate waypoint sequence for safe approach."""
        if not safe_approach:
            return [target]
        
        # Get current position
        current = self._get_robot_position(robot_id)
        
        # Calculate safe altitude (above both current and target)
        clearance_altitude = max(current[2], target[2]) + self.cfg.SAFE_HEIGHT_OFFSET
        
        # Waypoint 1: Above target at safe altitude
        elevated_position = target.copy()
        elevated_position[2] = clearance_altitude
        
        # Waypoint 2: Final target
        return [elevated_position, target]
    
    def _navigate_to_waypoint(self, waypoint: np.ndarray, robot_id: int,
                             wp_idx: int, total_wps: int) -> Tuple[bool, int]:
        """Navigate to single waypoint using PID control."""
        iteration_count = 0
        tolerance_streak = 0
        
        for iteration in range(self.cfg.MAX_STEPS_PER_TARGET):
            current = self._get_robot_position(robot_id)
            
            # Compute control action
            velocity_command = self.pid.update(
                tuple(current), tuple(waypoint), dt=self.timestep
            )
            
            # Execute motion
            self._apply_velocity_command(velocity_command, robot_id)
            iteration_count += 1
            
            # Track position
            self.current_trajectory.append(current.copy())
            
            # Check convergence
            position_error = np.linalg.norm(waypoint - current)
            if position_error <= self.cfg.TOLERANCE:
                tolerance_streak += 1
                if tolerance_streak >= self.cfg.CONSECUTIVE_REQUIRED:
                    if self.cfg.VERBOSE:
                        print(f"    Waypoint {wp_idx+1}/{total_wps}: Reached in {iteration+1} iterations (error: {position_error*1000:.3f} mm)")
                    return True, iteration_count
            else:
                tolerance_streak = 0
        
        # Failed to converge
        final_error = self._calculate_position_error(waypoint, robot_id)
        if self.cfg.VERBOSE:
            print(f"    Waypoint {wp_idx+1}/{total_wps}: Timeout (error: {final_error*1000:.3f} mm)")
        
        return False, iteration_count
    
    def _get_robot_position(self, robot_id: int) -> np.ndarray:
        """Query current robot position from simulation."""
        state_data = self.sim.get_states()
        position = state_data[f'robotId_{robot_id}']['pipette_position']
        return np.array(position)
    
    def _apply_velocity_command(self, velocity: Tuple, robot_id: int):
        """Send velocity command to robot."""
        vx, vy, vz = velocity
        
        # Build action array
        robot_index = self.sim.robotIds.index(robot_id)
        num_robots = len(self.sim.robotIds)
        action_array = [[0, 0, 0, 0]] * num_robots
        action_array[robot_index] = [vx, vy, vz, 0]  # 0 = no dispense
        
        self.sim.run(action_array, num_steps=1)
    
    def _calculate_position_error(self, target: np.ndarray, robot_id: int) -> float:
        """Calculate Euclidean distance to target."""
        current = self._get_robot_position(robot_id)
        return float(np.linalg.norm(target - current))
    
    def _execute_settling_pause(self, robot_id: int, duration_steps: int):
        """Hold position for settling. All robots hold, so no index is needed."""
        num_robots = len(self.sim.robotIds)
        hold_action = [[0, 0, 0, 0]] * num_robots
        self.sim.run(hold_action, num_steps=duration_steps)
    
    def trigger_liquid_dispense(self, robot_id: int):
        """
        Actuate liquid dispensing mechanism.
        
        Args:
            robot_id: Robot identifier
        """
        robot_index = self.sim.robotIds.index(robot_id)
        num_robots = len(self.sim.robotIds)
        
        # Trigger dispense
        dispense_action = [[0, 0, 0, 0]] * num_robots
        dispense_action[robot_index] = [0.0, 0.0, 0.0, 1]  # 1 = dispense
        self.sim.run(dispense_action, num_steps=1)
        
        # Allow liquid physics to settle
        settling_action = [[0, 0, 0, 0]] * num_robots
        self.sim.run(settling_action, num_steps=self.cfg.DROP_SETTLE_STEPS)
        
        if self.cfg.VERBOSE:
            print("    ✓ Dispense completed")


class AutomatedInoculationOrchestrator:
    """
    High-level orchestrator for autonomous root inoculation workflow.
    
    Coordinates: CSV loading → Transformation → Navigation → Dispensing → Logging
    """
    
    def __init__(self, spatial_transformer, motion_controller, config):
        """
        Initialize orchestrator.
        
        Args:
            spatial_transformer: SpatialTransformationEngine instance
            motion_controller: MotionController instance
            config: Configuration module
        """
        self.transformer = spatial_transformer
        self.motion_ctrl = motion_controller
        self.cfg = config
        
        # Results accumulator
        self.trajectory_archive = []
    
    def load_and_prepare_targets(self, csv_source: str) -> pd.DataFrame:
        """
        Load CSV and prepare validated target set.
        Args:
            csv_source: Path to root tips CSV
            
        Returns:
            Filtered DataFrame with robot coordinates
        """
        # Transform coordinates
        data = self.transformer.process_csv_batch(csv_source)
        
        # Apply detection filter
        if self.cfg.REQUIRE_DETECTED_FLAG and 'Detected' in data.columns:
            # Explicit == True: the flag arrives from a CSV and may parse as
            # object dtype, where truthiness would accept any non-empty string.
            valid_targets = data[data['Detected'] == True].copy()  # noqa: E712
            print(f" Filtered to {len(valid_targets)} validated targets (from {len(data)})")
        else:
            valid_targets = data.copy()
        
        # Workspace boundary validation
        if self.cfg.CHECK_WORKSPACE_BOUNDS:
            workspace_valid = (
                (valid_targets['Robot X (m)'].between(
                    self.cfg.ROBOT_WORKSPACE_X_MIN, self.cfg.ROBOT_WORKSPACE_X_MAX)) &
                (valid_targets['Robot Y (m)'].between(
                    self.cfg.ROBOT_WORKSPACE_Y_MIN, self.cfg.ROBOT_WORKSPACE_Y_MAX))
            )
            
            excluded = (~workspace_valid).sum()
            if excluded > 0:
                print(f"{excluded} targets outside workspace")
                valid_targets = valid_targets[workspace_valid].copy()
        
        print(f"Ready for execution: {len(valid_targets)} targets")
        return valid_targets
    
    def execute_inoculation_sequence(self, csv_input: str, 
                                     robot_id: int) -> pd.DataFrame:
        """
        Execute complete autonomous inoculation workflow.
        
        Workflow:
        1. Load and validate targets from CSV
        2. For each target:
           - Navigate to position
           - Dispense liquid
           - Log results
        3. Compile and save results
        
        Args:
            csv_input: Path to root tips CSV
            robot_id: Robot identifier
            
        Returns:
            DataFrame with execution results
        """
        
        # Prepare target set
        target_set = self.load_and_prepare_targets(csv_input)
        
        if len(target_set) == 0:
            print("No valid targets - aborting sequence")
            return pd.DataFrame()
        
        sequence_results = []
        total_iterations = 0
        
        # enumerate() rather than the DataFrame index: target_set has been
        # filtered by detection and workspace bounds without reindexing, so the
        # index is not a 1..N sequence.
        for target_number, (_, row) in enumerate(target_set.iterrows(), start=1):
            plant_identifier = row['Plant ID']
            
            # Extract destination
            destination = np.array([
                row['Robot X (m)'],
                row['Robot Y (m)'],
                row['Robot Z (m)']
            ])

            print(f"Target {target_number}/{len(target_set)}: {plant_identifier}")
            print(f"  Position: X={destination[0]:.4f}, Y={destination[1]:.4f}, Z={destination[2]:.4f} m")
            
            # Execute navigation
            start_time = time.time()
            success, positioning_error, path_history = self.motion_ctrl.execute_navigation_sequence(
                destination, robot_id, use_safe_approach=self.cfg.USE_WAYPOINTS
            )
            elapsed = time.time() - start_time
            
            total_iterations += len(path_history)
            
            # Dispense if successful
            if success:
                self.motion_ctrl.trigger_liquid_dispense(robot_id)
                print(f"Success, Error: {positioning_error*1000:.3f} mm, Steps: {len(path_history)}, Time: {elapsed:.2f}s")
            else:
                print(f"Failed to converge. Error: {positioning_error*1000:.3f} mm")
            
            # Record outcome
            sequence_results.append({
                'Target Number': target_number,
                'Plant ID': plant_identifier,
                'Image': row.get('Image', 'Unknown'),
                'Target X (m)': destination[0],
                'Target Y (m)': destination[1],
                'Target Z (m)': destination[2],
                'Success': success,
                'Final Error (mm)': positioning_error * 1000,
                'Steps': len(path_history),
                'Time (s)': elapsed,
                'Converged': success
            })
            
            # Archive trajectory if logging enabled
            if self.cfg.SAVE_TRAJECTORY_LOG:
                self.trajectory_archive.append({
                    'target_id': target_number,
                    'plant_id': plant_identifier,
                    'path': np.array(path_history)
                })
            
            # Brief inter-target pause
            time.sleep(0.1)
        
        # Compile results
        results_dataframe = pd.DataFrame(sequence_results)
        
        # Generate summary
        self._generate_performance_summary(results_dataframe, total_iterations)
        
        # Persist results
        self._save_execution_results(results_dataframe)
        
        return results_dataframe
    
    def _generate_performance_summary(self, results: pd.DataFrame, total_iters: int):
        """Print execution summary statistics."""

        successful_runs = results['Success'].sum()
        total_runs = len(results)
        success_percentage = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        print(f" Success Rate: {successful_runs}/{total_runs} ({success_percentage:.1f}%)")
        print(f"  Total iterations: {total_iters}")

        successful_data = results[results['Success'] == True]  # noqa: E712
        if len(successful_data) > 0:
            errors = successful_data['Final Error (mm)']
            print(f"  Mean error:   {errors.mean():.3f} mm")
            print(f"  Median error: {errors.median():.3f} mm")
            print(f"  Std dev:      {errors.std():.3f} mm")
            print(f"  Range:        [{errors.min():.3f}, {errors.max():.3f}] mm")
            
            # Requirement check
            if errors.mean() <= 1.0:
                print("Within 1mm requirement")
            else:
                print("Exceeds 1mm requirement")
        
        # Efficiency metrics
        if len(results) > 0:
            durations = results['Time (s)']
            print(f"  Mean time per target: {durations.mean():.2f} s")
            print(f"  Total execution time: {durations.sum():.2f} s")
    
    def _save_execution_results(self, results: pd.DataFrame):
        """Persist results to storage."""
        output_directory = Path(self.cfg.OUTPUT_DIR)
        output_directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = output_directory / f'execution_log_{timestamp}.csv'
        results.to_csv(results_file, index=False)
        print(f" Results saved: {results_file}")
        
        # Save trajectories if enabled
        if self.cfg.SAVE_TRAJECTORY_LOG and self.trajectory_archive:
            trajectory_file = output_directory / f'trajectories_{timestamp}.npz'
            trajectory_dict = {
                f"target_{entry['target_id']}": entry['path'] 
                for entry in self.trajectory_archive
            }
            np.savez(trajectory_file, **trajectory_dict)
            print(f" Trajectories saved: {trajectory_file}")
    

# Simplified factory function
def build_orchestrator(transformer, robot_interface, pid_controller, config):
    """
    Construct complete orchestration system.
    Args:
        transformer: Spatial transformation engine
        robot_interface: Simulation or robot interface
        pid_controller: Tuned ThreeAxisPIDController instance
        config: Configuration module
        
    Returns:
        Configured AutomatedInoculationOrchestrator
    """
    motion_ctrl = MotionController(pid_controller, robot_interface, config)
    orchestrator = AutomatedInoculationOrchestrator(transformer, motion_ctrl, config)
    
    return orchestrator


if __name__ == "__main__":
    print("Autonomous Inoculation Orchestration System")
