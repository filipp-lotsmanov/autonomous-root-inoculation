
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import argparse
from datetime import datetime
import json

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.extend([
    str(SCRIPT_DIR),
    str(PROJECT_ROOT / "controllers" / "pid"),
])

# Import system modules
import system_config as cfg
from spatial_transform import SpatialTransformationEngine, GeometricParameters
from inoculation_orchestrator import build_orchestrator
from pid_controller import ThreeAxisPIDController


class SystemBenchmark:
    """
    Comprehensive benchmarking suite for autonomous inoculation system.
    
    Measures:
    - Positioning accuracy (drop location vs target)
    - Success rate (percentage of successful inoculations)
    - Execution time (speed metrics)
    - Consistency (variation across runs)
    """
    
    def __init__(self, num_runs: int = 10, output_dir: str = None):
        """
        Initialize benchmarking suite.
        
        Args:
            num_runs: Number of benchmark runs to execute
            output_dir: Directory for benchmark results
        """
        self.num_runs = num_runs
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_runs = []
        self.aggregate_stats = {}

        print("Configuration:")
        print(f"  Number of runs: {num_runs}")
        print(f"  Output directory: {self.output_dir}")
    
    def execute_benchmark_suite(self):
        """Run complete benchmark suite."""
        print(f" Starting benchmark suite: {self.num_runs} runs")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        benchmark_start = time.time()
        
        for run_number in range(1, self.num_runs + 1):
            print(f"BENCHMARK RUN {run_number}/{self.num_runs}")
            
            try:
                run_results = self._execute_single_run(run_number)
                if run_results is not None:
                    self.all_runs.append(run_results)
                    print(f"Run {run_number} completed")
                else:
                    print(f"Run {run_number} failed - skipping")
            
            except Exception as e:
                print(f" Run {run_number} error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Brief pause between runs
            if run_number < self.num_runs:
                time.sleep(2)
        
        total_time = time.time() - benchmark_start
        
        # Analyze results
        print(f"Total benchmark time: {total_time:.1f} seconds")
        print(f"Successful runs: {len(self.all_runs)}/{self.num_runs}")
        
        if len(self.all_runs) > 0:
            self._compute_aggregate_statistics()
            self._save_benchmark_results()
        else:
            print(" No successful runs - cannot generate analysis")
    
    def _execute_single_run(self, run_id: int) -> dict:
        """Execute single benchmark run."""
        from sim_class import Simulation
        import pybullet as p
        import os
        
        run_start = time.time()
        
        # Initialize simulation
        original_dir = Path.cwd()
        os.chdir(PROJECT_ROOT)
        
        try:
            sim = Simulation(num_agents=1, render=False)  # No rendering for speed
            os.chdir(original_dir)
            
            robot_id = sim.robotIds[0]
            specimen_id = sim.specimenIds[0]
            
            # Get specimen position
            specimen_pos_tuple, _ = p.getBasePositionAndOrientation(specimen_id)
            specimen_pos = np.array(specimen_pos_tuple)
            
            # Get loaded texture
            loaded_texture = None
            if hasattr(sim, 'get_plate_image'):
                loaded_texture = sim.get_plate_image()
                plate_name = Path(loaded_texture).name if loaded_texture else f"run_{run_id}"
            else:
                plate_name = f"run_{run_id}"
            
            print(f"  Plate: {plate_name}")
            print(f"  Specimen position: {specimen_pos}")
            
            # Setup transformer
            geo_params = GeometricParameters(
                robot_workspace_center=tuple(specimen_pos),
                physical_dish_diameter=cfg.DISH_PHYSICAL_DIAMETER,
                pixel_dimensions=cfg.CAPTURED_IMAGE_DIMENSIONS,
                dish_coverage_ratio=cfg.DISH_FILL_PROPORTION,
                optical_axis_angle=cfg.OPTICAL_AXIS_ORIENTATION,
                texture_flip_enabled=cfg.TEXTURE_ORIENTATION_CORRECTION,
                drop_altitude=cfg.INOCULATION_ALTITUDE
            )
            transformer = SpatialTransformationEngine(geo_params)
            
            # Setup controller
            pid_controller = ThreeAxisPIDController(
                gains_x=(cfg.CONTROL_GAINS['xy_proportional'], cfg.CONTROL_GAINS['xy_integral'], cfg.CONTROL_GAINS['xy_derivative']),
                gains_y=(cfg.CONTROL_GAINS['xy_proportional'], cfg.CONTROL_GAINS['xy_integral'], cfg.CONTROL_GAINS['xy_derivative']),
                gains_z=(cfg.CONTROL_GAINS['z_proportional'], cfg.CONTROL_GAINS['z_integral'], cfg.CONTROL_GAINS['z_derivative']),
                output_limits={'x': (-cfg.VELOCITY_CONSTRAINTS['xy_maximum'], cfg.VELOCITY_CONSTRAINTS['xy_maximum']),
                             'y': (-cfg.VELOCITY_CONSTRAINTS['xy_maximum'], cfg.VELOCITY_CONSTRAINTS['xy_maximum']),
                             'z': (-cfg.VELOCITY_CONSTRAINTS['z_maximum'], cfg.VELOCITY_CONSTRAINTS['z_maximum'])},
                integral_limits={'x': (-cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit'], cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit']),
                               'y': (-cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit'], cfg.INTEGRAL_ACCUMULATION_LIMITS['xy_limit']),
                               'z': (-cfg.INTEGRAL_ACCUMULATION_LIMITS['z_limit'], cfg.INTEGRAL_ACCUMULATION_LIMITS['z_limit'])}
            )
            
            orchestrator = build_orchestrator(transformer, sim, pid_controller, cfg)
            
            # Filter CSV to current plate
            full_csv = pd.read_csv(cfg.ROOT_TIP_COORDINATES_CSV)
            if loaded_texture:
                plate_data = full_csv[full_csv['Image'] == plate_name].copy()
            else:
                # Use first available plate
                first_plate = full_csv['Image'].unique()[0]
                plate_data = full_csv[full_csv['Image'] == first_plate].copy()
            
            # Save temporary CSV
            temp_csv = self.output_dir / f'temp_run_{run_id}.csv'
            plate_data.to_csv(temp_csv, index=False)
            
            # Execute inoculation
            execution_start = time.time()
            results = orchestrator.execute_inoculation_sequence(str(temp_csv), robot_id)
            execution_time = time.time() - execution_start
            
            # Cleanup
            sim.close()
            if temp_csv.exists():
                temp_csv.unlink()
            
            # Compile run statistics
            run_stats = self._extract_run_statistics(results, run_id, plate_name, execution_time)
            run_stats['total_run_time'] = time.time() - run_start
            
            return run_stats
            
        except Exception as e:
            os.chdir(original_dir)
            print(f"  Error in run: {e}")
            if 'sim' in locals():
                sim.close()
            return None
    
    def _extract_run_statistics(self, results: pd.DataFrame, run_id: int, 
                                plate_name: str, exec_time: float) -> dict:
        """Extract statistics from single run."""
        successful = results[results['Success'] == True]  # noqa: E712
        
        stats = {
            'run_id': run_id,
            'plate_name': plate_name,
            'total_targets': len(results),
            'successful_targets': len(successful),
            'failed_targets': len(results) - len(successful),
            'success_rate': len(successful) / len(results) * 100 if len(results) > 0 else 0,
            'execution_time_s': exec_time,
        }
        
        if len(successful) > 0:
            errors = successful['Final Error (mm)']
            times = successful['Time (s)']
            steps = successful['Steps']
            
            stats.update({
                'mean_error_mm': errors.mean(),
                'median_error_mm': errors.median(),
                'std_error_mm': errors.std(),
                'min_error_mm': errors.min(),
                'max_error_mm': errors.max(),
                'mean_time_per_target_s': times.mean(),
                'mean_steps_per_target': steps.mean(),
                'total_steps': steps.sum()
            })
        else:
            stats.update({
                'mean_error_mm': np.nan,
                'median_error_mm': np.nan,
                'std_error_mm': np.nan,
                'min_error_mm': np.nan,
                'max_error_mm': np.nan,
                'mean_time_per_target_s': np.nan,
                'mean_steps_per_target': np.nan,
                'total_steps': 0
            })
        
        # Save detailed results for this run
        results.to_csv(self.output_dir / f'run_{run_id}_detailed.csv', index=False)
        
        return stats
    
    def _compute_aggregate_statistics(self):
        """Compute statistics across all runs."""
        df_runs = pd.DataFrame(self.all_runs)

        # Success rate analysis
        print(f"  Mean: {df_runs['success_rate'].mean():.1f}%")
        print(f"  Std:  {df_runs['success_rate'].std():.1f}%")
        print(f"  Min:  {df_runs['success_rate'].min():.1f}%")
        print(f"  Max:  {df_runs['success_rate'].max():.1f}%")
        
        # Accuracy analysis
        print(f"  Mean error across all runs: {df_runs['mean_error_mm'].mean():.3f} mm")
        print(f"  Overall std: {df_runs['mean_error_mm'].std():.3f} mm")
        print(f"  Best run: {df_runs['mean_error_mm'].min():.3f} mm")
        print(f"  Worst run: {df_runs['mean_error_mm'].max():.3f} mm")
        
        # Client requirement check
        meets_requirement = int((df_runs['mean_error_mm'] < 1.0).sum())  # Explicit int conversion
        print(f"  Runs meeting <1mm requirement: {meets_requirement}/{len(df_runs)} ({meets_requirement/len(df_runs)*100:.1f}%)")

        # Timing analysis
        print(f"  Mean time per run: {df_runs['execution_time_s'].mean():.2f} s")
        print(f"  Mean time per target: {df_runs['mean_time_per_target_s'].mean():.2f} s")
        print(f"  Mean steps per target: {df_runs['mean_steps_per_target'].mean():.0f}")

        # Store aggregate stats (with explicit type conversions for JSON)
        self.aggregate_stats = {
            'total_runs': int(len(df_runs)),
            'successful_runs': int(len(df_runs[df_runs['success_rate'] == 100])),
            'mean_success_rate': float(df_runs['success_rate'].mean()),
            'mean_positioning_error_mm': float(df_runs['mean_error_mm'].mean()),
            'std_positioning_error_mm': float(df_runs['mean_error_mm'].std()),
            'best_error_mm': float(df_runs['mean_error_mm'].min()),
            'worst_error_mm': float(df_runs['mean_error_mm'].max()),
            'mean_execution_time_s': float(df_runs['execution_time_s'].mean()),
            'mean_time_per_target_s': float(df_runs['mean_time_per_target_s'].mean()),
            'mean_steps_per_target': float(df_runs['mean_steps_per_target'].mean()),
            'runs_meeting_requirement': int(meets_requirement),
            'requirement_compliance_rate': float(meets_requirement / len(df_runs) * 100)
        }

        return df_runs

    def _save_benchmark_results(self):
        """Save all benchmark results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save summary statistics
        df_summary = pd.DataFrame(self.all_runs)
        summary_path = self.output_dir / f'benchmark_summary_{timestamp}.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f" Benchmark summary: {summary_path}")

        # Save aggregate statistics as JSON
        stats_path = self.output_dir / f'aggregate_stats_{timestamp}.json'
        with open(stats_path, 'w') as f:
            json.dump(self.aggregate_stats, f, indent=2)
        print(f" Aggregate stats: {stats_path}")

        return df_summary



def main():
    """Main benchmarking execution."""
    parser = argparse.ArgumentParser(description='Autonomous inoculation system benchmarking')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Create and run benchmark
    benchmark = SystemBenchmark(num_runs=args.runs, output_dir=args.output)
    benchmark.execute_benchmark_suite()

    print(f"\nResults saved to: {benchmark.output_dir}")
    print("\nTo visualize results, run:")
    print(f"  python visualize_benchmarks.py {benchmark.output_dir}")


if __name__ == "__main__":
    main()