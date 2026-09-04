import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
from pid_controller import PIDController, simulator_units_to_meters
from validate_gains import VelocityRobotSimulator, calculate_metrics, run_step_response


# Starting point the grid search is compared against: the hand-tuned gains used
# before the search was run.
BASELINE_GAINS = {
    'X': (2.0, 1.0, 0.8),
    'Z': (2.0, 1.5, 0.8),
}


def optimize_gains_for_axis(axis_name: str,
                            kp_range: Tuple[float, float, int],
                            ki_range: Tuple[float, float, int],
                            kd_range: Tuple[float, float, int],
                            max_overshoot: float = 50.0,
                            max_settling_time: float = 12.0):
    print(f"\n{'=' * 70}")
    print(f"Optimizing {axis_name}-Axis PID Gains")
    print(f"{'=' * 70}")

    kp_values = np.linspace(kp_range[0], kp_range[1], kp_range[2])
    ki_values = np.linspace(ki_range[0], ki_range[1], ki_range[2])
    kd_values = np.linspace(kd_range[0], kd_range[1], kd_range[2])

    total_tests = len(kp_values) * len(ki_values) * len(kd_values)
    print(f"Testing {total_tests} gain combinations...")
    print(f"Kp: {kp_range[0]} to {kp_range[1]} ({kp_range[2]} values)")
    print(f"Ki: {ki_range[0]} to {ki_range[1]} ({ki_range[2]} values)")
    print(f"Kd: {kd_range[0]} to {kd_range[1]} ({kd_range[2]} values)")
    print(f"\nConstraints:")
    print(f"  Max overshoot: {max_overshoot}%")
    print(f"  Max settling time: {max_settling_time}s\n")

    results = []
    target = 100.0
    test_count = 0

    for kp in kp_values:
        for ki in ki_values:
            for kd in kd_values:
                test_count += 1

                pid = PIDController(kp, ki, kd,
                                    output_limits=(-200, 200),
                                    integral_limits=(-100, 100))
                sim = VelocityRobotSimulator(axis_name)

                time_data, pos_data = run_step_response(pid, sim, target, duration=15.0)
                metrics = calculate_metrics(time_data, pos_data, target)

                if (metrics['overshoot_percent'] <= max_overshoot and
                        metrics['settling_time'] <= max_settling_time):
                    error_mm = metrics['steady_state_error_meters'] * 1000

                    results.append({
                        'Kp': kp,
                        'Ki': ki,
                        'Kd': kd,
                        'error_mm': error_mm,
                        'overshoot': metrics['overshoot_percent'],
                        'settling_time': metrics['settling_time'],
                        'rise_time': metrics['rise_time'],
                        'rmse': metrics['rmse']
                    })

                if test_count % 50 == 0:
                    print(f"Progress: {test_count}/{total_tests} tests completed...")

    # Steady-state error saturates near zero for many gain combinations, so
    # ranking on it alone produces arbitrary ties. Break them on settling time
    # and then overshoot, which still discriminate at that point.
    results.sort(key=lambda x: (x['error_mm'], x['settling_time'], x['overshoot']))

    print(f"\nCompleted {total_tests} tests")
    print(f"Found {len(results)} valid configurations")

    return results


def measure_gains(axis_name: str, gains: Tuple[float, float, float],
                  duration: float = 15.0) -> Dict[str, float]:
    """Run a step response for one gain set and return its metrics."""
    pid = PIDController(gains[0], gains[1], gains[2],
                        output_limits=(-200, 200),
                        integral_limits=(-100, 100))
    sim = VelocityRobotSimulator(axis_name)
    time_data, pos_data = run_step_response(pid, sim, 100.0, duration=duration)
    metrics = calculate_metrics(time_data, pos_data, 100.0)
    metrics['error_mm'] = metrics['steady_state_error_meters'] * 1000
    return metrics


def print_top_results(results: List[Dict], axis_name: str,
                      baseline_gains: Tuple[float, float, float], top_n: int = 10):
    print(f"\n{'=' * 78}")
    print(f"Top {top_n} Gain Combinations for {axis_name}-Axis")
    print(f"{'=' * 78}")
    print(f"{'Rank':<6} {'Kp':<6} {'Ki':<6} {'Kd':<6} {'Error(mm)':<14} {'Overshoot':<12} {'Settling':<10}")
    print(f"{'-' * 78}")

    for i, result in enumerate(results[:top_n], 1):
        print(f"{i:<6} {result['Kp']:<6.2f} {result['Ki']:<6.2f} {result['Kd']:<6.2f} "
              f"{result['error_mm']:<14.5f} {result['overshoot']:<12.1f} {result['settling_time']:<10.2f}")

    if results:
        print(f"{'-' * 78}")
        best = results[0]

        # Measure the baseline under identical conditions rather than quoting a
        # stored number, so the comparison below reflects this run.
        baseline = measure_gains(axis_name, baseline_gains)

        print(f"\nBest Configuration:")
        print(f"  Kp={best['Kp']:.2f}, Ki={best['Ki']:.2f}, Kd={best['Kd']:.2f}")
        print(f"  Error: {best['error_mm']:.5f} mm")
        print(f"  Overshoot: {best['overshoot']:.1f}%")
        print(f"  Settling time: {best['settling_time']:.2f}s")
        print(f"\nBaseline Configuration (measured this run):")
        print(f"  Kp={baseline_gains[0]:.2f}, Ki={baseline_gains[1]:.2f}, Kd={baseline_gains[2]:.2f}")
        print(f"  Error: {baseline['error_mm']:.5f} mm")
        print(f"  Overshoot: {baseline['overshoot_percent']:.1f}%")
        print(f"  Settling time: {baseline['settling_time']:.2f}s")

        if baseline['error_mm'] > 0:
            improvement = ((baseline['error_mm'] - best['error_mm']) / baseline['error_mm']) * 100
            print(f"\nSteady-state error reduced by {improvement:.1f}%"
                  f" ({baseline['error_mm']:.5f} -> {best['error_mm']:.5f} mm)")
        print("\nNote: steady-state error saturates near the integrator's numerical floor,"
              "\nso the top entries are effectively tied on accuracy and are separated"
              "\nby settling time and overshoot.")


def plot_comparison(axis_name: str, current_gains: Tuple[float, float, float],
                    optimal_gains: Tuple[float, float, float], save_path: str = None):
    from validate_gains import calculate_metrics, run_step_response

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    target = 100.0

    pid_current = PIDController(current_gains[0], current_gains[1], current_gains[2],
                                output_limits=(-200, 200),
                                integral_limits=(-100, 100))
    sim_current = VelocityRobotSimulator(axis_name)
    time_current, pos_current = run_step_response(pid_current, sim_current, target, duration=15.0)
    metrics_current = calculate_metrics(time_current, pos_current, target)

    pid_optimal = PIDController(optimal_gains[0], optimal_gains[1], optimal_gains[2],
                                output_limits=(-200, 200),
                                integral_limits=(-100, 100))
    sim_optimal = VelocityRobotSimulator(axis_name)
    time_optimal, pos_optimal = run_step_response(pid_optimal, sim_optimal, target, duration=15.0)
    metrics_optimal = calculate_metrics(time_optimal, pos_optimal, target)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(time_current, pos_current, 'b-', linewidth=2, label='Position')
    ax1.axhline(y=target, color='r', linestyle='--', linewidth=2, label='Setpoint')
    ax1.axhline(y=target * 1.02, color='g', linestyle=':', alpha=0.5, label='±2% Band')
    ax1.axhline(y=target * 0.98, color='g', linestyle=':', alpha=0.5)

    error_mm_current = metrics_current['steady_state_error_meters'] * 1000
    textstr_current = '\n'.join([
        f"Kp={current_gains[0]:.1f}, Ki={current_gains[1]:.1f}, Kd={current_gains[2]:.1f}",
        f"Error: {error_mm_current:.3f} mm",
        f"Overshoot: {metrics_current['overshoot_percent']:.1f}%",
        f"Settling: {metrics_current['settling_time']:.2f}s"
    ])
    ax1.text(0.02, 0.98, textstr_current, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Position (units)', fontsize=11)
    ax1.set_title(f'{axis_name}-Axis: Current Gains', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_optimal, pos_optimal, 'g-', linewidth=2, label='Position')
    ax2.axhline(y=target, color='r', linestyle='--', linewidth=2, label='Setpoint')
    ax2.axhline(y=target * 1.02, color='purple', linestyle=':', alpha=0.5, label='±2% Band')
    ax2.axhline(y=target * 0.98, color='purple', linestyle=':', alpha=0.5)

    error_mm_optimal = metrics_optimal['steady_state_error_meters'] * 1000
    improvement = ((error_mm_current - error_mm_optimal) / error_mm_current) * 100
    textstr_optimal = '\n'.join([
        f"Kp={optimal_gains[0]:.1f}, Ki={optimal_gains[1]:.1f}, Kd={optimal_gains[2]:.1f}",
        f"Error: {error_mm_optimal:.3f} mm",
        f"Overshoot: {metrics_optimal['overshoot_percent']:.1f}%",
        f"Settling: {metrics_optimal['settling_time']:.2f}s",
        f"Improvement: {improvement:.1f}%"
    ])
    ax2.text(0.02, 0.98, textstr_optimal, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position (units)', fontsize=11)
    ax2.set_title(f'{axis_name}-Axis: Optimized Gains', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("\n" + "=" * 70)
    print("  PID GAIN OPTIMIZATION")
    print("=" * 70)

    results_x = optimize_gains_for_axis(
        axis_name='X',
        kp_range=(1.5, 3.5, 5),
        ki_range=(0.8, 2.0, 7),
        kd_range=(0.6, 1.2, 4),
        max_overshoot=45.0,
        max_settling_time=10.0
    )
    print_top_results(results_x, 'X', baseline_gains=BASELINE_GAINS['X'], top_n=10)

    results_z = optimize_gains_for_axis(
        axis_name='Z',
        kp_range=(1.5, 3.5, 5),
        ki_range=(1.0, 2.5, 8),
        kd_range=(0.6, 1.2, 4),
        max_overshoot=45.0,
        max_settling_time=11.0
    )
    print_top_results(results_z, 'Z', baseline_gains=BASELINE_GAINS['Z'], top_n=10)

    if results_x:
        best_x = results_x[0]
        plot_comparison('X',
                        current_gains=BASELINE_GAINS['X'],
                        optimal_gains=(best_x['Kp'], best_x['Ki'], best_x['Kd']),
                        save_path='plots/x_axis_optimization_comparison.png')

    if results_z:
        best_z = results_z[0]
        plot_comparison('Z',
                        current_gains=BASELINE_GAINS['Z'],
                        optimal_gains=(best_z['Kp'], best_z['Ki'], best_z['Kd']),
                        save_path='plots/z_axis_optimization_comparison.png')

    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()