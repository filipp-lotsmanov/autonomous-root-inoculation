import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pid_controller import PIDController, simulator_units_to_meters


class VelocityRobotSimulator:
    def __init__(self, axis_name: str = 'X'):
        self.axis_name = axis_name.upper()

        if self.axis_name in ['X', 'Y']:
            self.velocity_lag = 0.05
            self.friction = 0.8
            self.max_acceleration = 200.0
        else:
            self.velocity_lag = 0.08
            self.friction = 1.2
            self.max_acceleration = 150.0

        self.position = 0.0
        self.velocity = 0.0

    def apply_velocity_command(self, velocity_command: float, dt: float):
        velocity_error = velocity_command - self.velocity
        desired_acceleration = velocity_error / self.velocity_lag

        acceleration = np.clip(desired_acceleration, -self.max_acceleration, self.max_acceleration)
        friction_force = -self.friction * self.velocity

        self.velocity += (acceleration + friction_force) * dt
        self.position += self.velocity * dt

    def get_position(self) -> float:
        return self.position

    def reset(self, position: float = 0.0):
        self.position = position
        self.velocity = 0.0


def calculate_metrics(time_data: List[float], position_data: List[float],
                      setpoint: float) -> Dict[str, float]:
    position_array = np.array(position_data)
    time_array = np.array(time_data)

    idx_10 = np.where(position_array >= 0.1 * setpoint)[0]
    idx_90 = np.where(position_array >= 0.9 * setpoint)[0]
    rise_time = time_array[idx_90[0]] - time_array[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else np.nan

    tolerance = 0.02 * abs(setpoint)
    error = np.abs(position_array - setpoint)
    violations = error > tolerance

    if np.any(violations):
        last_violation_idx = np.where(violations)[0][-1]
        if last_violation_idx < len(time_array) - 1:
            settling_time = time_array[last_violation_idx + 1]
        else:
            settling_time = time_array[-1]
    else:
        within_band = error <= tolerance
        if np.any(within_band):
            settling_time = time_array[np.where(within_band)[0][0]]
        else:
            settling_time = np.nan

    if setpoint > position_array[0]:
        max_position = np.max(position_array)
        overshoot = ((max_position - setpoint) / setpoint) * 100 if setpoint != 0 else 0.0
    else:
        min_position = np.min(position_array)
        overshoot = ((setpoint - min_position) / abs(setpoint)) * 100 if setpoint != 0 else 0.0

    final_values = position_array[-max(1, int(len(position_array) * 0.1)):]
    ss_error = abs(np.mean(final_values) - setpoint)
    rmse = np.sqrt(np.mean((position_array - setpoint) ** 2))

    return {
        'rise_time': rise_time,
        'settling_time': settling_time,
        'overshoot_percent': overshoot,
        'steady_state_error': ss_error,
        'steady_state_error_meters': simulator_units_to_meters(ss_error),
        'rmse': rmse
    }


def run_step_response(pid: PIDController, sim: VelocityRobotSimulator,
                       target: float, duration: float = 10.0, dt: float = 0.01) -> Tuple[List, List]:
    pid.reset()
    sim.reset()

    time_data = []
    position_data = []

    for i in range(int(duration / dt)):
        current_pos = sim.get_position()
        velocity_command = pid.update(current_pos, target, dt)
        sim.apply_velocity_command(velocity_command, dt)

        time_data.append(i * dt)
        position_data.append(current_pos)

    return time_data, position_data


def plot_and_save_response(time_data: List[float], position_data: List[float],
                           setpoint: float, metrics: Dict[str, float],
                           title: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_data, position_data, 'b-', linewidth=2, label='Position')
    ax.axhline(y=setpoint, color='r', linestyle='--', linewidth=2, label='Setpoint')
    ax.axhline(y=setpoint * 1.02, color='g', linestyle=':', alpha=0.5, label='±2% Band')
    ax.axhline(y=setpoint * 0.98, color='g', linestyle=':', alpha=0.5)

    if not np.isnan(metrics['settling_time']):
        ax.axvline(x=metrics['settling_time'], color='orange', linestyle='--',
                   alpha=0.7, label=f"Settling: {metrics['settling_time']:.2f}s")

    error_mm = metrics['steady_state_error_meters'] * 1000

    textstr = '\n'.join([
        f"Rise: {metrics['rise_time']:.3f}s",
        f"Settling: {metrics['settling_time']:.3f}s",
        f"Overshoot: {metrics['overshoot_percent']:.1f}%",
        f"SS Error: {error_mm:.4f} mm",
        f"RMSE: {metrics['rmse']:.3f}"
    ])

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (simulator units)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def save_all_axis_plots(output_dir: str = 'plots'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    gains = {
        'X': (3.0, 1.8, 0.6),
        'Y': (3.0, 1.8, 0.6),
        'Z': (3.5, 2.29, 0.6)
    }

    target = 100.0

    print("Generating performance validation plots...")

    for axis_name, (Kp, Ki, Kd) in gains.items():
        print(f"\n{axis_name}-Axis:")
        print(f"  Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")

        pid = PIDController(Kp, Ki, Kd,
                            output_limits=(-200, 200),
                            integral_limits=(-100, 100))
        sim = VelocityRobotSimulator(axis_name)

        time_data, pos_data = run_step_response(pid, sim, target, duration=10.0)
        metrics = calculate_metrics(time_data, pos_data, target)

        error_mm = metrics['steady_state_error_meters'] * 1000

        print(f"  Steady-state error: {error_mm:.4f} mm")
        print(f"  Settling time: {metrics['settling_time']:.2f} s")
        print(f"  Overshoot: {metrics['overshoot_percent']:.1f}%")

        save_path = os.path.join(output_dir, f'{axis_name.lower()}_axis_response.png')
        plot_and_save_response(time_data, pos_data, target, metrics,
                               f'{axis_name}-Axis Step Response', save_path)

    print(f"All plots saved to '{output_dir}/' directory")


if __name__ == "__main__":
    save_all_axis_plots()