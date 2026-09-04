"""Tests for the PID controller.

These exercise the controller in isolation, with no robot or simulator: the
plant is either absent (checking the control law directly) or the analytic
VelocityRobotSimulator used for gain tuning.
"""
import numpy as np
import pytest

from pid_controller import (
    PIDController,
    ThreeAxisPIDController,
    simulator_units_to_meters,
)
from validate_gains import (
    VelocityRobotSimulator,
    calculate_metrics,
    run_step_response,
)


class TestPIDController:
    def test_zero_error_with_no_history_produces_no_output(self):
        pid = PIDController(1.0, 0.0, 0.0)
        assert pid.update(5.0, 5.0, dt=0.01) == 0.0

    def test_proportional_term_scales_with_error(self):
        pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)
        pid.update(0.0, 0.0, dt=0.01)  # initialise measurement history
        assert pid.update(0.0, 10.0, dt=0.01) == pytest.approx(20.0)

    def test_integral_accumulates_on_sustained_error(self):
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0)
        pid.update(0.0, 1.0, dt=0.1)
        outputs = [pid.update(0.0, 1.0, dt=0.1) for _ in range(3)]
        # Each step adds error*dt = 0.1 to the integral.
        assert outputs == pytest.approx([0.1, 0.2, 0.3])

    def test_output_is_clamped_to_limits(self):
        pid = PIDController(Kp=1000.0, Ki=0.0, Kd=0.0, output_limits=(-5.0, 5.0))
        pid.update(0.0, 0.0, dt=0.01)
        assert pid.update(0.0, 100.0, dt=0.01) == 5.0
        assert pid.update(0.0, -100.0, dt=0.01) == -5.0

    def test_integral_does_not_wind_up_while_output_saturated(self):
        """Conditional integration: no accumulation when the output is clamped."""
        pid = PIDController(Kp=10.0, Ki=1.0, Kd=0.0, output_limits=(-1.0, 1.0))
        pid.update(0.0, 0.0, dt=0.01)
        for _ in range(100):
            pid.update(0.0, 50.0, dt=0.01)
        assert pid._integral == 0.0

    def test_integral_limits_are_enforced(self):
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0,
                            output_limits=(-1e6, 1e6),
                            integral_limits=(-0.5, 0.5))
        pid.update(0.0, 1.0, dt=0.1)
        for _ in range(100):
            pid.update(0.0, 1.0, dt=0.1)
        assert pid._integral == pytest.approx(0.5)

    def test_derivative_acts_on_measurement_not_setpoint(self):
        """A setpoint step must not produce a derivative kick."""
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0)
        pid.update(0.0, 0.0, dt=0.1)
        assert pid.update(0.0, 100.0, dt=0.1) == pytest.approx(0.0)

    def test_derivative_opposes_measurement_change(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=2.0)
        pid.update(0.0, 0.0, dt=0.1)
        # Measurement rises by 1.0 over dt=0.1 -> rate 10.0 -> output -2*10.
        assert pid.update(1.0, 0.0, dt=0.1) == pytest.approx(-20.0)

    def test_reset_clears_accumulated_state(self):
        pid = PIDController(Kp=1.0, Ki=1.0, Kd=1.0)
        for _ in range(10):
            pid.update(0.0, 1.0, dt=0.1)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._previous_measurement == 0.0
        assert not pid._initialized

    def test_zero_dt_does_not_divide_by_zero(self):
        pid = PIDController(Kp=1.0, Ki=1.0, Kd=1.0)
        pid.update(0.0, 1.0, dt=0.01)
        assert np.isfinite(pid.update(0.5, 1.0, dt=0.0))

    def test_set_gains_updates_behaviour(self):
        pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.0)
        pid.update(0.0, 0.0, dt=0.01)
        assert pid.update(0.0, 1.0, dt=0.01) == pytest.approx(1.0)
        pid.set_gains(5.0, 0.0, 0.0)
        assert pid.update(0.0, 1.0, dt=0.01) == pytest.approx(5.0)


class TestThreeAxisPIDController:
    def test_axes_are_controlled_independently(self):
        ctrl = ThreeAxisPIDController(
            gains_x=(1.0, 0.0, 0.0),
            gains_y=(2.0, 0.0, 0.0),
            gains_z=(3.0, 0.0, 0.0),
        )
        ctrl.update((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), dt=0.01)
        out = ctrl.update((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), dt=0.01)
        assert out == pytest.approx((1.0, 2.0, 3.0))

    def test_per_axis_output_limits_apply(self):
        ctrl = ThreeAxisPIDController(
            gains_x=(100.0, 0.0, 0.0),
            gains_y=(100.0, 0.0, 0.0),
            gains_z=(100.0, 0.0, 0.0),
            output_limits={'x': (-1.0, 1.0), 'y': (-2.0, 2.0), 'z': (-3.0, 3.0)},
        )
        ctrl.update((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), dt=0.01)
        assert ctrl.update((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), dt=0.01) \
            == pytest.approx((1.0, 2.0, 3.0))

    def test_reset_clears_all_three_axes(self):
        ctrl = ThreeAxisPIDController((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0))
        for _ in range(5):
            ctrl.update((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), dt=0.1)
        ctrl.reset()
        assert (ctrl.pid_x._integral, ctrl.pid_y._integral, ctrl.pid_z._integral) == (0.0, 0.0, 0.0)

    def test_invalid_axis_name_is_rejected(self):
        ctrl = ThreeAxisPIDController((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="Invalid axis"):
            ctrl.set_gains('w', 1.0, 1.0, 1.0)


class TestUnitConversion:
    def test_simulator_units_are_millimetres(self):
        assert simulator_units_to_meters(1.0) == pytest.approx(0.001)
        assert simulator_units_to_meters(0.0) == 0.0
        assert simulator_units_to_meters(-100.0) == pytest.approx(-0.1)


class TestTunedGainsStepResponse:
    """Regression guard on the documented step-response figures.

    docs/pid_controller.md reports these at t = 10 s; the numbers are
    window-dependent, so the duration is pinned here deliberately.
    """

    @pytest.mark.parametrize("axis,gains,expected_error_mm", [
        ('X', (3.0, 1.8, 0.6), 0.031),
        ('Y', (3.0, 1.8, 0.6), 0.031),
        ('Z', (3.5, 2.29, 0.6), 0.011),
    ])
    def test_documented_steady_state_error_reproduces(self, axis, gains, expected_error_mm):
        pid = PIDController(*gains, output_limits=(-200, 200), integral_limits=(-100, 100))
        sim = VelocityRobotSimulator(axis)
        time_data, pos_data = run_step_response(pid, sim, target=100.0, duration=10.0)
        metrics = calculate_metrics(time_data, pos_data, 100.0)
        error_mm = metrics['steady_state_error_meters'] * 1000
        assert error_mm == pytest.approx(expected_error_mm, abs=0.001)

    def test_controller_converges_towards_setpoint(self):
        pid = PIDController(3.0, 1.8, 0.6, output_limits=(-200, 200),
                            integral_limits=(-100, 100))
        sim = VelocityRobotSimulator('X')
        _, pos_data = run_step_response(pid, sim, target=100.0, duration=10.0)
        assert pos_data[-1] == pytest.approx(100.0, abs=0.1)

    def test_step_response_is_deterministic(self):
        runs = []
        for _ in range(2):
            pid = PIDController(3.0, 1.8, 0.6, output_limits=(-200, 200),
                                integral_limits=(-100, 100))
            sim = VelocityRobotSimulator('X')
            _, pos = run_step_response(pid, sim, target=100.0, duration=2.0)
            runs.append(pos)
        assert runs[0] == runs[1]

    def test_z_axis_is_slower_than_xy(self):
        """Z carries more friction and more lag, so it must rise more slowly."""
        results = {}
        for axis in ('X', 'Z'):
            pid = PIDController(3.0, 1.8, 0.6, output_limits=(-200, 200),
                                integral_limits=(-100, 100))
            sim = VelocityRobotSimulator(axis)
            time_data, pos_data = run_step_response(pid, sim, target=100.0, duration=10.0)
            results[axis] = calculate_metrics(time_data, pos_data, 100.0)['rise_time']
        assert results['Z'] > results['X']
