import time
import numpy as np
from typing import Tuple, Optional

SIMULATOR_UNIT_TO_METERS = 0.001


def simulator_units_to_meters(units: float) -> float:
    return units * SIMULATOR_UNIT_TO_METERS


class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float,
                 output_limits: Tuple[float, float] = (-float('inf'), float('inf')),
                 integral_limits: Tuple[float, float] = (-float('inf'), float('inf'))):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits

        self._integral = 0.0
        self._previous_measurement = 0.0
        self._last_time = None
        self._initialized = False

    def update(self, current_value: float, setpoint: float, dt: Optional[float] = None) -> float:
        if dt is None:
            current_time = time.time()
            if self._last_time is None:
                self._last_time = current_time
                self._previous_measurement = current_value
                self._initialized = True
                return 0.0
            dt = current_time - self._last_time
            self._last_time = current_time
        else:
            if not self._initialized:
                self._previous_measurement = current_value
                self._initialized = True
                error = setpoint - current_value
                return np.clip(self.Kp * error, self.output_limits[0], self.output_limits[1])

        if dt < 1e-6:
            dt = 1e-6

        error = setpoint - current_value
        p_term = self.Kp * error
        d_term = -self.Kd * (current_value - self._previous_measurement) / dt
        self._previous_measurement = current_value

        provisional_output = p_term + (self.Ki * self._integral) + d_term

        if self.output_limits[0] < provisional_output < self.output_limits[1]:
            self._integral += error * dt
            self._integral = np.clip(self._integral, self.integral_limits[0], self.integral_limits[1])

        i_term = self.Ki * self._integral
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output

    def reset(self):
        self._integral = 0.0
        self._previous_measurement = 0.0
        self._last_time = None
        self._initialized = False

    def set_gains(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


class ThreeAxisPIDController:
    def __init__(self,
                 gains_x: Tuple[float, float, float],
                 gains_y: Tuple[float, float, float],
                 gains_z: Tuple[float, float, float],
                 output_limits: Optional[dict] = None,
                 integral_limits: Optional[dict] = None):

        if output_limits is None:
            output_limits = {
                'x': (-float('inf'), float('inf')),
                'y': (-float('inf'), float('inf')),
                'z': (-float('inf'), float('inf'))
            }

        if integral_limits is None:
            integral_limits = {
                'x': (-float('inf'), float('inf')),
                'y': (-float('inf'), float('inf')),
                'z': (-float('inf'), float('inf'))
            }

        self.pid_x = PIDController(
            gains_x[0], gains_x[1], gains_x[2],
            output_limits=output_limits.get('x', (-float('inf'), float('inf'))),
            integral_limits=integral_limits.get('x', (-float('inf'), float('inf')))
        )
        self.pid_y = PIDController(
            gains_y[0], gains_y[1], gains_y[2],
            output_limits=output_limits.get('y', (-float('inf'), float('inf'))),
            integral_limits=integral_limits.get('y', (-float('inf'), float('inf')))
        )
        self.pid_z = PIDController(
            gains_z[0], gains_z[1], gains_z[2],
            output_limits=output_limits.get('z', (-float('inf'), float('inf'))),
            integral_limits=integral_limits.get('z', (-float('inf'), float('inf')))
        )

    def update(self, current_position: Tuple[float, float, float],
               target_position: Tuple[float, float, float],
               dt: Optional[float] = None) -> Tuple[float, float, float]:

        control_x = self.pid_x.update(current_position[0], target_position[0], dt)
        control_y = self.pid_y.update(current_position[1], target_position[1], dt)
        control_z = self.pid_z.update(current_position[2], target_position[2], dt)

        return (control_x, control_y, control_z)

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def set_gains(self, axis: str, Kp: float, Ki: float, Kd: float):
        axis = axis.lower()
        if axis == 'x':
            self.pid_x.set_gains(Kp, Ki, Kd)
        elif axis == 'y':
            self.pid_y.set_gains(Kp, Ki, Kd)
        elif axis == 'z':
            self.pid_z.set_gains(Kp, Ki, Kd)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")