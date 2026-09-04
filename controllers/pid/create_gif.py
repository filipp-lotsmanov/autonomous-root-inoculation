#!/usr/bin/env python3
import time
import numpy as np
from typing import Tuple, Optional
from sim_class import Simulation
from pid_controller import PIDController

Kp_xy, Ki_xy, Kd_xy = 6.0, 3.6, 1.2
Kp_z, Ki_z, Kd_z = 7.0, 4.6, 1.2

VXY_MAX = 0.25
VZ_MAX = 0.20

I_LIM_XY = 0.50
I_LIM_Z = 0.80

TOL = 0.001
SIM_HZ = 240.0
DT = 1.0 / SIM_HZ

Z_MARGIN_DOWN = 0.15
Z_MARGIN_UP = 0.15


def get_pipette_xyz(sim, robot_key):
    states = sim.get_states()
    pip = states[robot_key]["pipette_position"]
    return float(pip[0]), float(pip[1]), float(pip[2])


def parse_xyz(line: str) -> Optional[Tuple[float, float, float]]:
    try:
        for char in ",()[]":
            line = line.replace(char, " ")
        parts = [float(p) for p in line.split() if p]
        if len(parts) == 3:
            return tuple(parts)
        return None
    except (ValueError, TypeError):
        return None


def move_pipette_to(sim, robot_index, target_xyz, verbose=True):
    robot_id = sim.robotIds[robot_index]
    robot_key = f"robotId_{robot_id}"

    cx, cy, cz = get_pipette_xyz(sim, robot_key)
    z_min, z_max = cz - Z_MARGIN_DOWN, cz + Z_MARGIN_UP

    tx, ty, tz = target_xyz
    tz = np.clip(tz, z_min, z_max)

    pid_x = PIDController(Kp_xy, Ki_xy, Kd_xy, (-VXY_MAX, VXY_MAX), (-I_LIM_XY, I_LIM_XY))
    pid_y = PIDController(Kp_xy, Ki_xy, Kd_xy, (-VXY_MAX, VXY_MAX), (-I_LIM_XY, I_LIM_XY))
    pid_z = PIDController(Kp_z, Ki_z, Kd_z, (-VZ_MAX, VZ_MAX), (-I_LIM_Z, I_LIM_Z))

    if verbose:
        print(f"Moving to: {tx:.4f}, {ty:.4f}, {tz:.4f}")

    start_time = time.time()
    hold_counter = 0
    required_hold_ticks = 10

    while (time.time() - start_time) < 15.0:
        cx, cy, cz = get_pipette_xyz(sim, robot_key)

        vx = pid_x.update(cx, tx, dt=DT)
        vy = pid_y.update(cy, ty, dt=DT)
        vz = pid_z.update(cz, tz, dt=DT)

        actions = [[0, 0, 0, 0] for _ in range(len(sim.robotIds))]
        actions[robot_index] = [vx, vy, vz, 0]
        sim.run(actions, num_steps=1)

        cx, cy, cz = get_pipette_xyz(sim, robot_key)
        err = np.linalg.norm([tx - cx, ty - cy, tz - cz])

        if err <= TOL:
            hold_counter += 1
            if hold_counter >= required_hold_ticks:
                if verbose:
                    print(f"Success! Final error: {err * 1000:.2f}mm")
                return True
        else:
            hold_counter = 0

    cx, cy, cz = get_pipette_xyz(sim, robot_key)
    final_err = np.linalg.norm([tx - cx, ty - cy, tz - cz])
    if verbose:
        print(f"Timeout! Final error: {final_err * 1000:.2f}mm")
    return False


def main():
    sim = Simulation(num_agents=1, render=True)
    try:
        robot_key = f"robotId_{sim.robotIds[0]}"
        x0, y0, z0 = get_pipette_xyz(sim, robot_key)

        print(f"Initial position: X={x0:.4f}, Y={y0:.4f}, Z={z0:.4f}")
        print("Initializing working envelope...")

        print("-> Calibrating Z-Max...")
        move_pipette_to(sim, 0, (x0, y0, z0 + 0.06), verbose=False)
        print("Envelope initialized.\n")

        while True:
            line = input("target (x y z)> ").strip().lower()
            if line in {"q", "quit", "exit"}:
                break

            xyz = parse_xyz(line)
            if xyz:
                move_pipette_to(sim, 0, xyz)
            else:
                print("Error: Please provide 3 numbers (e.g., 0.05 0.05 0.18)")

    finally:
        sim.close()


if __name__ == "__main__":
    main()