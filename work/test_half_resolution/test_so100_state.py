#!/usr/bin/env python

"""
Connect to SO100 follower arm, send a small action, and plot state vs time.
All parameters are hardcoded - run without arguments.

Usage:
    python test_so100_state.py
"""

import time
import matplotlib.pyplot as plt
import numpy as np

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


# ============ Hardcoded Parameters ============
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "start_new_heihei_2"
FPS = 30
DURATION_S = 1  # How long to record (seconds)
ACTION_DELTA = 5.0  # Small position change in degrees to send


MAX_RELATIVE_TARGET = {
    "shoulder_pan": 4.11,
    "shoulder_lift": 3.16,
    "elbow_flex": 3.69,
    "wrist_flex": 5.38,
    "wrist_roll": 6.96,
    "gripper": 12.1,
}


def main():
    # Create robot config and instance
    robot_config = SO100FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        # cameras=CAMERAS,
        # max_relative_target=MAX_RELATIVE_TARGET,
    )
    robot = SO100Follower(robot_config)

    # Connect to robot
    print(f"Connecting to SO100 on port {ROBOT_PORT}...")
    robot.connect(calibrate=False)
    print("Connected!")

    # Get motor names from the bus
    motor_names = list(robot.bus.motors.keys())
    print(f"Motors: {motor_names}")

    # Read initial position
    initial_obs = robot.get_observation()
    initial_positions = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
    print(f"Initial positions: {initial_positions}")

    # Create a small action: add a small delta to each motor
    action = {}
    for motor_name in motor_names:
        key = f"{motor_name}.pos"
        action[key] = initial_positions[key] + ACTION_DELTA
    print(f"Sending action with +{ACTION_DELTA} degree offset: {action}")

    # Send the action once to move to the new position
    robot.send_action(action)
    print("Initial action sent. Starting to record state...")

    # Record state over time
    timestamps = []
    state_history = {motor: [] for motor in motor_names}

    start_time = time.perf_counter()
    loop_interval = 1.0 / FPS
    num_samples = 0

    try:
        while (time.perf_counter() - start_time) < DURATION_S:
            loop_start = time.perf_counter()

            # Get current observation
            obs = robot.get_observation()

            # Record timestamp and positions (in milliseconds)
            t = (time.perf_counter() - start_time) * 1000
            timestamps.append(t)

            for motor in motor_names:
                key = f"{motor}.pos"
                state_history[motor].append(obs[key])

            num_samples += 1

            # Wait for next loop iteration
            elapsed = time.perf_counter() - loop_start
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")

    # Print statistics
    print(f"\nRecorded {num_samples} samples over {timestamps[-1]:.2f} ms")
    print(f"Effective FPS: {num_samples / (timestamps[-1] / 1000):.1f}")

    # Plot state vs time
    num_motors = len(motor_names)
    fig, axes = plt.subplots(num_motors, 1, figsize=(10, 2 * num_motors), sharex=True)
    if num_motors == 1:
        axes = [axes]

    for idx, motor in enumerate(motor_names):
        ax = axes[idx]
        ax.plot(timestamps, state_history[motor], marker='.', markersize=3, linewidth=1)
        ax.set_ylabel(f"{motor}\n(deg)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=initial_positions[f"{motor}.pos"], color='r', linestyle='--', alpha=0.5, label='Initial')
        ax.axhline(y=initial_positions[f"{motor}.pos"] + ACTION_DELTA, color='g', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlim(left=0)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("SO100 Joint Positions Over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig("so100_state_vs_time.png", dpi=150)
    print("\nPlot saved to so100_state_vs_time.png")
    plt.show()


if __name__ == "__main__":
    main()