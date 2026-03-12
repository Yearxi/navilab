# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record and plot monitored metrics (v, omega, turn radius R, contact-sensor force) over time.

Usage:
  python scripts/record.py --task Navilab-Jackal-v0 --steps 500
  python scripts/record.py --task Navilab-Jackal-v0 --steps 500 --headless --out_dir ./my_logs

To add more metrics: extend collect_metrics() to return extra keys and add them to history dict + CSV/plot.
"""

import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record env metrics and plot curves.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Navilab-Jackal-Nav-v0")
parser.add_argument("--steps", type=int, default=500, help="Number of env steps to record.")
parser.add_argument("--out_dir", type=str, default=None, help="Directory to save plot and CSV (default: navilab/logs/record_metrics).")
parser.add_argument("--env_index", type=int, default=0, help="Which env to record when num_envs > 1.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import navilab.tasks  # noqa: F401


def collect_metrics(env, env_index: int):
    """Read v, omega, R and contact-sensor net force for one env. Returns dict with scalar values.

    Add more keys here to monitor e.g. distance_to_goal, reward, position, etc.
    Then add the same keys to the history dict and to the CSV/plot section below.
    """
    robot = env.unwrapped.scene["robot"]
    lin_vel_w = robot.data.root_lin_vel_w[env_index]
    ang_vel_w = robot.data.root_ang_vel_w[env_index]
    heading = robot.data.heading_w[env_index].item()

    vx_w, vy_w = lin_vel_w[0].item(), lin_vel_w[1].item()
    v = vx_w * math.cos(heading) + vy_w * math.sin(heading)
    omega = ang_vel_w[2].item()
    R = abs(v / omega) if abs(omega) > 1e-6 else 0.0

    # Contact sensor: net normal force on robot body (N, B, 3); take magnitude for one body
    contact_sensor = env.unwrapped.scene["contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    if net_forces is not None and net_forces.numel() > 0:
        f = net_forces[env_index]  # (num_bodies, 3)
        force_mag = torch.linalg.norm(f, dim=-1).sum().item()  # total force magnitude across bodies
        force_x = f[:, 0].sum().item()
        force_y = f[:, 1].sum().item()
        force_z = f[:, 2].sum().item()
    else:
        force_mag = force_x = force_y = force_z = float("nan")

    return {"v": v, "omega": omega, "R": R, "force_mag": force_mag, "force_x": force_x, "force_y": force_y, "force_z": force_z}


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_index = min(args_cli.env_index, env.unwrapped.num_envs - 1)

    # buffers for recording (v, omega, R, force sensor)
    history = {k: [] for k in ["v", "omega", "R", "force_mag", "force_x", "force_y", "force_z"]}
    steps = args_cli.steps

    obs, _ = env.reset()
    with torch.inference_mode():
        for t in range(steps):
            if not simulation_app.is_running():
                break
            actions = torch.ones(env.action_space.shape, device=env.unwrapped.device)
            obs, _, _, _, _ = env.step(actions)
            m = collect_metrics(env, env_index)
            for k in history:
                history[k].append(m[k])

    env.close()

    # save and plot
    out_dir = args_cli.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "record_metrics")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_header = "step,v,omega,R,force_mag,force_x,force_y,force_z\n"
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write(csv_header)
        for i in range(len(history["v"])):
            f.write(
                f"{i},{history['v'][i]},{history['omega'][i]},{history['R'][i]},"
                f"{history['force_mag'][i]},{history['force_x'][i]},{history['force_y'][i]},{history['force_z'][i]}\n"
            )
    print(f"[INFO]: Saved {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]: matplotlib not installed, skip plot.")
        return

    n_axes = 4  # v, omega, R, force sensor
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(8, 10))
    steps_x = list(range(len(history["v"])))

    axes[0].plot(steps_x, history["v"], label="v (m/s)")
    axes[0].set_ylabel("v")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps_x, history["omega"], label="omega (rad/s)")
    axes[1].set_ylabel("omega")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps_x, history["R"], label="R (m)")
    axes[2].set_ylabel("R")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Force sensor: magnitude and x,y,z components
    ax_force = axes[3]
    ax_force.plot(steps_x, history["force_mag"], label="force_mag (N)", alpha=0.9)
    ax_force.plot(steps_x, history["force_x"], label="force_x", alpha=0.7)
    ax_force.plot(steps_x, history["force_y"], label="force_y", alpha=0.7)
    ax_force.plot(steps_x, history["force_z"], label="force_z", alpha=0.7)
    ax_force.set_xlabel("step")
    ax_force.set_ylabel("force (N)")
    ax_force.legend(loc="upper right")
    ax_force.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO]: Saved {plot_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
