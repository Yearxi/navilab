import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp import rewards as base_rewards
from isaaclab.envs.mdp import terminations as base_terminations
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


def goal_relative_pose(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal position and heading relative to robot base: (x, y, yaw_error)."""
    command = env.command_manager.get_term(command_name)
    # UniformPose2dCommand.command is (x_rel, y_rel, z_rel, heading_rel) in base frame
    cmd_b = command.command
    rel_xy = cmd_b[:, :2]
    rel_yaw = cmd_b[:, 3].unsqueeze(-1)
    return torch.cat([rel_xy, rel_yaw], dim=-1)


def robot_pose_2d(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Robot pose in world frame: (x, y, heading)."""
    robot: Articulation = env.scene[asset_cfg.name]
    pos = robot.data.root_pos_w[:, :2]
    heading = robot.data.heading_w.unsqueeze(-1)
    return torch.cat([pos, heading], dim=-1)


def lidar_ranges(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened LiDAR ranges from RayCaster hits.

    Distances for rays without hits are set to max_distance.
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w  # (N, B, 3), NaN if no hit
    # distance in XY-plane (navigation style)
    pos = sensor.data.pos_w.unsqueeze(1)
    vec = hits - pos
    dist = torch.linalg.norm(vec[..., :2], dim=-1)
    # replace NaNs (no hit) with max_distance
    dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, sensor.cfg.max_distance))
    # normalize to [0, 1] by max_distance
    dist = torch.clamp(dist / sensor.cfg.max_distance, 0.0, 1.0)
    return dist.reshape(env.num_envs, -1)


def reset_robot_pose(
    env: ManagerBasedRLEnv,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot root pose and small velocity around origin."""
    robot: Articulation = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    # sample x, y, yaw
    r = torch.empty(num_envs, device=env.device)
    x = r.uniform_(*pose_range["x"])
    y = r.uniform_(*pose_range["y"])
    yaw = r.uniform_(*pose_range["yaw"])
    # convert yaw to quaternion (w, x, y, z)
    qw = torch.cos(yaw * 0.5)
    qz = torch.sin(yaw * 0.5)
    quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)
    # position at small height
    pos = torch.stack([x, y, torch.full_like(x, 0.2)], dim=-1)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = pos
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=None)


def _collect_obstacle_assets(env: ManagerBasedEnv, prefix: str) -> list[RigidObject]:
    """Collect rigid objects in the scene whose name starts with prefix."""
    assets: list[RigidObject] = []
    for name in env.scene.keys():
        asset = env.scene[name]
        if isinstance(asset, RigidObject) and name.startswith(prefix):
            assets.append(asset)
    return assets


def randomize_static_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    area_xy: tuple[float, float] = (-8.0, 8.0),
    min_distance_from_robot: float = 1.0,
) -> None:
    """Randomly scatter static obstacles in XY within given box.

    Event term signature requires (env, env_ids, ...) for manager param resolution.

    Args:
        env: The environment.
        env_ids: Environment indices to update (None = all).
        area_xy: Tuple (low, high) for uniform sampling on both x and y.
        min_distance_from_robot: Minimum allowed distance to robot in XY.
    """
    robot: Articulation = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    static_assets = _collect_obstacle_assets(env, "static_obstacle_")
    if not static_assets:
        return
    low, high = area_xy
    num_envs = env.num_envs
    for asset in static_assets:
        r = torch.empty(num_envs, 2, device=env.device)
        pos_xy = r.uniform_(low, high)
        too_close = (torch.norm(pos_xy - robot_pos, dim=-1) < min_distance_from_robot)
        pos_xy[too_close] = torch.tensor([high + 5.0, high + 5.0], device=env.device)
        z = torch.full((num_envs,), 0.5, device=env.device)
        pos = torch.stack([pos_xy[:, 0], pos_xy[:, 1], z], dim=-1)
        quat = asset.data.root_link_state_w[:, 3:7]
        root_state = torch.cat([pos, quat, asset.data.root_com_velocity_w], dim=-1)
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_dynamic_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    area_xy: tuple[float, float] = (-8.0, 8.0),
) -> None:
    """Randomize dynamic obstacle root positions within area (simple random walk).

    Event term signature requires (env, env_ids, ...) for manager param resolution.
    """
    dyn_assets = _collect_obstacle_assets(env, "dynamic_obstacle_")
    if not dyn_assets:
        return
    low, high = area_xy
    num_envs = env.num_envs
    for asset in dyn_assets:
        r = torch.empty(num_envs, 2, device=env.device)
        delta_xy = r.uniform_(-1.0, 1.0)
        pos = asset.data.root_link_state_w[:, :3]
        pos_xy = pos[:, :2] + delta_xy
        pos_xy = torch.clamp(pos_xy, low, high)
        z = torch.full((num_envs,), pos[:, 2].mean(), device=env.device)
        new_pos = torch.stack([pos_xy[:, 0], pos_xy[:, 1], z], dim=-1)
        quat = asset.data.root_link_state_w[:, 3:7]
        root_state = torch.cat([new_pos, quat, asset.data.root_com_velocity_w], dim=-1)
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def distance_to_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Euclidean distance from robot base to goal in world XY."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    pos = robot.data.root_pos_w[:, :2]
    goal = command.pos_command_w[:, :2]
    return torch.norm(goal - pos, dim=-1)


def goal_reached(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.3) -> torch.Tensor:
    """Episode done if within threshold distance to goal."""
    dist = distance_to_goal(env, command_name)
    return dist < threshold


def success_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.3) -> torch.Tensor:
    """Sparse reward for reaching goal."""
    reached = goal_reached(env, command_name, threshold)
    return reached.float()


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary collision indicator from contact sensor."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # reuse illegal_contact logic to detect contacts
    mask = base_terminations.illegal_contact(
        env,
        sensor_cfg=sensor_cfg,
        threshold=threshold,
    )
    return mask.float()


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 norm of last applied action."""
    return torch.sum(env.action_manager.action, dim=-1)


def goal_distance_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    min_radius: float,
    max_radius: float,
) -> dict:
    """Simple curriculum hook: shrink / expand goal radius based on success rate.

    Returns a dict that can be used by the command config if desired.
    As a placeholder, we just return fixed values.
    """
    return {"min_radius": min_radius, "max_radius": max_radius}


# Re-export commonly used base terms for convenience
time_out = base_terminations.time_out
illegal_contact = base_terminations.illegal_contact

