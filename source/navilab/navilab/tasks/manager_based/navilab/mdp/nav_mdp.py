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
    cmd_b = command.command
    cmd_xy = cmd_b[:, :2]
    cmd_yaw = cmd_b[:, 3].unsqueeze(-1)
    return torch.cat([cmd_xy, cmd_yaw], dim=-1)


def lidar_ranges(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened LiDAR ranges from RayCaster hits.

    Distances for rays without hits are set to max_distance.
    Hits that fall outside the current env's spatial cell (e.g. other envs' obstacles
    when raycasting sees them) are treated as no-hit so LiDAR stays consistent with collision.
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w  # (N, B, 3), NaN or inf if no hit
    # distance in XY-plane (navigation style)
    pos = sensor.data.pos_w.unsqueeze(1)
    vec = hits - pos
    dist = torch.linalg.norm(vec[..., :2], dim=-1)
    # replace NaNs/inf (no hit) with max_distance
    dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, sensor.cfg.max_distance))
    # only count hits inside this env's region (radius REGION_RADIUS=8; ignore cross-env "ghost" obstacles)
    REGION_RADIUS = 8.0
    origin_xy = env.scene.env_origins[:, :2]  # (N, 2)
    hit_xy = hits[..., :2]  # (N, B, 2)
    dist_to_origin = torch.linalg.norm(hit_xy - origin_xy.unsqueeze(1), dim=-1)
    in_region = dist_to_origin <= REGION_RADIUS
    valid_hit = torch.isfinite(hits[..., 0])
    dist = torch.where(valid_hit & in_region, dist, torch.full_like(dist, sensor.cfg.max_distance))
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
    static_assets = _collect_obstacle_assets(env, "static_obstacle_")
    if not static_assets:
        return
    low, high = area_xy
    num_envs = env.num_envs
    effective_ids = (
        env_ids if env_ids is not None else torch.arange(num_envs, device=env.device)
    )
    n = effective_ids.shape[0]
    env_origins = env.scene.env_origins[effective_ids]  # (n, 3) world position of each env
    robot_pos_w = robot.data.root_pos_w[effective_ids, :2]  # world XY
    for asset in static_assets:
        # sample in local frame (relative to env origin)
        pos_xy_local = torch.empty(n, 2, device=env.device)
        pos_xy_local.uniform_(low, high)
        pos_xy_world = env_origins[:, :2] + pos_xy_local
        too_close = torch.norm(pos_xy_world - robot_pos_w, dim=-1) < min_distance_from_robot
        max_attempts = 10
        for _ in range(max_attempts):
            if not too_close.any():
                break
            resample = torch.empty(too_close.sum(), 2, device=env.device)
            resample.uniform_(low, high)
            pos_xy_local[too_close] = resample
            pos_xy_world = env_origins[:, :2] + pos_xy_local
            too_close = torch.norm(pos_xy_world - robot_pos_w, dim=-1) < min_distance_from_robot
        z_local = torch.full((n,), 0.5, device=env.device)
        pos_world = env_origins.clone()
        pos_world[:, :2] += pos_xy_local
        pos_world[:, 2] = env_origins[:, 2] + z_local
        quat = asset.data.root_link_state_w[effective_ids, 3:7].clone()
        vel = asset.data.root_com_vel_w[effective_ids].clone()
        root_state = torch.cat([pos_world, quat, vel], dim=-1)
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
    effective_ids = (
        env_ids if env_ids is not None else torch.arange(num_envs, device=env.device)
    )
    env_origins = env.scene.env_origins[effective_ids]  # (n, 3)
    for asset in dyn_assets:
        pos_world = asset.data.root_link_state_w[effective_ids, :3].clone()
        pos_local = pos_world - env_origins
        delta_xy = torch.empty(effective_ids.shape[0], 2, device=env.device)
        delta_xy.uniform_(-1.0, 1.0)
        pos_xy_local = pos_local[:, :2] + delta_xy
        pos_xy_local = torch.clamp(pos_xy_local, low, high)
        new_pos_world = env_origins.clone()
        new_pos_world[:, :2] += pos_xy_local
        new_pos_world[:, 2] = pos_world[:, 2]
        quat = asset.data.root_link_state_w[effective_ids, 3:7].clone()
        vel = asset.data.root_com_vel_w[effective_ids].clone()
        root_state = torch.cat([new_pos_world, quat, vel], dim=-1)
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


def illegal_contact_xy(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when the horizontal (xy) contact force on the sensor exceeds the force threshold."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # (N, T, B, 3)
    if net_contact_forces is None or net_contact_forces.numel() == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    forces_xy = net_contact_forces[:, :, sensor_cfg.body_ids, :2]  # (N, T, B, 2)
    force_xy_norm = torch.linalg.norm(forces_xy, dim=-1)  # (N, T, B)
    return torch.any(
        torch.max(force_xy_norm, dim=1)[0] > threshold,
        dim=1,
    )


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary collision indicator from contact sensor (horizontal xy force only)."""
    mask = illegal_contact_xy(env, sensor_cfg=sensor_cfg, threshold=threshold)
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

