# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Base navigation environment configuration.

Scene contains terrain, obstacles, and sensors; robot is MISSING and must be set by child env configs.
Commands, observations, rewards, terminations, events, and curriculum are shared.
Actions are robot-specific and overridden in each child (Jackal: 4-wheel, Dingo: 2-wheel, Ridgeback: dummy joints).
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.managers import (
    CommandTermCfg as CommandTerm,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.diff_drive_actions import DiffDriveWheelVelocityActionCfg


##
# Scene (robot = MISSING, set by child)
##


@configclass
class NavSceneCfg(InteractiveSceneCfg):
    """Base scene for navigation: terrain, obstacles, LiDAR, contact sensor. Robot is MISSING."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot: ArticulationCfg = MISSING

    lidar_2d = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.2)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=1,
        ),
        max_distance=20.0,
        mesh_prim_paths=["/World/ground", "{ENV_REGEX_NS}/obstacles_static_box_.*", "{ENV_REGEX_NS}/obstacles_dynamic_box_.*"],
        debug_vis=True,
    )


    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        history_length=2,
        track_air_time=False,
        # Only count contacts with obstacles; exclude ground to avoid false collision terminations.
        filter_prim_paths_expr=["{ENV_REGEX_NS}/obstacles_static_box_.*", "{ENV_REGEX_NS}/obstacles_dynamic_box_.*"],
    )

    # DomeLight omitted to avoid native crash in lighting/DomeLight property commands (Isaac Sim 5.1).
    # Rely on default stage/viewport lighting if needed.

    def __post_init__(self):
        super().__post_init__()
        num_static = 10
        num_dynamic = 0
        # Use {ENV_REGEX_NS}/obstacles_*_box_{i} so each env gets its own obstacles (clone uses rsplit("/", 1)).
        for i in range(num_static):
            setattr(
                self,
                f"static_obstacle_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/obstacles_static_box_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.6, 0.6, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                    ),
                ),
            )
        for i in range(num_dynamic):
            setattr(
                self,
                f"dynamic_obstacle_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/obstacles_dynamic_box_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.6, 0.6, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
                    ),
                ),
            )
        mesh_paths = ["/World/ground"]
        if num_static > 0:
            mesh_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/obstacles_static_box_.*",
                    track_mesh_transforms=True,
                )
            )
        if num_dynamic > 0:
            mesh_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/obstacles_dynamic_box_.*",
                    track_mesh_transforms=True,
                )
            )
        self.lidar_2d.mesh_prim_paths = mesh_paths


##
# MDP (shared)
##


# Region radius for each env (circle radius 8); goal and obstacles use box [-8, 8] in x, y.
REGION_RADIUS = 8.0


@configclass
class CommandsCfg:
    """Goal command for navigation (within region radius 8)."""

    goal_2d = UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        debug_vis=True,
        resampling_time_range=(1000.0, 1000.0),
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-REGION_RADIUS, REGION_RADIUS),
            pos_y=(-REGION_RADIUS, REGION_RADIUS),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg:
    """Observations for navigation policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        goal_rel = ObsTerm(
            func=mdp.goal_relative_pose,
            params={"command_name": "goal_2d"},
        )
        
        lidar_2d = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_2d")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-REGION_RADIUS, REGION_RADIUS),
                "y": (-REGION_RADIUS, REGION_RADIUS),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    # Obstacles within region radius 8 (box [-8, 8] in x, y).
    randomize_static_obstacles = EventTerm(
        func=mdp.randomize_static_obstacles,
        mode="reset",
        params={"area_xy": (-REGION_RADIUS, REGION_RADIUS), "min_distance_from_robot": 1.0},
    )
    randomize_dynamic_obstacles = EventTerm(
        func=mdp.randomize_dynamic_obstacles,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"area_xy": (-REGION_RADIUS, REGION_RADIUS)},
    )


@configclass
class RewardsCfg:
    """Reward terms."""

    dist_to_goal = RewTerm(
        func=mdp.distance_to_goal,
        weight=-1.0,
        params={"command_name": "goal_2d"},
    )
    success = RewTerm(
        func=mdp.success_reward,
        weight=5.0,
        params={"command_name": "goal_2d", "threshold": 0.3},
    )
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 50.0},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached_goal = DoneTerm(
        func=mdp.goal_reached,
        time_out=False,
        params={"command_name": "goal_2d", "threshold": 0.3},
    )
    collision = DoneTerm(
        func=mdp.illegal_contact_xy,
        time_out=False,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 5.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum (distance to goal within region radius 8)."""

    goal_distance = CurrTerm(
        func=mdp.goal_distance_curriculum,
        params={"min_radius": 2.0, "max_radius": REGION_RADIUS},
    )


@configclass
class ActionsCfg:
    """Default 4-wheel twist action (overridden by child for Dingo 2-wheel / Ridgeback dummy)."""

    base_twist = DiffDriveWheelVelocityActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["front_left_wheel_joint", "rear_left_wheel_joint"],
        right_wheel_joint_names=["front_right_wheel_joint", "rear_right_wheel_joint"],
        scale=(1.0, 1.0),
        offset=(0.0, 0.0),
        clip={".*": (-1.5, 1.5)},
        track_width=0.3,
        wheel_radius=0.097,
    )


##
# Base environment (not registered; only child envs are used)
##


@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    """Base navigation env: scene with robot=MISSING, shared MDP. Child envs set scene.robot and may override actions."""

    # Each env runs in a circle of radius REGION_RADIUS=8; env_spacing=16 so envs do not overlap.
    scene: NavSceneCfg = NavSceneCfg(num_envs=1024, env_spacing=16.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (12.0, 0.0, 8.0)
        self.scene.lidar_2d.update_period = self.decimation * self.sim.dt
        self.scene.contact_sensor.update_period = self.sim.dt
