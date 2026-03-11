import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import NonHolonomicActionCfg

from .mdp.diff_drive_actions import DiffDriveWheelVelocityActionCfg
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

current_dir = os.path.dirname(os.path.abspath(__file__)) 
robot_usd_path = os.path.join(current_dir, "robot", "jackal.usd")

@configclass
class JackalNavSceneCfg(InteractiveSceneCfg):
    """Scene for Jackal navigation with LiDAR and obstacles."""

    # flat terrain (can be replaced with heightfield later)
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

    # Jackal base
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            activate_contact_sensors=True,
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=1e6,
                effort_limit_sim=1000.0,
            ),
        },
    )

    # 2D LiDAR mounted on base (MultiMeshRayCaster supports multiple meshes: ground + obstacles)
    lidar_2d = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.4)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=0.5,
        ),
        max_distance=20.0,
        mesh_prim_paths=["/World/ground", "/World/obstacles_static", "/World/obstacles_dynamic"],
        debug_vis=False,
    )

    # 3D LiDAR mounted slightly higher
    lidar_3d = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.6)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=40.0,
        mesh_prim_paths=["/World/ground", "/World/obstacles_static", "/World/obstacles_dynamic"],
        debug_vis=False,
    )

    # contact sensor on robot base only (exclude wheels to avoid ground contact triggering termination)
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        history_length=2,
        track_air_time=False,
    )

    # simple dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=5000.0,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        num_static = 1
        num_dynamic = 0
        # create static obstacles (count fixed here so scene does not see an int as asset)
        for i in range(num_static):
            setattr(
                self,
                f"static_obstacle_{i}",
                AssetBaseCfg(
                    prim_path=f"/World/obstacles_static/box_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.6, 0.6, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.4, 0.0),
                        ),
                    ),
                ),
            )
        # create dynamic obstacles (simple rigid boxes)
        for i in range(num_dynamic):
            setattr(
                self,
                f"dynamic_obstacle_{i}",
                AssetBaseCfg(
                    prim_path=f"/World/obstacles_dynamic/box_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.6, 0.6, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.7, 0.0),
                        ),
                    ),
                ),
            )
        # LiDAR only references existing mesh roots; omit path when count is 0
        mesh_paths = ["/World/ground"]
        if num_static > 0:
            mesh_paths.append("/World/obstacles_static")
        if num_dynamic > 0:
            mesh_paths.append("/World/obstacles_dynamic")
        self.lidar_2d.mesh_prim_paths = mesh_paths
        self.lidar_3d.mesh_prim_paths = mesh_paths


@configclass
class CommandsCfg:
    """Goal command for navigation."""

    goal_2d = UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        debug_vis=True,
        resampling_time_range=(1000.0, 1000.0),  # goal fixed per episode (no resample; inf not supported by uniform_)
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-8.0, 8.0),
            pos_y=(-8.0, 8.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """SE(2) twist actions (v, omega) mapped to 4-wheel velocities (Jackal skid-steer)."""

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


@configclass
class ObservationsCfg:
    """Observations for navigation policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        # goal pose relative to robot (x, y, yaw_error)
        goal_rel = ObsTerm(
            func=mdp.goal_relative_pose,
            params={"command_name": "goal_2d"},
        )
        # robot pose in world (x, y, heading)
        robot_pose = ObsTerm(func=mdp.robot_pose_2d)
        # 2D LiDAR ranges
        lidar_2d = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_2d")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 1.0),
        )
        # 3D LiDAR ranges (flattened)
        lidar_3d = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_3d")},
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
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
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

    randomize_static_obstacles = EventTerm(
        func=mdp.randomize_static_obstacles,
        mode="reset",
        params={
            "area_xy": (-8.0, 8.0),
            "min_distance_from_robot": 1.0,
        },
    )

    randomize_dynamic_obstacles = EventTerm(
        func=mdp.randomize_dynamic_obstacles,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "area_xy": (-8.0, 8.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms (initial simple version)."""

    # encourage getting closer to goal
    dist_to_goal = RewTerm(
        func=mdp.distance_to_goal,
        weight=-1.0,
        params={"command_name": "goal_2d"},
    )
    # success bonus (sparse)
    success = RewTerm(
        func=mdp.success_reward,
        weight=5.0,
        params={"command_name": "goal_2d", "threshold": 0.3},
    )
    # collision penalty
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0},
    )
    # small penalty on action magnitude for smoothness
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.01,
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0},
    )

    # goal_reached = DoneTerm(
    #     func=mdp.goal_reached,
    #     params={"command_name": "goal_2d", "threshold": 0.3},
    # )


@configclass
class CurriculumCfg:
    """Placeholder curriculum (distance to goal)."""

    goal_distance = CurrTerm(
        func=mdp.goal_distance_curriculum,
        params={
            "min_radius": 2.0,
            "max_radius": 8.0,
        },
    )


@configclass
class JackalNavEnvCfg(ManagerBasedRLEnvCfg):
    """Jackal navigation environment with 2D/3D LiDAR and obstacles."""

    scene: JackalNavSceneCfg = JackalNavSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # base sim settings
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (12.0, 0.0, 8.0)
        # sensor update periods
        self.scene.lidar_2d.update_period = self.decimation * self.sim.dt
        self.scene.lidar_3d.update_period = self.decimation * self.sim.dt
        self.scene.contact_sensor.update_period = self.sim.dt

