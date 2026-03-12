# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Ridgeback UR navigation environment: dummy joints (base_x, base_y, base_yaw), NonHolonomicAction."""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .navilab_env_cfg import (
    ActionsCfg,
    NavEnvCfg,
    NavSceneCfg,
)
from .mdp.diff_drive_actions import DiffDriveWheelVelocityActionCfg

@configclass
class RidgebackSceneCfg(NavSceneCfg):
    """Scene with Ridgeback UR (dummy-joint mobile base)."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/iRobot/Create3/create_3.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                stiffness=0.0,
                damping=1e5,
                effort_limit_sim=1000.0,
            ),
        },
    )


@configclass
class RidgebackActionsCfg(ActionsCfg):
    """Dummy joints: (v, omega) -> base_x, base_y, base_yaw velocity targets."""

    base_twist = DiffDriveWheelVelocityActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["left_wheel_joint"],
        right_wheel_joint_names=["right_wheel_joint"],
        scale=(1.0, 1.0),
        offset=(0.0, 0.0),
        clip={".*": (-1.5, 1.5)},
        track_width=0.23,
        wheel_radius=0.035,
    )


@configclass
class IRobotNavEnvCfg(NavEnvCfg):
    """Ridgeback UR navigation: virtual/dummy joints, NonHolonomicAction."""

    scene: RidgebackSceneCfg = RidgebackSceneCfg(num_envs=1024, env_spacing=4.0)
    actions: RidgebackActionsCfg = RidgebackActionsCfg()
