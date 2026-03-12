# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Differential-drive action: map (v, omega) to 4-wheel velocity targets for skid-steer robots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DiffDriveWheelVelocityAction(ActionTerm):
    """Maps 2D action (v, omega) to 4-wheel joint velocity targets using skid-steer kinematics."""

    cfg: "DiffDriveWheelVelocityActionCfg"
    _asset: Articulation
    _left_joint_ids: list[int]
    _right_joint_ids: list[int]
    _joint_ids: list[int]
    _raw_actions: torch.Tensor
    _processed_actions: torch.Tensor
    _wheel_velocities: torch.Tensor

    def __init__(self, cfg: "DiffDriveWheelVelocityActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        lid, _ = self._asset.find_joints(cfg.left_wheel_joint_names, preserve_order=True)
        rid, _ = self._asset.find_joints(cfg.right_wheel_joint_names, preserve_order=True)
        self._left_joint_ids = lid
        self._right_joint_ids = rid
        self._joint_ids = lid + rid
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wheel_velocities = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._scale = torch.tensor(cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(cfg.offset, device=self.device).unsqueeze(0)
        self._clip_lo = torch.tensor([-float("inf"), -float("inf")], device=self.device)
        self._clip_hi = torch.tensor([float("inf"), float("inf")], device=self.device)
        if cfg.clip is not None and isinstance(cfg.clip, dict):
            if ".*" in cfg.clip:
                v = cfg.clip[".*"]
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    self._clip_lo[:] = v[0]
                    self._clip_hi[:] = v[1]

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset
        self._processed_actions[:] = torch.clamp(
            self._processed_actions,
            min=self._clip_lo.unsqueeze(0),
            max=self._clip_hi.unsqueeze(0),
        )

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        omega = self._processed_actions[:, 1]
        r = self.cfg.wheel_radius
        L = self.cfg.track_width
        left_ang = (v - omega * L / 2) / r
        right_ang = (v + omega * L / 2) / r
        n_left = len(self._left_joint_ids)
        for i in range(n_left):
            self._wheel_velocities[:, i] = left_ang
        for i in range(n_left, len(self._joint_ids)):
            self._wheel_velocities[:, i] = right_ang
        self._asset.set_joint_velocity_target(self._wheel_velocities, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0


@configclass
class DiffDriveWheelVelocityActionCfg(ActionTermCfg):
    """Config for (v, omega) -> 4 wheel joint velocity action (skid-steer)."""

    class_type: type = DiffDriveWheelVelocityAction

    left_wheel_joint_names: list[str] = MISSING
    """Joint names for left side wheels (e.g. front_left, rear_left)."""
    right_wheel_joint_names: list[str] = MISSING
    """Joint names for right side wheels (e.g. front_right, rear_right)."""
    scale: tuple[float, float] = (1.0, 1.0)
    """Scale for (linear_vel, angular_vel)."""
    offset: tuple[float, float] = (0.0, 0.0)
    """Offset for (linear_vel, angular_vel)."""
    clip: dict | None = None
    """Clip (min, max) for (linear_vel, angular_vel), e.g. {".*": (-1.5, 1.5)}."""
    track_width: float = 0.2
    """Distance between left and right wheels (m)."""
    wheel_radius: float = 0.05
    """Wheel radius (m), for linear vel to joint angular vel."""
