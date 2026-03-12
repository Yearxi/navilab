# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO Runner config for NaviLab navigation (LiDAR + goal, obstacle avoidance).

Tuned for: high-dim LiDAR obs (~271) + goal_relative_pose (3), continuous (v, omega) action,
episode_length_s=30, decimation=4. See docstrings below for rationale.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Runner: 24 steps × num_envs per iteration for richer rollout (goal-reaching needs longer horizon)
    num_steps_per_env = 24
    # Navigation + obstacle avoidance needs many iterations (align with Isaac navigation tasks)
    max_iterations = 1500
    save_interval = 50
    experiment_name = "navilab_irobot"

    # Policy: larger net for LiDAR; moderate init noise for stable exploration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )