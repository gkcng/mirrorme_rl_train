"""MIRRORME RL Train manager-based backflip acrobatic task for BPX.

Training interface
------------------
* Actor observation group: exactly 45 dimensions.
  - Base angular velocity (3D)
  - Projected gravity (3D)
  - Phase clock: [sin(2*pi*phase), cos(2*pi*phase), 1.0] (3D)
  - Joint positions relative to default (12D)
  - Joint velocities (12D)
  - Previous raw action (12D)
* Action: 12 joint-position residuals.
* Critic: actor observations plus privileged base linear velocity (48 dimensions).
* Curriculum: Progressive difficulty ramp scaling launch thrust, apex height, and pitch rate.
* Discipline: Zero-twist single-axis rotation, spot-landing proximity, and firm touchdown stabilization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from ..locomotion.bpx_flat_env_cfg import (
    BASE_BODY_NAME,
    FOOT_BODY_NAMES,
    JOINT_NAMES,
    ActionsCfg,
    BPXSceneCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Backflip Phase Clock, Rewards, & Progressive Curriculum
# -----------------------------------------------------------------------------

def flip_phase_clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a periodic phase clock [sin(2*pi*phase), cos(2*pi*phase), 1.0]."""
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    return torch.stack([
        torch.sin(2.0 * math.pi * phase),
        torch.cos(2.0 * math.pi * phase),
        torch.ones_like(phase),
    ], dim=1)


def takeoff_thrust(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_vz: float = 2.0,
    max_phase: float = 0.35,
    ramp_iterations: int = 3000,
) -> torch.Tensor:
    """Reward positive vertical launch velocity, progressively increasing target vz."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_takeoff = (phase < max_phase).float()

    current_iter = env.common_step_counter // 24
    progress = min(1.0, current_iter / float(ramp_iterations))
    effective_target_vz = 1.0 + progress * (target_vz - 1.0)

    return torch.clamp(vz / effective_target_vz, min=0.0, max=1.5) * in_takeoff


def backflip_pitch_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_pitch_rate: float = -7.0,
    max_phase: float = 0.75,
    ramp_iterations: int = 3000,
) -> torch.Tensor:
    """Reward backward pitch rotation rate, progressively demanding faster rotation."""
    asset = env.scene[asset_cfg.name]
    pitch_rate = asset.data.root_ang_vel_b[:, 1]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_spin = (phase < max_phase).float()

    current_iter = env.common_step_counter // 24
    progress = min(1.0, current_iter / float(ramp_iterations))
    effective_target_rate = -3.5 + progress * (target_pitch_rate - (-3.5))

    return torch.clamp(-pitch_rate / (-effective_target_rate), min=0.0, max=2.0) * in_spin


def backflip_flight_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.70,
    min_phase: float = 0.25,
    max_phase: float = 0.80,
    ramp_iterations: int = 3000,
) -> torch.Tensor:
    """Reward torso apex flight height, progressively demanding higher clearance."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_flight = ((phase >= min_phase) & (phase < max_phase)).float()

    current_iter = env.common_step_counter // 24
    progress = min(1.0, current_iter / float(ramp_iterations))
    effective_target_h = 0.55 + progress * (target_height - 0.55)

    return torch.clamp(height / effective_target_h, min=0.0, max=1.5) * in_flight


def landing_upright_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.80,
) -> torch.Tensor:
    """Reward feet touchdown with an upright torso orientation during landing."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0).any(dim=1).float()
    proj_grav_z = asset.data.projected_gravity_b[:, 2]
    upright = torch.clamp(-proj_grav_z, min=0.0, max=1.0)
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_landing = (phase >= min_phase).float()
    return upright * feet_contact * in_landing


# -----------------------------------------------------------------------------
# Precision & Stabilization Rewards (Addressing a, b, c)
# -----------------------------------------------------------------------------

def flight_twist_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.15,
    max_phase: float = 0.85,
) -> torch.Tensor:
    """(a) Penalize parasitic roll (ang_vel_x) and yaw (ang_vel_z) spinning during flight."""
    asset = env.scene[asset_cfg.name]
    ang_vel_b = asset.data.root_ang_vel_b
    roll_rate = ang_vel_b[:, 0]
    yaw_rate = ang_vel_b[:, 2]
    twist_sq = roll_rate.square() + yaw_rate.square()
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_flight = ((phase >= min_phase) & (phase < max_phase)).float()
    return twist_sq * in_flight


def landing_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.80,
    std: float = 0.35,
) -> torch.Tensor:
    """(a) Reward landing facing the exact same forward direction as starting heading."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    yaw_error_sq = yaw.square()
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_landing = (phase >= min_phase).float()
    return torch.exp(-yaw_error_sq / (std ** 2)) * in_landing


def landing_spot_drift_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.80,
) -> torch.Tensor:
    """(b) Penalize horizontal (X, Y) landing displacement away from the origin."""
    asset = env.scene[asset_cfg.name]
    drift_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    drift_sq = torch.sum(drift_xy.square(), dim=1)
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_landing = (phase >= min_phase).float()
    return drift_sq * in_landing


def landing_base_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.85,
) -> torch.Tensor:
    """(c) Penalize residual base linear velocity upon touchdown (eliminating landing bounce)."""
    asset = env.scene[asset_cfg.name]
    lin_vel_sq = torch.sum(asset.data.root_lin_vel_w.square(), dim=1)
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_settle = (phase >= min_phase).float()
    return lin_vel_sq * in_settle


def landing_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.85,
) -> torch.Tensor:
    """(c) Penalize residual joint velocities upon touchdown (eliminating leg scrambling)."""
    asset = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(asset.data.joint_vel[:, asset_cfg.joint_ids].square(), dim=1)
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_settle = (phase >= min_phase).float()
    return joint_vel_sq * in_settle


def landing_nominal_pose_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_phase: float = 0.85,
) -> torch.Tensor:
    """(c) Reward returning joints to nominal standing pose during touchdown."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_settle = (phase >= min_phase).float()
    return deviation * in_settle


def early_base_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 15.0,
    max_phase: float = 0.25,
) -> torch.Tensor:
    """Terminate only if the torso hits the floor during the initial crouch/takeoff."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    return (net_force > threshold) & (phase < max_phase)


def log_curriculum_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    ramp_iterations: int = 3000,
) -> torch.Tensor:
    """Track and log curriculum progression across training iterations."""
    current_iter = env.common_step_counter // 24
    progress = min(1.0, current_iter / float(ramp_iterations))
    return torch.tensor(progress, device=env.device)


# -----------------------------------------------------------------------------
# Configuration Classes
# -----------------------------------------------------------------------------

@configclass
class BackflipObservationsCfg:
    """Deployable 45-D actor observations and 48-D critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        flip_clock = ObsTerm(
            func=flip_phase_clock,
            clip=(-10.0, 10.0),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
            scale=0.05,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-10.0, 10.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class BackflipRewardsCfg:
    """Reward suite shaping backflip dynamics, flight discipline, and stable landing."""

    # 1. Primary Backflip Flight Objectives (Curriculum)
    takeoff = RewTerm(func=takeoff_thrust, weight=3.0, params={"target_vz": 2.0, "max_phase": 0.35, "ramp_iterations": 3000})
    pitch_spin = RewTerm(func=backflip_pitch_rate, weight=4.0, params={"target_pitch_rate": -7.0, "max_phase": 0.75, "ramp_iterations": 3000})
    flight = RewTerm(func=backflip_flight_height, weight=2.0, params={"target_height": 0.70, "ramp_iterations": 3000})
    landing = RewTerm(
        func=landing_upright_contact,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True)},
    )

    # 2. Heading Discipline (Fix a: eliminate corkscrew twist & land facing forward)
    flight_twist_penalty = RewTerm(func=flight_twist_penalty, weight=-0.05)
    landing_heading = RewTerm(func=landing_heading_alignment, weight=2.0, params={"std": 0.35})

    # 3. Spot Landing (Fix b: eliminate horizontal drift away from start)
    landing_spot_drift = RewTerm(func=landing_spot_drift_l2, weight=-2.0)

    # 4. Stick the Landing (Fix c: eliminate bounce, leg scrambling, and settle into stance)
    landing_base_vel = RewTerm(func=landing_base_velocity_l2, weight=-0.5)
    landing_joint_vel = RewTerm(func=landing_joint_vel_l2, weight=-0.01)
    landing_nominal_pose = RewTerm(func=landing_nominal_pose_l1, weight=-0.2)

    # 5. Smoothness & Safety Regularization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    undesired_leg_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(hip|thigh|calf).*"),
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class BackflipTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    early_contact = DoneTerm(
        func=early_base_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_NAME),
            "threshold": 15.0,
            "max_phase": 0.25,
        },
    )


@configclass
class BackflipCurriculumCfg:
    """Progressive difficulty curriculum tracking."""

    progress = CurrTerm(
        func=log_curriculum_progress,
        params={"ramp_iterations": 3000},
    )


@configclass
class BackflipEventCfg:
    """(Fix d) Dynamic domain randomization and per-episode perturbations."""

    # Physics material randomization at startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.40, 1.60),
            "dynamic_friction_range": (0.30, 1.35),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 128,
        },
    )

    # Payload & Center-of-Mass randomization at startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-1.5, 2.5),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "com_range": {"x": (-0.035, 0.035), "y": (-0.035, 0.035), "z": (-0.035, 0.035)},
        },
    )

    # Every Episode Reset: randomize initial pose, heading, and root velocities
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "yaw": (-0.25, 0.25),  # Random initial heading to force directional awareness
            },
            "velocity_range": {
                "x": (-0.20, 0.20),
                "y": (-0.20, 0.20),
                "z": (-0.05, 0.05),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.20, 0.20),
            },
        },
    )

    # Every Episode Reset: randomize joint angles and velocities
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.92, 1.08),
            "velocity_range": (-0.25, 0.25),
        },
    )


@configclass
class BPXBackflipEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the BPX backflip task."""

    scene: BPXSceneCfg = BPXSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BackflipObservationsCfg = BackflipObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: BackflipRewardsCfg = BackflipRewardsCfg()
    terminations: BackflipTerminationsCfg = BackflipTerminationsCfg()
    events: BackflipEventCfg = BackflipEventCfg()
    curriculum: BackflipCurriculumCfg = BackflipCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 1.6  # 80 policy steps at 50 Hz
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
