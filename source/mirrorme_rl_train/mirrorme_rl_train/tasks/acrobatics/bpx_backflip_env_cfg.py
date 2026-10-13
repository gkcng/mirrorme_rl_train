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
* Curriculum: Progressive gravity ramp from Mars (-4.90 m/s^2) to Earth (-9.81 m/s^2).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from ..locomotion.bpx_flat_env_cfg import (
    BASE_BODY_NAME,
    FOOT_BODY_NAMES,
    JOINT_NAMES,
    ActionsCfg,
    BPXSceneCfg,
    EventCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Backflip Phase Clock, Rewards, & Curriculum
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
) -> torch.Tensor:
    """Reward positive vertical launch velocity during the takeoff window."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_takeoff = (phase < max_phase).float()
    return torch.clamp(vz / target_vz, min=0.0, max=1.5) * in_takeoff


def backflip_pitch_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_pitch_rate: float = -7.0,
    max_phase: float = 0.75,
) -> torch.Tensor:
    """Reward backward pitch rotation rate around the body Y-axis."""
    asset = env.scene[asset_cfg.name]
    pitch_rate = asset.data.root_ang_vel_b[:, 1]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_spin = (phase < max_phase).float()
    return torch.clamp(-pitch_rate / (-target_pitch_rate), min=0.0, max=2.0) * in_spin


def backflip_flight_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.70,
    min_phase: float = 0.25,
    max_phase: float = 0.80,
) -> torch.Tensor:
    """Reward torso apex flight height above the ground during mid-air flip."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    in_flight = ((phase >= min_phase) & (phase < max_phase)).float()
    return torch.clamp(height / target_height, min=0.0, max=1.5) * in_flight


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


def early_base_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    max_phase: float = 0.25,
) -> torch.Tensor:
    """Terminate only if the torso hits the floor during the initial crouch/takeoff."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    phase = env.episode_length_buf.float() / float(env.max_episode_length)
    return (net_force > threshold) & (phase < max_phase)


def ramp_gravity_assistance(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
    ramp_iterations: int = 3000,
    start_assist_acc: float = 4.90,
) -> torch.Tensor:
    """Gradually ramp buoyant upward acceleration from 4.90 m/s^2 down to 0.0 m/s^2 (Earth gravity)."""
    # 24 steps per iteration in PPO
    current_iter = env.common_step_counter // 24
    progress = min(1.0, current_iter / float(ramp_iterations))
    assist_acc = start_assist_acc * (1.0 - progress)

    asset = env.scene[asset_cfg.name]
    if assist_acc > 0.01:
        body_id = asset_cfg.body_ids[0]
        torso_mass = asset.data.default_mass[env_ids, body_id]
        upward_force = torso_mass * assist_acc

        forces = torch.zeros((len(env_ids), len(asset_cfg.body_ids), 3), device=env.device)
        forces[:, 0, 2] = upward_force
        torques = torch.zeros_like(forces)
        asset.set_external_force_and_torque(forces, torques, body_ids=asset_cfg.body_ids, env_ids=env_ids)

    return torch.tensor(assist_acc, device=env.device)


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
        # Phase clock preserves the 3-D command interface of bpx_flat (3D)
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
    """Reward terms shaping the four backflip phases (crouch, thrust, spin, land)."""

    # Primary Backflip Objectives
    takeoff = RewTerm(func=takeoff_thrust, weight=3.0, params={"target_vz": 2.0, "max_phase": 0.35})
    pitch_spin = RewTerm(func=backflip_pitch_rate, weight=4.0, params={"target_pitch_rate": -7.0, "max_phase": 0.75})
    flight = RewTerm(func=backflip_flight_height, weight=2.0, params={"target_height": 0.70})
    landing = RewTerm(
        func=landing_upright_contact,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True)},
    )

    # Smoothness & Safety Regularization
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
            "threshold": 5.0,
            "max_phase": 0.25,
        },
    )
    # Notice: bad_orientation termination is intentionally absent to allow full 360-degree flip!


@configclass
class BackflipCurriculumCfg:
    """Gravity assistance curriculum transitioning smoothly from Mars to Earth."""

    gravity = CurrTerm(
        func=ramp_gravity_assistance,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "ramp_iterations": 3000,
            "start_assist_acc": 4.90,
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
    events: EventCfg = EventCfg()
    curriculum: BackflipCurriculumCfg = BackflipCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 1.6  # 80 policy steps at 50 Hz
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
