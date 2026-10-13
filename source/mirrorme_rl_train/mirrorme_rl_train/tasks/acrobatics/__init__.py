"""Gym registration for the BPX backflip acrobatic task."""

import gymnasium as gym

from . import agents


gym.register(
    id="bpx_backflip",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bpx_backflip_env_cfg:BPXBackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BPXBackflipPPORunnerCfg",
    },
)
