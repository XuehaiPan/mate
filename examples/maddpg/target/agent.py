import copy

from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy

import mate
from examples.maddpg.target.config import config as _config
from examples.maddpg.target.config import make_env as _make_env
from examples.utils import RLlibPolicyMixIn


class MADDPGTargetAgent(RLlibPolicyMixIn, mate.TargetAgentBase):
    """MADDPG/MA-TD3 Target Agent

    A wrapper for the trained RLlib policy.

    Note:
        The agent always produces a primitive continuous action. If the RLlib policy is trained with
        discrete actions, the output action will be converted to primitive continuous action.
    """

    POLICY_CLASS = DDPGTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(self, config=None, checkpoint_path=None, make_env=_make_env, seed=None):
        super().__init__(
            config=config, checkpoint_path=checkpoint_path, make_env=make_env, seed=seed
        )

        self.frame_skip = self.config.get('env_config', {}).get('frame_skip', 1)
        self.discrete_levels = self.config.get('env_config', {}).get('discrete_levels', None)
        assert self.discrete_levels is None, 'DDPG/TD3 only supports continuous actions.'
        self.normalized_action_grid = None

        self.last_action = None

    def reset(self, observation):
        super().reset(observation)

        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        if self.episode_step % self.frame_skip == 0:
            self.last_action, self.hidden_state = self.compute_single_action(
                observation, state=self.hidden_state, info=info, deterministic=deterministic
            )

            if self.normalized_action_grid is not None:
                # Convert discretized action to primitive continuous action
                self.last_action = (
                    self.action_space.high * self.normalized_action_grid[self.last_action]
                )

        return self.last_action
