import copy

import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy

import mate
from examples.hrl.tarmac.camera.config import config as _config
from examples.hrl.tarmac.camera.config import make_env as _make_env
from examples.hrl.wrappers import HierarchicalCamera
from examples.utils import RLlibPolicyMixIn


class HRLTarMACCameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """Hierarchical TarMAC Camera Agent

    A wrapper for the trained RLlib policy.

    Note:
        The agent always produces a primitive continuous action. If the RLlib policy is trained with
        discrete actions, the output action will be converted to primitive continuous action.
    """

    POLICY_CLASS = PPOTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(self, config=None, checkpoint_path=None, make_env=_make_env, seed=None):
        super().__init__(
            config=config, checkpoint_path=checkpoint_path, make_env=make_env, seed=seed
        )

        self.frame_skip = self.config.get('env_config', {}).get('frame_skip', 1)

        self.multi_selection = self.config.get('env_config', {}).get('multi_selection', False)
        self.last_action = None
        self.last_selection = None
        self.last_mask = None
        self.index2onehot = None

    def reset(self, observation):
        super().reset(observation)

        self.index2onehot = np.eye(self.num_targets + 1, self.num_targets, dtype=np.bool8)
        self.last_action = None
        self.last_selection = None
        self.last_mask = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        self.last_mask = observation[self.observation_slices['opponent_mask']].astype(np.bool8)

        if self.episode_step % self.frame_skip == 0:
            self.last_selection, self.hidden_state = self.compute_single_action(
                observation, state=self.hidden_state, info=info, deterministic=deterministic
            )

            if not self.multi_selection:
                self.last_selection = self.index2onehot[self.last_selection]
            else:
                self.last_selection = np.asarray(self.last_selection, dtype=np.bool8)

        # Convert target selection to primitive continuous action
        target_states, tracked_bits = self.get_all_opponent_states(observation)
        self.last_action = HierarchicalCamera.executor(
            self.state,
            target_states,
            target_selection_bits=self.last_selection,
            target_view_mask=tracked_bits,
        )

        return self.last_action
