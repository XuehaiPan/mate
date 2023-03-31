import copy

import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor

import mate
from examples.tarmac.camera.config import config as _config
from examples.tarmac.camera.config import make_env as _make_env
from examples.tarmac.models import TarMACModel
from examples.utils import RLlibPolicyMixIn


class TarMACCameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """TarMAC Camera Agent

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
        assert isinstance(self.model, TarMACModel)
        assert isinstance(self.preprocessor, DictFlatteningPreprocessor)
        self.flat_obs_slices = self.model.flat_obs_slices

        self.frame_skip = self.config.get('env_config', {}).get('frame_skip', 1)
        self.discrete_levels = self.config.get('env_config', {}).get('discrete_levels', None)
        if self.discrete_levels is not None:
            self.normalized_action_grid = mate.DiscreteCamera.discrete_action_grid(
                levels=self.discrete_levels
            )
        else:
            self.normalized_action_grid = None

        self.last_action = None
        self.last_action = None
        self.zeros_message = np.zeros_like(self.model.action_space['message'].sample())
        self.last_message = self.zeros_message.copy()

    def reset(self, observation):
        super().reset(observation)

        self.last_action = None
        self.last_message = self.zeros_message.copy()

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        if self.episode_step % self.frame_skip == 0:
            action, self.hidden_state = self.compute_single_action(
                observation, state=self.hidden_state, info=info, deterministic=deterministic
            )
            self.last_action = action['action']
            self.last_message = action['message']

            if self.normalized_action_grid is not None:
                # Convert discretized action to primitive continuous action
                self.last_action = (
                    self.action_space.high * self.normalized_action_grid[self.last_action]
                )

        return self.last_action

    def send_responses(self):
        if self.episode_step % self.frame_skip != 0:
            return []

        message = self.pack_message(
            content=self.last_message.copy(),  # output of the previous step (frame-skipped)
            recipient=None,  # broadcasting
        )

        return [message]

    def preprocess_observation(self, observation):
        preprocessed_observation = self.preprocess_raw_observation(observation)

        dummy_preprocessed_observation = np.zeros(
            shape=self.preprocessor.observation_space.shape,
            dtype=self.preprocessor.observation_space.dtype,
        )
        dummy_preprocessed_observation[
            self.flat_obs_slices['obs']
        ] = preprocessed_observation.ravel()

        messages = sorted(self.last_responses, key=lambda m: m.sender)
        joint_messages = np.ravel([m.content for m in messages])
        dummy_preprocessed_observation[self.flat_obs_slices['messages']] = joint_messages

        return dummy_preprocessed_observation.ravel()
