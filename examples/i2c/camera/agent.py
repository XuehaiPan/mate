import copy

import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor

import mate
from examples.i2c.camera.config import config as _config
from examples.i2c.camera.config import make_env as _make_env
from examples.i2c.models import I2CModel
from examples.utils import RLlibPolicyMixIn


class I2CCameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """I2C Camera Agent

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
        assert isinstance(self.model, I2CModel)
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

    def send_requests(self):
        if self.episode_step % self.frame_skip != 0:
            return []

        preprocessed_observation = self.preprocess_raw_observation(self.last_observation)
        comm_mask = self.model.get_communication_mask(preprocessed_observation)

        requests = []
        for i in np.flatnonzero(comm_mask):
            message = self.pack_message(
                content=None, recipient=(self.index + 1 + i) % self.num_teammates
            )
            requests.append(message)

        return requests

    def send_responses(self):
        if self.episode_step % self.frame_skip != 0:
            return []

        responses = []
        for request in self.last_requests:
            message = self.pack_message(
                content=self.last_observation.copy(), recipient=request.sender
            )
            responses.append(message)

        return responses

    def preprocess_observation(self, observation):
        preprocessed_observation = self.preprocess_raw_observation(observation)

        dummy_preprocessed_observation = np.zeros(
            shape=self.preprocessor.observation_space.shape,
            dtype=self.preprocessor.observation_space.dtype,
        )
        dummy_preprocessed_observation[
            self.flat_obs_slices['obs']
        ] = preprocessed_observation.ravel()

        joint_observation = np.zeros(
            shape=(self.num_teammates, self.observation_dim),
            dtype=dummy_preprocessed_observation.dtype,
        )
        joint_observation[self.index] = observation
        for message in self.last_responses:
            joint_observation[message.sender] = message.content
        preprocessed_joint_observation = tuple(self.preprocess_raw_observation(joint_observation))
        cycled_preprocessed_joint_observation = (
            preprocessed_joint_observation + preprocessed_joint_observation
        )
        dummy_preprocessed_observation[self.flat_obs_slices['others_joint_observation']] = np.ravel(
            cycled_preprocessed_joint_observation[self.index + 1 : self.index + self.num_teammates]
        )

        return dummy_preprocessed_observation.ravel()
