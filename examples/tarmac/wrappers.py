from collections import OrderedDict

import gym
import numpy as np
from gym import spaces

import mate
from examples.utils.wrappers import (
    RLlibHomogeneousMultiAgentEnv,
    RLlibMultiAgentCentralizedTraining,
)


class ActionWithMessage(gym.Wrapper, RLlibHomogeneousMultiAgentEnv, metaclass=mate.WrapperMeta):
    def __init__(self, env, message_dim):
        assert isinstance(env, RLlibMultiAgentCentralizedTraining), (
            f'You should use wrapper `{self.__class__}` with wrapper `RLlibMultiAgentCentralizedTraining`. '
            f'Please wrap the environment with wrapper `RLlibMultiAgentCentralizedTraining` first. '
            f'Got env = {env}.'
        )

        super().__init__(env)

        self.agent_ids = env.agent_ids
        self._agent_ids = env._agent_ids

        self.message_dim = message_dim
        self.message_space = spaces.Box(-1.0, +1.0, shape=(self.message_dim,), dtype=np.float64)

        self.action_space = spaces.Dict(
            spaces=OrderedDict(
                [
                    ('action', env.action_space),
                    ('message', self.message_space),
                ]
            )
        )

        subspaces = env.observation_space.spaces.copy()
        subspaces['messages'] = spaces.Tuple((self.message_space,) * self.num_teammates)
        self.observation_space = spaces.Dict(spaces=subspaces)

    def load_config(self, config=None):
        self.env.load_config(config=config)

        self.__init__(self.env, message_dim=self.message_dim)

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)

        dummy_messages = self.observation_space['messages'].sample()
        zeros_messages = tuple(map(np.zeros_like, dummy_messages))

        for observation in observations.values():
            observation['messages'] = zeros_messages

        return observations

    def step(self, action):
        messages = tuple(action[agent_id]['message'] for agent_id in self.agent_ids)

        observations, rewards, dones, infos = self.env.step(
            {agent_id: action[agent_id]['action'] for agent_id in self.agent_ids}
        )

        for observation in observations.values():
            observation['messages'] = messages  # broadcasting

        return observations, rewards, dones, infos
