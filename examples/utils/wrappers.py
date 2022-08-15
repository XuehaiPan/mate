import re
from collections import OrderedDict

import gym
import numpy as np
from gym import spaces
from ray import rllib

import mate

from .callbacks import MetricCollector


__all__ = ['RLlibMultiAgentAPI', 'RLlibMultiAgentCentralizedTraining', 'FrameSkip']


class RLlibHomogeneousMultiAgentEnv(rllib.MultiAgentEnv):
    def observation_space_sample(self, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.get_agent_ids()

        observation = {agent_id: self.observation_space.sample() for agent_id in agent_ids}
        return observation

    def action_space_sample(self, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.get_agent_ids()

        actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
        return actions

    def action_space_contains(self, x):
        if not isinstance(x, dict):
            return False

        return all(map(self.action_space.contains, x.values()))

    def observation_space_contains(self, x):
        if not isinstance(x, dict):
            return False

        return all(map(self.observation_space.contains, x.values()))


class RLlibMultiAgentAPI(gym.Wrapper, RLlibHomogeneousMultiAgentEnv, metaclass=mate.WrapperMeta):
    def __init__(self, env):
        mate.wrappers.typing.assert_multi_agent_environment(env)
        assert isinstance(env, mate.RepeatedRewardIndividualDone), (
            f'You should use wrapper `{self.__class__}` with wrapper `RepeatedRewardIndividualDone`. '
            f'Please wrap the environment with wrapper `RepeatedRewardIndividualDone` first. '
            f'Got env = {env}.'
        )
        assert isinstance(env, (mate.MultiCamera, mate.MultiTarget)), (
            f'You should provide a single-team multi-agent environment '
            f'(i.e. `mate.MultiCamera` or `mate.MultiTarget`). '
            f'Got env = {env}.'
        )

        super().__init__(env)

        self.id_format = (
            'camera_{}'.format if isinstance(env, mate.MultiCamera) else 'target_{}'.format
        )

        self.observation_space = env.observation_space[0]
        self.action_space = env.action_space[0]
        self.agent_ids = list(self.seq2dict(env.observation_space).keys())

        self._agent_ids = set(self.agent_ids)
        setattr(self.unwrapped, '_agent_ids', self._agent_ids)
        setattr(self.unwrapped, 'get_agent_ids', lambda: self._agent_ids)

    def load_config(self, config=None):
        self.env.load_config(config=config)

        self.__init__(self.env)

    def reset(self, **kwargs):
        return self.seq2dict(self.env.reset(**kwargs))

    def step(self, action):
        action = np.asarray(list(map(action.get, self.agent_ids)))
        observations, rewards, dones, infos = tuple(map(self.seq2dict, self.env.step(action)))
        dones['__all__'] = all(dones.values())
        return observations, rewards, dones, infos

    def seq2dict(self, seq):
        return OrderedDict([(self.id_format(i), item) for i, item in enumerate(seq)])


class RLlibMultiAgentCentralizedTraining(
    gym.Wrapper, RLlibHomogeneousMultiAgentEnv, metaclass=mate.WrapperMeta
):
    def __init__(
        self, env, normalize_state=True, add_joint_observation=False, add_action_mask=False
    ):
        assert isinstance(env, RLlibMultiAgentAPI), (
            f'You should use wrapper `{self.__class__}` with wrapper `RLlibMultiAgentAPI`. '
            f'Please wrap the environment with wrapper `RLlibMultiAgentAPI` first. '
            f'Got env = {env}.'
        )
        assert isinstance(env, (mate.MultiCamera, mate.MultiTarget)), (
            f'You should provide a single-team multi-agent environment '
            f'(i.e. `mate.MultiCamera` or `mate.MultiTarget`). '
            f'Got env = {env}.'
        )

        super().__init__(env)

        self.agent_ids = env.agent_ids
        self._agent_ids = env._agent_ids

        self.normalize_state = normalize_state
        if self.normalize_state:
            self.state_space = spaces.Box(
                low=mate.normalize_observation(env.state_space.low, env.state_space),
                high=mate.normalize_observation(env.state_space.high, env.state_space),
                dtype=env.state_space.dtype,
            )
        else:
            self.state_space = env.state_space

        self.action_space = env.action_space
        self.others_joint_observation_space = spaces.Tuple(
            spaces=(env.observation_space,) * (self.num_teammates - 1)
        )
        self.others_joint_action_space = spaces.Tuple(
            spaces=(self.action_space,) * (self.num_teammates - 1)
        )

        subspaces = OrderedDict(
            [
                # Local observation of the current agent
                ('obs', env.observation_space),
                # Global state of the environment
                ('state', self.state_space),
                # Joint action for other agents (exclude the current agent)
                # Can be shift with callback `ShiftAgentActionTimestep`
                ('prev_others_joint_action', self.others_joint_action_space),
            ]
        )

        if add_action_mask:
            assert (
                hasattr(env, 'action_mask_space')
                and hasattr(env, 'action_mask')
                and callable(env.action_mask)
            )
            self.has_action_mask = True
            subspaces['action_mask'] = env.action_mask_space
        else:
            self.has_action_mask = False

        self.add_joint_observation = add_joint_observation
        if self.add_joint_observation:
            subspaces['others_joint_observation'] = self.others_joint_observation_space

        self.observation_space = spaces.Dict(spaces=subspaces)

    def load_config(self, config=None):
        self.env.load_config(config=config)

        self.__init__(
            self.env,
            normalize_state=self.normalize_state,
            add_joint_observation=self.add_joint_observation,
            add_action_mask=self.has_action_mask,
        )

    def state(self):
        state = self.env.state()
        if self.normalize_state:
            state = mate.normalize_observation(state, self.env.state_space)
        return state

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)

        if self.add_joint_observation:
            joint_observation = tuple(observations[agent_id] for agent_id in self.agent_ids)
            cycled_joint_observation = joint_observation + joint_observation
        else:
            cycled_joint_observation = None

        dummy_prev_others_joint_action = self.observation_space['prev_others_joint_action'].sample()
        zeros_prev_others_joint_action = tuple(map(np.zeros_like, dummy_prev_others_joint_action))

        state = self.state()
        for i, agent_id in enumerate(self.agent_ids):
            local_observation = observations[agent_id]
            observation = OrderedDict(
                [
                    ('obs', local_observation),
                    ('state', state),
                    # Can be shift with callback `ShiftAgentActionTimestep`
                    ('prev_others_joint_action', zeros_prev_others_joint_action),
                ]
            )

            if self.has_action_mask:
                action_mask = self.action_mask(local_observation)
                observation['action_mask'] = action_mask

            if self.add_joint_observation:
                observation['others_joint_observation'] = cycled_joint_observation[
                    i + 1 : i + self.num_teammates
                ]

            observations[agent_id] = observation

        return observations

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        if self.add_joint_observation:
            joint_observation = tuple(observations[agent_id] for agent_id in self.agent_ids)
            cycled_joint_observation = joint_observation + joint_observation
        else:
            cycled_joint_observation = None

        joint_action = tuple(action[agent_id] for agent_id in self.agent_ids)
        cycled_joint_action = joint_action + joint_action

        state = self.state()
        for i, agent_id in enumerate(self.agent_ids):
            local_observation = observations[agent_id]
            observation = OrderedDict(
                [
                    ('obs', local_observation),
                    ('state', state),
                    # Can be shift with callback `ShiftAgentActionTimestep`
                    (
                        'prev_others_joint_action',
                        cycled_joint_action[i + 1 : i + self.num_teammates],
                    ),
                ]
            )

            if self.has_action_mask:
                action_mask = self.action_mask(local_observation)
                observation['action_mask'] = action_mask

            if self.add_joint_observation:
                observation['others_joint_observation'] = cycled_joint_observation[
                    i + 1 : i + self.num_teammates
                ]

            observations[agent_id] = observation

        return observations, rewards, dones, infos


class FrameSkip(gym.Wrapper, metaclass=mate.WrapperMeta):
    INFO_KEYS = {
        'raw_reward': 'sum',
        'normalized_raw_reward': 'sum',
        re.compile(r'^auxiliary_reward(\w*)$'): 'sum',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
    }

    def __init__(self, env, frame_skip=1):
        from examples.hrl.wrappers import HierarchicalCamera

        assert isinstance(env, (mate.MultiCamera, mate.MultiTarget)), (
            f'You should provide a single-team multi-agent environment '
            f'(i.e. `mate.MultiCamera` or `mate.MultiTarget`). '
            f'Got env = {env}.'
        )
        assert not isinstance(env, HierarchicalCamera), (
            f'You should not use wrapper `{self.__class__}` with wrapper `HierarchicalCamera`. '
            f'Got env = {env}.'
        )
        assert frame_skip > 0, (
            f'The argument `frame_skip` should be a positive number. '
            f'Got frame_skip = {frame_skip}.'
        )

        super().__init__(env)

        self.last_observations = None

        self.frame_skip = frame_skip

    def load_config(self, config=None):
        self.env.load_config(config=config)

        self.__init__(self.env, frame_skip=self.frame_skip)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.last_observations = observation

        return observation

    def step(self, action):
        fragment_rewards = []
        info_collectors = [
            MetricCollector(self.INFO_KEYS) for _ in range(len(self.last_observations))
        ]
        for f in range(self.frame_skip):
            observations, rewards, dones, infos = self.env.step(action)
            fragment_rewards.append(rewards)

            for collector, info in zip(info_collectors, infos):
                collector.add(info)

            if all(dones):
                break

        self.last_observations = observations
        for collector, info in zip(info_collectors, infos):
            info.update(collector.collect())

        rewards = np.sum(fragment_rewards, axis=0)
        if isinstance(rewards, np.ndarray):
            rewards = rewards.tolist()
        return observations, rewards, dones, infos

    def __str__(self) -> str:
        return f'<{self.__class__.__name__}({self.frame_skip}){self.env}>'
