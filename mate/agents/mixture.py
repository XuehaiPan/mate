"""Agent wrapper to built a mixture of agents."""

from abc import ABCMeta
from typing import List, Optional

import numpy as np
from gym.utils import seeding

from mate.agents.base import AgentBase, AgentType, CameraAgentBase, TargetAgentBase


__all__ = ['MixtureCameraAgent', 'MixtureTargetAgent']


# pylint: disable-next=too-many-instance-attributes
class MixtureAgentMixIn(AgentBase, metaclass=ABCMeta):
    """Helper class for mixture of agents.

    Randomly choose a underlying agent in candidates at episode start.
    """

    def __init__(self, candidates, weights=None, mixture_seed=None, seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """

        candidates = list(candidates)
        if weights is None:
            weights = [1.0] * len(candidates)
        weights = np.array(weights, dtype=np.float64, copy=True).ravel()
        assert len(candidates) == len(weights), (
            f'The number of sample weights must be equal to the number of agent candidates. '
            f'Got weights = {weights}.'
        )
        assert (weights >= 0.0).all() and weights.any(), (
            f'The sample weights for agent candidates must be non-negative and '
            f'should have at least one positive value. '
            f'Got weights = {weights}.'
        )

        self.candidates = [candidates[a] for a in np.flatnonzero(weights)]
        self.weights = weights[weights > 0.0]
        self.weights /= self.weights.sum()

        super().__init__(seed=seed)

        self._np_random_mixture = None
        self.seed_mixture(seed=mixture_seed)

        self.current_agent = None

    def clone(self) -> AgentType:
        """Clone an independent copy of the agent."""

        candidates = [candidate.clone() for candidate in self.candidates]
        seed = self.np_random.randint(np.iinfo(int).max)
        mixture_seed = self.np_random.randint(np.iinfo(int).max)

        clone = type(self)(
            candidates=candidates, weights=self.weights, mixture_seed=mixture_seed, seed=seed
        )
        return clone

    def spawn(self, num_agents: int) -> List[AgentType]:
        """Spawn new agents."""

        agents = [self.clone() for _ in range(num_agents)]
        mixture_seed = self.np_random.randint(np.iinfo(int).max)
        for agent in agents:
            agent.seed_mixture(seed=mixture_seed)

        return agents

    @property
    def np_random_mixture(self) -> np.random.RandomState:  # pylint: disable=no-member
        """The random number generator of the policy mixture."""

        if self._np_random_mixture is None:
            self.seed_mixture()
        return self._np_random_mixture

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set seed for the agent's random number generator.
        This function will be called before the first call of reset().
        """

        seeds = super().seed(seed)

        int_max = np.iinfo(int).max
        for candidate in self.candidates:
            seeds.append(candidate.seed(self.np_random.randint(int_max))[0])

        return seeds

    def seed_mixture(self, seed: Optional[int] = None) -> List[int]:
        """Set seed for the random number generator of the policy mixture."""

        self._np_random_mixture, seed = seeding.np_random(seed=seed)
        return [seed]

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().
        """

        super().reset(observation)

        self.current_agent = self.np_random_mixture.choice(self.candidates, p=self.weights)
        self.current_agent.reset(observation)

    def observe(self, observation, info=None):
        """The agent observe the environment before sending messages.
        This function will be called before send_requests().
        """

        self.state, self.last_observation, self.last_info, _ = self.check_inputs(observation, info)

        self.current_agent.observe(observation, info)

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        action = self.current_agent.act(observation, info, deterministic=deterministic)
        return action

    def send_requests(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called after observe() but before receive_requests().
        """

        return self.current_agent.send_requests()

    def receive_requests(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_requests().
        """

        self.last_requests = messages = tuple(messages)

        self.current_agent.receive_requests(messages)

    def send_responses(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called after receive_requests().
        """

        return self.current_agent.send_responses()

    def receive_responses(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_responses() but before act().
        """

        self.last_responses = messages = tuple(messages)

        self.current_agent.receive_responses(messages)

    def __str__(self):
        return super().__str__() + str(tuple(zip(self.weights, self.candidates)))

    def __repr__(self):
        return super().__repr__() + repr(tuple(zip(self.weights, self.candidates)))


class MixtureCameraAgent(MixtureAgentMixIn, CameraAgentBase):
    """Mixture Camera Agent

    Randomly choose a underlying camera agent in candidates at episode start.
    """


class MixtureTargetAgent(MixtureAgentMixIn, TargetAgentBase):
    """Mixture Target Agent

    Randomly choose a underlying target agent in candidates at episode start.
    """
