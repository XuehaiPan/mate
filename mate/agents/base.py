"""Base classes for agents."""

import copy
import functools
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from gym import spaces
from gym.utils import seeding

from mate import constants as consts
from mate.agents import utils
from mate.utils import Message, Team


__all__ = ['AgentBase', 'CameraAgentBase', 'TargetAgentBase']

StatePublicType = Union[utils.CameraStatePublic, utils.TargetStatePublic]
StatePrivateType = Union[utils.CameraStatePrivate, utils.TargetStatePrivate]

AgentType = Union['AgentBase', 'CameraAgentBase', 'TargetAgentBase']


class AgentBase(ABC):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Base class for all agents."""

    TEAM: Team

    DEFAULT_ACTION: Union[int, np.ndarray] = None
    observation_space: spaces.Space = None
    action_space: spaces.Space = None

    STATE_CLASS: Type[StatePrivateType]
    TEAMMATE_STATE_CLASS: Type[StatePublicType]
    OPPONENT_STATE_CLASS: Type[StatePublicType]

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the agent.
        This function will be called only once on initialization.

        Note:
            Agents can obtain the number of teammates and opponents on agent.reset(observation),
            but not here. You are responsible for writing scalable policies and
            code to handle this.
        """

        # The following attributes will change later when calling `agent.reset(observation)`
        self.num_cameras = None
        self.num_targets = None
        self.num_obstacles = None
        self.index = None
        self.agent_id = f'{self.TEAM.name.lower()}_0'
        self.action_space = None
        self.observation_dim = None
        self.observation_space = None
        self.observation_indices = None
        self.observation_slices = None
        self.convert_coordinates = NotImplemented
        self.rescale_observation = NotImplemented
        self.split_observation = NotImplemented

        # The following attributes will change when calling `agent.observe(observation, info)`
        self.state = None
        self.episode_step = -1
        self._step_counter = 0
        self.last_observation = None
        self.last_info = None
        self.last_requests = ()
        self.last_responses = ()

        self._np_random = None
        self.seed(seed)

    @property
    @abstractmethod
    def num_teammates(self) -> int:
        """Number of agents in the same team, including the current agent itself."""

        raise NotImplementedError

    @property
    @abstractmethod
    def num_opponents(self) -> int:
        """Number of adversarial agents in the opponent team."""

        raise NotImplementedError

    @property
    def num_adversaries(self) -> int:
        """Number of adversarial agents in the opponent team."""

        return self.num_opponents

    def clone(self) -> AgentType:
        """Clone an independent copy of the agent."""

        clone = copy.deepcopy(self)
        clone.seed(self.np_random.randint(np.iinfo(int).max))
        return clone

    def spawn(self, num_agents: int) -> List[AgentType]:
        """Spawn new agents."""

        return [self.clone() for _ in range(num_agents)]

    @property
    def np_random(self) -> np.random.RandomState:  # pylint: disable=no-member
        """The main random number generator of the agent."""

        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set seed for the agent's random number generator.
        This function will be called before the first call of reset().
        """

        self._np_random, seed = seeding.np_random(seed)

        seeds, int_max = [seed], np.iinfo(int).max
        if self.action_space is not None:
            seeds.append(self.action_space.seed(self.np_random.randint(int_max))[0])

        return seeds

    def reset(self, observation: np.ndarray) -> None:
        """Reset the agent.
        This function will be called immediately after env.reset().

        Note:
            observation is a 1D array, not a 2D array with an additional
            dimension for agent indices.
        """

        observation = np.asarray(observation, dtype=np.float64)
        assert observation.ndim == 1, (
            f'The observation should be a 1D NumPy array. '
            f'Got observation = {observation} with shape = {observation.shape}.'
        )

        self.num_cameras = int(np.round(observation[0]).astype(np.int64))
        self.num_targets = int(np.round(observation[1]).astype(np.int64))
        self.num_obstacles = int(np.round(observation[2]).astype(np.int64))
        self.index = int(np.round(observation[3]).astype(np.int64))
        self.agent_id = f'{self.TEAM.name.lower()}_{self.index}'

        kwargs = {
            'team': self.TEAM,
            'num_cameras': self.num_cameras,
            'num_targets': self.num_targets,
            'num_obstacles': self.num_obstacles,
        }
        self.observation_indices = consts.observation_indices_of(**kwargs)
        self.observation_slices = consts.observation_slices_of(**kwargs)
        self.observation_dim = self.observation_indices[-1]
        self.observation_space = consts.observation_space_of(**kwargs)
        self.convert_coordinates = functools.partial(utils.convert_coordinates, **kwargs)
        self.rescale_observation = functools.partial(utils.rescale_observation, **kwargs)
        self.split_observation = functools.partial(utils.split_observation, **kwargs)

        assert observation.shape == (self.observation_dim,), (
            f'The observation should be a 1D NumPy array with length of {self.observation_dim}. '
            f'Got observation = {observation} with shape = {observation.shape}.'
        )
        assert observation.shape == self.observation_space.shape, (
            f'The observation should be a 1D NumPy array with shape of {self.observation_space.shape}. '
            f'Got observation = {observation} with shape = {observation.shape}.'
        )

        self.state = self.STATE_CLASS(
            observation[self.observation_slices['self_state']], index=self.index
        )

        self.action_space = copy.deepcopy(self.state.action_space)
        self.action_space.seed(self.np_random.randint(np.iinfo(int).max))

        self.episode_step = -1
        self._step_counter = 0
        self.last_observation = None
        self.last_info = None
        self.last_requests = ()
        self.last_responses = ()

    def observe(self, observation: np.ndarray, info: Optional[dict] = None) -> None:
        r"""The agent observe the environment before sending messages.
        This function will be called before send_requests().

        .. code-block:: text

                env.step()
            --> agent.observe()
            --> agent.send_requests() --> agent.receive_requests()
            --> agent.send_responses() --> agent.receive_responses()
            --> agent.act()
            --> env.step()
            --> ...

               |                                                             time step                                                           |
            - env ----------------------------------- env --------------------------------------------- env ----------------------------------- env -
                  \                                 /     \                                           /     \                                 /
                    observe           send_requests         receive_requests           send_responses         receive_responses           act
                            \       /                                        \       /                                          \       /
            ----------------- agent ------------------------------------------ agent -------------------------------------------- agent -------------

        Note:
            observation is a 1D array, not a 2D array with an additional
            dimension for agent indices.
        """  # pylint: disable=line-too-long

        # pylint: disable-next=unused-variable
        self.state, self.last_observation, self.last_info, messages = self.check_inputs(
            observation, info
        )

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        info: Optional[dict] = None,
        deterministic: Optional[bool] = None,
    ) -> Union[int, np.ndarray]:
        r"""Get the agent action by the observation.
        This function will be called before every env.step().

        .. code-block:: text

                env.step()
            --> agent.observe()
            --> agent.send_requests() --> agent.receive_requests()
            --> agent.send_responses() --> agent.receive_responses()
            --> agent.act()
            --> env.step()
            --> ...

               |                                                             time step                                                           |
            - env ----------------------------------- env --------------------------------------------- env ----------------------------------- env -
                  \                                 /     \                                           /     \                                 /
                    observe           send_requests         receive_requests           send_responses         receive_responses           act
                            \       /                                        \       /                                          \       /
            ----------------- agent ------------------------------------------ agent -------------------------------------------- agent -------------

        Note:
            observation is a 1D array, not a 2D array with an additional
            dimension for agent indices.
        """  # pylint: disable=line-too-long

        # pylint: disable-next=unused-variable
        self.state, observation, info, messages = self.check_inputs(observation, info)

        # Override this
        raise NotImplementedError

        return self.DEFAULT_ACTION  # pylint: disable=unreachable

    def predict(
        self,
        observation: np.ndarray,
        info: Optional[dict] = None,
        deterministic: Optional[bool] = None,
    ) -> Union[int, np.ndarray]:
        """Get the agent action by the observation. Shortcut method for act().

        Note:
            You should implement method act() instead.
        """

        return self.act(observation, info, deterministic=deterministic)

    def __call__(
        self,
        observation: np.ndarray,
        info: Optional[dict] = None,
        deterministic: Optional[bool] = None,
    ) -> Union[int, np.ndarray]:
        """Shortcut method for act()."""

        return self.act(observation, info, deterministic=deterministic)

    def send_requests(self) -> Iterable[Message]:
        r"""Prepare messages to communicate with other agents in the same team.
        This function will be called after observe() but before receive_requests().

        .. code-block:: text

                env.step()
            --> agent.observe()
            --> agent.send_requests() --> agent.receive_requests()
            --> agent.send_responses() --> agent.receive_responses()
            --> agent.act()
            --> env.step()
            --> ...

               |                                                             time step                                                           |
            - env ----------------------------------- env --------------------------------------------- env ----------------------------------- env -
                  \                                 /     \                                           /     \                                 /
                    observe           send_requests         receive_requests           send_responses         receive_responses           act
                            \       /                                        \       /                                          \       /
            ----------------- agent ------------------------------------------ agent -------------------------------------------- agent -------------
        """  # pylint: disable=line-too-long

        return ()

    def receive_requests(self, messages: Tuple[Message, ...]) -> None:
        r"""Receive messages from other agents in the same team.
        This function will be called after send_requests().

        .. code-block:: text

                env.step()
            --> agent.observe()
            --> agent.send_requests() --> agent.receive_requests()
            --> agent.send_responses() --> agent.receive_responses()
            --> agent.act()
            --> env.step()
            --> ...

               |                                                             time step                                                           |
            - env ----------------------------------- env --------------------------------------------- env ----------------------------------- env -
                  \                                 /     \                                           /     \                                 /
                    observe           send_requests         receive_requests           send_responses         receive_responses           act
                            \       /                                        \       /                                          \       /
            ----------------- agent ------------------------------------------ agent -------------------------------------------- agent -------------
        """  # pylint: disable=line-too-long

        self.last_requests = tuple(messages)

    def send_responses(self) -> Iterable[Message]:
        r"""Prepare messages to communicate with other agents in the same team.
        This function will be called after receive_requests().

        .. code-block:: text

                env.step()
            --> agent.observe()
            --> agent.send_requests() --> agent.receive_requests()
            --> agent.send_responses() --> agent.receive_responses()
            --> agent.act()
            --> env.step()
            --> ...

               |                                                             time step                                                           |
            - env ----------------------------------- env --------------------------------------------- env ----------------------------------- env -
                  \                                 /     \                                           /     \                                 /
                    observe           send_requests         receive_requests           send_responses         receive_responses           act
                            \       /                                        \       /                                          \       /
            ----------------- agent ------------------------------------------ agent -------------------------------------------- agent -------------
        """  # pylint: disable=line-too-long

        return ()

    def receive_responses(self, messages: Tuple[Message, ...]) -> None:
        r"""Receive messages from other agents in the same team.
        This function will be called after send_responses() but before act().

        .. code-block:: text

                env.step()
            --> agent.observe()
            --> agent.send_requests() --> agent.receive_requests()
            --> agent.send_responses() --> agent.receive_responses()
            --> agent.act()
            --> env.step()
            --> ...

               |                                                             time step                                                           |
            - env ----------------------------------- env --------------------------------------------- env ----------------------------------- env -
                  \                                 /     \                                           /     \                                 /
                    observe           send_requests         receive_requests           send_responses         receive_responses           act
                            \       /                                        \       /                                          \       /
            ----------------- agent ------------------------------------------ agent -------------------------------------------- agent -------------
        """  # pylint: disable=line-too-long

        self.last_responses = tuple(messages)

    def check_inputs(
        self, observation: np.ndarray, info: Optional[dict] = None
    ) -> Tuple[StatePrivateType, np.ndarray, dict, List[Message]]:
        """Preprocess the inputs for observe() and act()."""

        observation = np.asarray(observation, dtype=np.float64)
        assert observation.shape == (self.observation_dim,), (
            f'The observation should be a 1D NumPy array with length of {self.observation_dim}. '
            f'Got observation = {observation} with shape = {observation.shape}.'
        )

        info = info or {}
        state = self.STATE_CLASS(
            observation[self.observation_slices['self_state']], index=self.index
        )
        messages = info.get('messages', [])

        if self._step_counter % 2 == 0:
            self.episode_step += 1
        self._step_counter += 1

        return state, observation, info, messages

    def pack_message(self, content: Any, recipient: Optional[int] = None) -> Message:
        """Pack the content into a Message object."""

        return Message(
            sender=self.index,
            recipient=recipient,
            content=content,
            team=self.TEAM,
            broadcasting=(recipient is None),
        )

    def get_teammate_state(
        self, observation: np.ndarray, index: int
    ) -> Tuple[utils.TargetStatePublic, bool]:
        """Get the teammate's public state from observation by index."""

        if not 0 <= index < self.num_teammates:
            raise IndexError('Teammate index out of range.')

        offset = self.observation_indices[4] + (self.TEAMMATE_STATE_CLASS.DIM + 1) * index
        teammate_state = self.TEAMMATE_STATE_CLASS(
            observation[..., offset : offset + self.TEAMMATE_STATE_CLASS.DIM], index=index
        )
        sensed = bool(observation[..., offset + self.TEAMMATE_STATE_CLASS.DIM])
        return teammate_state, sensed

    def get_teammate_states(
        self, observation: np.ndarray
    ) -> Tuple[Tuple[utils.TargetStatePublic, ...], Tuple[bool, ...]]:
        """Get all teammates' states from observation."""

        return tuple(
            zip(
                *[
                    self.get_teammate_state(observation, index)
                    for index in range(self.num_teammates)
                ]
            )
        )

    def get_opponent_state(
        self, observation: np.ndarray, index: int
    ) -> Tuple[StatePublicType, bool]:
        """Get the opponent agent state from observation by index."""

        if not 0 <= index < self.num_opponents:
            raise IndexError('Opponent index out of range.')

        offset = self.observation_indices[2] + (self.OPPONENT_STATE_CLASS.DIM + 1) * index
        opponent_state = self.OPPONENT_STATE_CLASS(
            observation[..., offset : offset + self.OPPONENT_STATE_CLASS.DIM], index=index
        )
        sensed = bool(observation[..., offset + self.OPPONENT_STATE_CLASS.DIM])
        return opponent_state, sensed

    def get_all_opponent_states(
        self, observation: np.ndarray
    ) -> Tuple[Tuple[StatePublicType, ...], Tuple[bool, ...]]:
        """Get all opponents' states from observation."""

        return tuple(
            zip(
                *[
                    self.get_opponent_state(observation, index)
                    for index in range(self.num_opponents)
                ]
            )
        )

    def get_obstacle_state(
        self, observation: np.ndarray, index: int
    ) -> Tuple[utils.ObstacleState, bool]:
        """Get the obstacle state from observation by index."""

        if not 0 <= index < self.num_obstacles:
            raise IndexError('Obstacle index out of range.')

        offset = self.observation_indices[3] + (consts.OBSTACLE_STATE_DIM + 1) * index
        obstacle_state = utils.ObstacleState(
            observation[..., offset : offset + consts.OBSTACLE_STATE_DIM], index=index
        )
        sensed = bool(observation[..., offset + consts.OBSTACLE_STATE_DIM])
        return obstacle_state, sensed

    def get_all_obstacle_states(
        self, observation: np.ndarray
    ) -> Tuple[Tuple[utils.ObstacleState, ...], Tuple[bool, ...]]:
        """Get all obstacle states from observation."""

        return tuple(
            zip(
                *[
                    self.get_obstacle_state(observation, index)
                    for index in range(self.num_obstacles)
                ]
            )
        )


class CameraAgentBase(AgentBase):
    """Base class for camera agents."""

    TEAM = Team.CAMERA

    DEFAULT_ACTION = consts.CAMERA_DEFAULT_ACTION

    STATE_CLASS = utils.CameraStatePrivate
    TEAMMATE_STATE_CLASS = utils.CameraStatePublic
    OPPONENT_STATE_CLASS = utils.TargetStatePublic

    @property
    def num_teammates(self) -> int:
        """Number of agents in the same team, including the current agent."""

        return self.num_cameras

    @property
    def num_opponents(self) -> int:
        """Number of adversarial agents in the opponent team."""

        return self.num_targets


class TargetAgentBase(AgentBase):
    """Base class for target agents."""

    TEAM = Team.TARGET

    DEFAULT_ACTION = consts.TARGET_DEFAULT_ACTION

    STATE_CLASS = utils.TargetStatePrivate
    TEAMMATE_STATE_CLASS = utils.TargetStatePublic
    OPPONENT_STATE_CLASS = utils.CameraStatePublic

    @property
    def num_teammates(self) -> int:
        """Number of agents in the same team, including the current agent."""

        return self.num_targets

    @property
    def num_opponents(self) -> int:
        """Number of adversarial agents in the opponent team."""

        return self.num_cameras
