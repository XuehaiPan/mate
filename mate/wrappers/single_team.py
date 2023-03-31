# pylint: disable=missing-module-docstring

import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np

from mate.agents.base import CameraAgentBase, TargetAgentBase
from mate.utils import Message, Team

# pylint: disable-next=cyclic-import
from mate.wrappers.typing import (
    AgentType,
    BaseEnvironmentType,
    WrapperMeta,
    assert_base_environment,
)


def group_reset(
    agents: Iterable[AgentType], joint_observation: Union[np.ndarray, Iterable[np.ndarray]]
) -> None:
    """Reset a group of agents."""

    for agent, observation in zip(agents, joint_observation):
        agent.reset(observation)


def group_observe(
    agents: Iterable[AgentType],
    joint_observation: Union[np.ndarray, Iterable[np.ndarray]],
    infos: Optional[List[dict]] = None,
) -> List[Union[int, np.ndarray]]:
    """Set the observation for a group of agents."""

    if infos is None:
        infos = itertools.repeat(None)

    for agent, observation, info in zip(agents, joint_observation, infos):
        agent.observe(observation, info)


def group_communicate(env: BaseEnvironmentType, agents: Iterable[AgentType]) -> None:
    """Send and receive messages from a group of agents to the environment."""

    agents = list(agents)

    for agent in agents:
        env.send_messages(agent.send_requests())

    for agent in agents:
        agent.receive_requests(env.receive_messages(agent=agent))

    for agent in agents:
        env.send_messages(agent.send_responses())

    for agent in agents:
        agent.receive_responses(env.receive_messages(agent=agent))


def group_act(
    agents: Iterable[AgentType],
    joint_observation: Union[np.ndarray, Iterable[np.ndarray]],
    infos: Optional[List[dict]] = None,
    deterministic: Optional[bool] = None,
) -> List[Union[int, np.ndarray]]:
    """Get the joint action of a group of agents."""

    if infos is None:
        infos = itertools.repeat(None)

    return [
        agent.act(observation, info, deterministic=deterministic)
        for agent, observation, info in zip(agents, joint_observation, infos)
    ]


def group_step(
    env: BaseEnvironmentType,
    agents: Iterable[AgentType],
    joint_observation: Union[np.ndarray, Iterable[np.ndarray]],
    infos: Optional[List[dict]] = None,
    deterministic: Optional[bool] = None,
) -> List[Union[int, np.ndarray]]:
    """Helper function to do a environment step for a group of agents."""

    group_observe(agents, joint_observation, infos)
    group_communicate(env, agents)
    joint_action = group_act(agents, joint_observation, infos, deterministic=deterministic)

    return joint_action


# pylint: disable-next=missing-class-docstring,too-many-instance-attributes
class SingleTeamHelper(gym.Wrapper, metaclass=WrapperMeta):
    def __init__(self, env: BaseEnvironmentType, team: Team) -> None:
        assert_base_environment(env)

        super().__init__(env)

        self.team = team

        # pylint: disable=unbalanced-tuple-unpacking
        self.num_teammates, self.num_opponents = self.swap(env.num_cameras, env.num_targets)
        self.teammate_action_space, self.opponent_action_space = self.swap(
            env.camera_action_space, env.target_action_space
        )
        self.teammate_joint_action_space, self.opponent_joint_action_space = self.swap(
            env.camera_joint_action_space, env.target_joint_action_space
        )
        self.teammate_observation_space, self.opponent_observation_space = self.swap(
            env.camera_observation_space, env.target_observation_space
        )
        self.teammate_joint_observation_space, self.opponent_joint_observation_space = self.swap(
            env.camera_joint_observation_space, env.target_joint_observation_space
        )
        self.teammate_message_buffer, self.opponent_message_buffer = self.swap(
            env.camera_message_buffer, env.target_message_buffer
        )
        self.teammate_message_queue, self.opponent_message_queue = self.swap(
            env.camera_message_queue, env.target_message_queue
        )
        # pylint: enable=unbalanced-tuple-unpacking

        assert self.num_teammates > 0, (
            f'There must be at least one agent in the {team.name.lower()} team. '
            f'Got num_teammates = {self.num_teammates}.'
        )
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from mate.wrappers.repeated_reward_individual_done import RepeatedRewardIndividualDone

        self.repeated_reward_individual_done = isinstance(env, RepeatedRewardIndividualDone)

    @property
    def num_adversaries(self):  # pylint: disable=missing-function-docstring
        return self.num_opponents

    def reset(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.swap(*self.env.reset(**kwargs))

    def step(
        self, action: Tuple[np.ndarray, np.ndarray]
    ) -> Union[
        # original form
        Tuple[
            Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, Tuple[List[dict], List[dict]]
        ],
        # repeated reward and individual done
        Tuple[
            Tuple[np.ndarray, np.ndarray],
            Tuple[List[float], List[float]],
            Tuple[List[bool], List[bool]],
            Tuple[List[dict], List[dict]],
        ],
    ]:
        return self.swap(*self.env.step(self.swap(*action)))

    # pylint: disable-next=missing-function-docstring
    def swap(self, *items) -> Union[Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
        assert len(items) == 2 or len(items) == 4

        if self.team is Team.CAMERA:
            return items

        if len(items) == 2:
            return items[1], items[0]
        return tuple(
            (item[1], item[0]) if isinstance(item, (tuple, list)) else item for item in items
        )


class SingleTeamMultiAgent(SingleTeamHelper):
    """Wrap the environment into a single-team multi-agent environment that
    users can use the Gym API to train and/or evaluate their agents.
    """

    def __init__(self, env: BaseEnvironmentType, team: Team, opponent_agent: AgentType) -> None:
        super().__init__(env, team=team)

        self.action_space = env.action_space.spaces[team.value]
        self.observation_space = env.observation_space.spaces[team.value]

        self.opponent_agent = opponent_agent
        self.opponent_agents_ordered = opponent_agent.spawn(self.num_opponents)
        self.opponent_agents = list(self.opponent_agents_ordered)
        self.opponent_joint_observation = None
        self.opponent_infos = None

    def load_config(self, config: Optional[Union[Dict[str, Any], str]] = None) -> None:
        """Reinitialize the Multi-Agent Tracking Environment from a dictionary mapping or a JSON/YAML file."""

        self.env.load_config(config=config)

        SingleTeamMultiAgent.__init__(
            self, self.env, team=self.team, opponent_agent=self.opponent_agent
        )

    def reset(self, **kwargs) -> np.ndarray:
        joint_observation, self.opponent_joint_observation = super().reset(**kwargs)

        self.opponent_agents = list(self.opponent_agents_ordered)
        if self.shuffle_entities:
            self.np_random.shuffle(self.opponent_agents)

        group_reset(self.opponent_agents, self.opponent_joint_observation)
        self.opponent_infos = None

        return joint_observation

    def send_messages(self, messages: Union[Message, Iterable[Message]]) -> None:
        """Buffer the messages from an agent to others in the same team.

        The environment will send the messages to recipients' through method
        receive_messages(), and also info field of step() results.
        """

        if isinstance(messages, Message):
            messages = (messages,)

        messages = list(messages)
        assert all(m.team is self.team for m in messages), (
            f'All messages must be from the {self.team.name.lower()} team. '
            f'Got messages = {messages}.'
        )

        self.env.send_messages(messages)

    def receive_messages(
        self, agent_id: Optional[Tuple[Team, int]] = None, agent: Optional['AgentType'] = None
    ) -> Union[List[List[Message]], List[Message]]:
        """Retrieve the messages to recipients. If no agent is specified, this
        method will return all the messages to all agents in the environment.

        The environment will also put the messages to recipients' info field of
        step() results.
        """

        if agent_id is None and agent is None:
            return [list(self.teammate_message_buffer[i]) for i in range(self.num_teammates)]

        return self.env.receive_messages(agent_id=agent_id, agent=agent)

    def step(
        self, action: np.ndarray
    ) -> Union[
        Tuple[np.ndarray, float, bool, List[dict]],
        Tuple[np.ndarray, List[float], List[bool], List[dict]],
    ]:
        opponent_joint_action = group_step(
            self.env, self.opponent_agents, self.opponent_joint_observation, self.opponent_infos
        )

        (
            (joint_observation, self.opponent_joint_observation),
            (reward, _),
            done,
            (infos, self.opponent_infos),
        ) = super().step((np.asarray(action), np.asarray(opponent_joint_action)))

        if self.repeated_reward_individual_done:
            done = done[0]

        return joint_observation, reward, done, infos

    def seed(self, seed: Optional[int] = None) -> List[int]:
        seeds = self.env.seed(seed)

        int_max = np.iinfo(int).max
        for agent in itertools.chain([self.opponent_agent], self.opponent_agents_ordered):
            seeds.append(agent.seed(self.np_random.randint(int_max))[0])

        return seeds

    def __str__(self) -> str:
        # pylint: disable-next=consider-using-f-string
        return '<{0}(opponent={1.__module__}.{1.__name__}){2}>'.format(
            self.__class__.__name__, self.opponent_agent.__class__, self.env
        )


class MultiCamera(SingleTeamMultiAgent):
    """Wrap the environment into a single-team multi-agent environment that
    users can use the Gym API to train and/or evaluate their camera agents.
    """

    def __init__(self, env: BaseEnvironmentType, target_agent: TargetAgentBase) -> None:
        assert isinstance(target_agent, TargetAgentBase), (
            f'You should provide an instance of target agent. '
            f'Got target_agent = {target_agent!r}.'
        )

        super().__init__(env, team=Team.CAMERA, opponent_agent=target_agent)


class MultiTarget(SingleTeamMultiAgent):
    """Wraps the environment into a single-team multi-agent environment that
    users can use the Gym API to train and/or evaluate their target agents.
    """

    def __init__(self, env: BaseEnvironmentType, camera_agent: CameraAgentBase) -> None:
        assert isinstance(camera_agent, CameraAgentBase), (
            f'You should provide an instance of camera agent. '
            f'Got camera_agent = {camera_agent!r}.'
        )

        super().__init__(env, team=Team.TARGET, opponent_agent=camera_agent)


class SingleTeamSingleAgent(SingleTeamHelper):  # pylint: disable=too-many-instance-attributes
    """Wrap the environment to a single-team single-agent environment that
    users can use the Gym API to train and/or evaluate their agent.
    """

    def __init__(
        self,
        env: BaseEnvironmentType,
        team: Team,
        teammate_agent: AgentType,
        opponent_agent: AgentType,
    ) -> None:
        super().__init__(env, team=team)

        self.action_space = self.teammate_action_space
        self.observation_space = self.teammate_observation_space

        self.index = None
        self.teammate_agent = teammate_agent
        self.teammate_agents_ordered = teammate_agent.spawn(self.num_teammates - 1)
        self.teammate_agents = list(self.teammate_agents_ordered)
        self.joint_observation = None
        self.infos = None

        self.opponent_agent = opponent_agent
        self.opponent_agents_ordered = opponent_agent.spawn(self.num_opponents)
        self.opponent_agents = list(self.opponent_agents_ordered)
        self.opponent_joint_observation = None
        self.opponent_infos = None

    def load_config(self, config: Optional[Union[Dict[str, Any], str]] = None) -> None:
        """Reinitialize the Multi-Agent Tracking Environment from a dictionary mapping or a JSON/YAML file."""

        self.env.load_config(config=config)

        SingleTeamSingleAgent.__init__(
            self,
            self.env,
            team=self.team,
            teammate_agent=self.teammate_agent,
            opponent_agent=self.opponent_agent,
        )

    def reset(self, **kwargs) -> np.ndarray:
        self.joint_observation, self.opponent_joint_observation = super().reset(**kwargs)

        self.opponent_agents = list(self.opponent_agents_ordered)
        if self.shuffle_entities:
            self.np_random.shuffle(self.opponent_agents)

        group_reset(self.opponent_agents, self.opponent_joint_observation)
        self.opponent_infos = None

        self.index = self.num_teammates - 1
        self.teammate_agents = list(self.teammate_agents_ordered)
        if self.shuffle_entities:
            self.index = self.np_random.randint(self.num_teammates)
            self.np_random.shuffle(self.teammate_agents)

        group_reset(
            self.teammate_agents,
            itertools.chain(
                self.joint_observation[: self.index], self.joint_observation[self.index + 1 :]
            ),
        )
        self.infos = None

        if isinstance(self.joint_observation, np.ndarray):
            observation = self.joint_observation[self.index]
        else:
            observation = tuple(item[self.index] for item in self.joint_observation)

        return observation

    def send_messages(self, messages: Union[Message, Iterable[Message]]) -> None:
        """Buffer the messages from an agent to others in the same team.

        The environment will send the messages to recipients' through method
        receive_messages(), and also info field of step() results.
        """

        if isinstance(messages, Message):
            messages = (messages,)

        messages = list(messages)
        assert all(m.team is self.team and m.sender == self.index for m in messages), (
            f'All messages must be from the {self.index}-th agent of the {self.team.name.lower()} team.'
            f'Got messages = {messages}.'
        )

        self.env.send_messages(messages)

    def receive_messages(
        self, agent_id: Optional[Tuple[Team, int]] = None, agent: Optional['AgentType'] = None
    ) -> Union[List[List[Message]], List[Message]]:
        """Retrieve the messages to recipients. If no agent is specified, this
        method will return all the messages to all agents in the environment.

        The environment will also put the messages to recipients' info field of
        step() results.
        """

        if agent_id is None and agent is None:
            agent_id = (self.team, self.index)

        return self.env.receive_messages(agent_id=agent_id, agent=agent)

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        teammate_joint_observation = list(
            itertools.chain(
                self.joint_observation[: self.index], self.joint_observation[self.index + 1 :]
            )
        )

        if self.infos is not None:
            teammate_infos = self.infos[: self.index] + self.infos[self.index + 1 :]
        else:
            teammate_infos = None

        joint_action = group_step(
            self.env, self.teammate_agents, teammate_joint_observation, teammate_infos
        )
        joint_action.insert(self.index, action)

        opponent_joint_action = group_step(
            self.env, self.opponent_agents, self.opponent_joint_observation, self.opponent_infos
        )

        (
            (self.joint_observation, self.opponent_joint_observation),
            (reward, _),
            done,
            (self.infos, self.opponent_infos),
        ) = super().step((np.asarray(joint_action), np.asarray(opponent_joint_action)))

        if self.repeated_reward_individual_done:
            reward = reward[self.index]
            done = done[0][self.index]

        return self.joint_observation[self.index], reward, done, self.infos[self.index]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        seeds = self.env.seed(seed)

        int_max = np.iinfo(int).max
        for agent in itertools.chain(
            [self.teammate_agent, self.opponent_agent],
            self.teammate_agents_ordered,
            self.opponent_agents_ordered,
        ):
            seeds.append(agent.seed(self.np_random.randint(int_max))[0])

        return seeds

    def __str__(self) -> str:
        return '<{0}(teammate={1.__module__}.{1.__name__}, opponent={2.__module__}.{2.__name__}){3}>'.format(  # pylint: disable=consider-using-f-string
            self.__class__.__name__,
            self.teammate_agent.__class__,
            self.opponent_agent.__class__,
            self.env,
        )


class SingleCamera(SingleTeamSingleAgent):
    """Wrap the environment to a single-team single-agent environment that
    users can use the Gym API to train and/or evaluate their camera agent.
    """

    def __init__(
        self,
        env: BaseEnvironmentType,
        other_camera_agent: CameraAgentBase,
        target_agent: TargetAgentBase,
    ) -> None:
        assert isinstance(other_camera_agent, CameraAgentBase), (
            f'You should provide an instance of camera agent. '
            f'Got other_camera_agent = {other_camera_agent!r}.'
        )
        assert isinstance(target_agent, TargetAgentBase), (
            f'You should provide an instance of target agent. '
            f'Got target_agent = {target_agent!r}.'
        )

        super().__init__(
            env, team=Team.CAMERA, teammate_agent=other_camera_agent, opponent_agent=target_agent
        )


class SingleTarget(SingleTeamSingleAgent):
    """Wrap the environment to a single-team single-agent environment that
    users can use the Gym API to train and/or evaluate their target agent.
    """

    def __init__(
        self,
        env: BaseEnvironmentType,
        other_target_agent: TargetAgentBase,
        camera_agent: CameraAgentBase,
    ) -> None:
        assert isinstance(other_target_agent, TargetAgentBase), (
            f'You should provide an instance of target agent. '
            f'Got other_target_agent = {other_target_agent!r}.'
        )
        assert isinstance(camera_agent, CameraAgentBase), (
            f'You should provide an instance of camera agent. '
            f'Got camera_agent = {camera_agent!r}.'
        )

        super().__init__(
            env, team=Team.TARGET, teammate_agent=other_target_agent, opponent_agent=camera_agent
        )
