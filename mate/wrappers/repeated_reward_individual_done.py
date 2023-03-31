# pylint: disable=missing-module-docstring

from typing import List, Tuple, Union

import gym
import numpy as np

from mate.utils import Team

# pylint: disable-next=cyclic-import
from mate.wrappers.typing import (
    MultiAgentEnvironmentType,
    WrapperMeta,
    assert_multi_agent_environment,
)


class RepeatedRewardIndividualDone(gym.Wrapper, metaclass=WrapperMeta):
    """Repeat the reward field and assign individual done field of step(),
    which is similar to the OpenAI Multi-Agent Particle Environment.
    (Not used in the evaluation script.)
    """

    def __init__(self, env: MultiAgentEnvironmentType, target_done_at_destination=False) -> None:
        assert_multi_agent_environment(env)
        assert not isinstance(
            env, RepeatedRewardIndividualDone
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'

        super().__init__(env)

        self.target_done_at_destination = target_done_at_destination

        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from mate.wrappers.single_team import SingleTeamHelper

        self.single_team = isinstance(env, SingleTeamHelper)

    def step(
        self, action: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
    ) -> Union[
        Tuple[
            Tuple[np.ndarray, np.ndarray],
            Tuple[List[float], List[float]],
            Tuple[List[bool], List[bool]],
            Tuple[List[dict], List[dict]],
        ],
        Tuple[np.ndarray, List[float], List[bool], List[dict]],
    ]:
        observation, reward, done, info = self.env.step(action)

        if self.target_done_at_destination:
            target_dones = self.target_dones.tolist()
        else:
            target_dones = [done] * self.num_targets

        if self.single_team:
            reward = [reward] * self.num_teammates
            if self.team is Team.TARGET:
                done = target_dones
            else:
                done = [done] * self.num_teammates
        else:
            camera_team_reward, target_team_reward = reward
            reward = (
                [camera_team_reward] * self.num_cameras,
                [target_team_reward] * self.num_targets,
            )
            done = ([done] * self.num_cameras, target_dones)
        return observation, reward, done, info
