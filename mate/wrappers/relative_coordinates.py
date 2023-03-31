# pylint: disable=missing-module-docstring

from typing import Tuple, Union

import gym
import numpy as np

from mate.agents.utils import convert_coordinates
from mate.utils import Team
from mate.wrappers.typing import MateEnvironmentType, WrapperMeta, assert_mate_environment


class RelativeCoordinates(gym.ObservationWrapper, metaclass=WrapperMeta):
    """Convert all locations of other entities in the observation to relative
    coordinates (exclude the current agent itself). (Not used in the evaluation script.)
    """

    def __init__(self, env: MateEnvironmentType) -> None:
        assert_mate_environment(env)
        assert not isinstance(
            env, RelativeCoordinates
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'

        super().__init__(env)

        # pylint: disable-next=import-outside-toplevel
        from mate.wrappers.single_team import SingleTeamHelper

        self.single_team = isinstance(env, SingleTeamHelper)

    def observation(
        self, observation: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.single_team:
            return self.convert_coordinates(observation, team=self.team)

        camera_joint_observation, target_joint_observation = observation
        camera_joint_observation = self.convert_coordinates(
            camera_joint_observation, team=Team.CAMERA
        )
        target_joint_observation = self.convert_coordinates(
            target_joint_observation, team=Team.TARGET
        )
        return camera_joint_observation, target_joint_observation

    # pylint: disable-next=missing-function-docstring
    def convert_coordinates(self, observation: np.ndarray, team: Team) -> np.ndarray:
        return convert_coordinates(
            observation,
            team=team,
            num_cameras=self.num_cameras,
            num_targets=self.num_targets,
            num_obstacles=self.num_obstacles,
        )
