# pylint: disable=missing-module-docstring

from typing import Tuple, Union

import gym
import numpy as np
from gym import spaces

from mate.agents.utils import rescale_observation
from mate.utils import Team
from mate.wrappers.typing import MateEnvironmentType, WrapperMeta, assert_mate_environment


# pylint: disable-next=too-many-instance-attributes
class RescaledObservation(gym.ObservationWrapper, metaclass=WrapperMeta):
    """Rescale all entity states in the observation to [-1., +1.]. (Not used in the evaluation script.)"""

    def __init__(self, env: MateEnvironmentType) -> None:
        assert_mate_environment(env)
        assert not isinstance(
            env, RescaledObservation
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'

        super().__init__(env)

        # pylint: disable-next=import-outside-toplevel
        from mate.wrappers.single_team import SingleTeamHelper, SingleTeamSingleAgent

        self.single_team = isinstance(env, SingleTeamHelper)

        camera_observation_space = spaces.Box(
            low=self.rescale_observation(env.camera_observation_space.low, team=Team.CAMERA),
            high=self.rescale_observation(env.camera_observation_space.high, team=Team.CAMERA),
            dtype=np.float64,
        )
        target_observation_space = spaces.Box(
            low=self.rescale_observation(env.target_observation_space.low, team=Team.TARGET),
            high=self.rescale_observation(env.target_observation_space.high, team=Team.TARGET),
            dtype=np.float64,
        )
        camera_joint_observation_space = spaces.Tuple(
            spaces=(camera_observation_space,) * env.num_cameras
        )
        target_joint_observation_space = spaces.Tuple(
            spaces=(target_observation_space,) * env.num_targets
        )

        if self.single_team:
            self.teammate_observation_space, self.opponent_observation_space = env.swap(
                camera_observation_space, target_observation_space
            )
            self.teammate_joint_observation_space, self.opponent_joint_observation_space = env.swap(
                camera_joint_observation_space, target_joint_observation_space
            )
            if env.team is Team.CAMERA:
                self.camera_observation_space = camera_observation_space
                self.camera_joint_observation_space = camera_joint_observation_space
            else:
                self.target_observation_space = target_observation_space
                self.target_joint_observation_space = target_joint_observation_space
            if isinstance(env, SingleTeamSingleAgent):
                self.observation_space = self.teammate_observation_space
            else:
                self.observation_space = spaces.Tuple(
                    spaces=(self.teammate_observation_space,) * env.num_teammates
                )
        else:
            self.camera_observation_space = camera_observation_space
            self.target_observation_space = target_observation_space
            self.camera_joint_observation_space = camera_joint_observation_space
            self.target_joint_observation_space = target_joint_observation_space
            self.observation_space = spaces.Tuple(
                spaces=(self.camera_joint_observation_space, self.target_joint_observation_space)
            )

    def observation(
        self, observation: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.single_team:
            return self.rescale_observation(observation, team=self.team)

        camera_joint_observation, target_joint_observation = observation
        camera_joint_observation = self.rescale_observation(
            camera_joint_observation, team=Team.CAMERA
        )
        target_joint_observation = self.rescale_observation(
            target_joint_observation, team=Team.TARGET
        )
        return camera_joint_observation, target_joint_observation

    # pylint: disable-next=missing-function-docstring
    def rescale_observation(self, observation: np.ndarray, team: Team) -> np.ndarray:
        return rescale_observation(
            observation,
            team=team,
            num_cameras=self.num_cameras,
            num_targets=self.num_targets,
            num_obstacles=self.num_obstacles,
        )
