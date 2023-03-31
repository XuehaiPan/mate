# pylint: disable=missing-module-docstring

from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces

from mate import constants as consts
from mate.wrappers.typing import BaseEnvironmentType, WrapperMeta, assert_base_environment


def indices_of_nearest_grid_point(continuous: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Convert continuous values to the indices of the nearest grid points."""

    diff = continuous - grid[:, np.newaxis, :]
    indices = np.argmin(np.linalg.norm(diff, axis=-1), axis=0)
    return indices


class DiscreteCamera(gym.ActionWrapper, metaclass=WrapperMeta):
    """Wrap the environment to allow cameras to use discrete actions."""

    def __init__(self, env: BaseEnvironmentType, levels: int = 5) -> None:
        assert_base_environment(env)
        assert not isinstance(
            env, DiscreteCamera
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'
        assert levels >= 3 and levels % 2 == 1, (
            f'The discrete level must be an odd number that not less than 3. '
            f'Got levels = {levels}.'
        )
        assert env.num_cameras > 0, 'There must be at least one camera in the environment.'

        super().__init__(env)

        self.levels = levels
        self.camera_action_space = spaces.Discrete(levels * levels)
        self.camera_joint_action_space = spaces.Tuple(
            spaces=(self.camera_action_space,) * env.num_cameras
        )
        self.action_space = spaces.Tuple(
            spaces=(self.camera_joint_action_space, env.target_joint_action_space)
        )

        self.action_high = np.asarray(
            [env.camera_rotation_step, env.camera_zooming_step], dtype=np.float64
        )

        self.normalized_action_grid = self.discrete_action_grid(levels=self.levels)

    def load_config(self, config: Optional[Union[Dict[str, Any], str]] = None) -> None:
        """Reinitialize the Multi-Agent Tracking Environment from a dictionary mapping or a JSON/YAML file."""

        self.env.load_config(config=config)

        self.__init__(self.env, levels=self.levels)  # pylint: disable=unnecessary-dunder-call

    def action(self, action: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert joint action of cameras from discrete to continuous."""

        camera_joint_action_discrete, target_joint_action = action
        camera_joint_action_discrete = np.asarray(
            camera_joint_action_discrete, dtype=np.int64
        ).ravel()
        assert self.camera_joint_action_space.contains(tuple(camera_joint_action_discrete)), (
            f'Joint action {tuple(camera_joint_action_discrete)} outside given '
            f'joint action space {self.camera_joint_action_space}.'
        )

        camera_joint_action_continuous = (
            self.action_high * self.normalized_action_grid[camera_joint_action_discrete]
        )
        return camera_joint_action_continuous, target_joint_action

    def reverse_action(
        self, action: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert joint action of cameras from continuous to discrete."""

        camera_joint_action_continuous, target_joint_action = action
        camera_joint_action_continuous = np.asarray(
            camera_joint_action_continuous, dtype=np.float64
        )
        camera_joint_action_continuous = camera_joint_action_continuous.reshape(
            self.num_cameras, consts.CAMERA_ACTION_DIM
        )

        camera_joint_action_discrete = indices_of_nearest_grid_point(
            camera_joint_action_continuous / self.action_high, self.normalized_action_grid
        )
        return camera_joint_action_discrete, target_joint_action

    def __str__(self) -> str:
        return f'<{self.__class__.__name__}(levels={self.levels}){self.env}>'

    @staticmethod
    def discrete_action_grid(levels):  # pylint: disable=missing-function-docstring
        assert levels >= 3 and levels % 2 == 1, (
            f'The discrete level must be an odd number that not less than 3. '
            f'Got levels = {levels}.'
        )

        # num_actions = levels * levels
        # ti, tj = i / (levels - 1), j / (levels - 1)
        # xi = -1. * (1. - ti) + 1. * ti
        # yj = -1. * (1. - tj) + 1. * tj
        # action_grid[i + levels * j] = np.array([xi, yj])
        normalized_action_grid = np.stack(
            np.meshgrid(
                np.linspace(start=-1.0, stop=+1.0, num=levels, endpoint=True),
                np.linspace(start=-1.0, stop=+1.0, num=levels, endpoint=True),
            ),
            axis=-1,
        ).reshape(-1, consts.CAMERA_ACTION_DIM)

        return normalized_action_grid


class DiscreteTarget(gym.ActionWrapper, metaclass=WrapperMeta):
    """Wrap the environment to allow targets to use discrete actions."""

    def __init__(self, env: BaseEnvironmentType, levels: int = 5) -> None:
        assert_base_environment(env)
        assert not isinstance(
            env, DiscreteTarget
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'
        assert levels >= 3 and levels % 2 == 1, (
            f'The discrete level must be an odd number that not less than 3. '
            f'Got levels = {levels}.'
        )

        super().__init__(env)

        self.levels = levels
        self.target_action_space = spaces.Discrete(levels * levels)
        self.target_joint_action_space = spaces.Tuple(
            spaces=(self.target_action_space,) * env.num_targets
        )
        self.action_space = spaces.Tuple(
            spaces=(env.camera_joint_action_space, self.target_joint_action_space)
        )

        self.action_high = env.target_step_size * np.ones(
            (env.num_targets, consts.TARGET_ACTION_DIM), dtype=np.float64
        )

        self.normalized_action_grid = self.discrete_action_grid(levels=self.levels)

    def load_config(self, config: Optional[Union[Dict[str, Any], str]] = None) -> None:
        """Reinitialize the Multi-Agent Tracking Environment from a dictionary mapping or a JSON/YAML file."""

        self.env.load_config(config=config)

        self.__init__(self.env, levels=self.levels)  # pylint: disable=unnecessary-dunder-call

    def reset(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        joint_observations = self.env.reset(**kwargs)

        for t, target in enumerate(self.targets):
            self.action_high[t] = target.step_size

        return joint_observations

    def action(self, action: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert joint action of targets from discrete to continuous."""

        camera_joint_action, target_joint_action_discrete = action
        target_joint_action_discrete = np.asarray(
            target_joint_action_discrete, dtype=np.int64
        ).ravel()
        assert self.target_joint_action_space.contains(tuple(target_joint_action_discrete)), (
            f'Joint action {tuple(target_joint_action_discrete)} outside given '
            f'joint action space {self.target_joint_action_space}.'
        )

        target_joint_action_continuous = (
            self.action_high * self.normalized_action_grid[target_joint_action_discrete]
        )
        return camera_joint_action, target_joint_action_continuous

    def reverse_action(
        self, action: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert joint action of targets from continuous to discrete."""

        camera_joint_action, target_joint_action_continuous = action
        target_joint_action_continuous = np.asarray(
            target_joint_action_continuous, dtype=np.float64
        )
        target_joint_action_continuous = target_joint_action_continuous.shape(
            self.num_targets, consts.TARGET_ACTION_DIM
        )

        target_joint_action_discrete = indices_of_nearest_grid_point(
            target_joint_action_continuous / self.action_high, self.normalized_action_grid
        )
        return camera_joint_action, target_joint_action_discrete

    def __str__(self) -> str:
        return f'<{self.__class__.__name__}(levels={self.levels}){self.env}>'

    @staticmethod
    def discrete_action_grid(levels):  # pylint: disable=missing-function-docstring
        assert levels >= 3 and levels % 2 == 1, (
            f'The discrete level must be an odd number that not less than 3. '
            f'Got levels = {levels}.'
        )

        # num_actions = levels * levels
        # ti, tj = i / (levels - 1), j / (levels - 1)
        # xi = -1. * (1. - ti) + 1. * ti
        # yj = -1. * (1. - tj) + 1. * tj
        # norm = np.linalg.norm([xi, yj])
        # bound = np.sqrt(1. + np.square(np.max(np.abs([xi, yj])) / np.min(np.abs([xi, yj]))))
        # action_grid[i + levels * j] = (np.array([xi, yj]) / norm) * (norm / bound) = np.array([xi, yj]) / bound
        action_grid = np.stack(
            np.meshgrid(
                np.linspace(start=-1.0, stop=+1.0, num=levels, endpoint=True),
                np.linspace(start=-1.0, stop=+1.0, num=levels, endpoint=True),
            ),
            axis=-1,
        ).reshape(-1, consts.TARGET_ACTION_DIM)
        angle = np.arctan2(action_grid[..., -1], action_grid[..., 0])
        bound = 1.0 / np.cos(np.pi * ((angle / np.pi + 0.25) % 0.5 - 0.25))
        normalized_action_grid = action_grid / bound[..., np.newaxis]

        return normalized_action_grid
