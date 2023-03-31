# pylint: disable=missing-module-docstring

from typing import Callable, Dict, List, Tuple, Union

import gym
import numpy as np

from mate import constants as consts
from mate.wrappers.auxiliary_camera_rewards import AuxiliaryCameraRewards
from mate.wrappers.repeated_reward_individual_done import RepeatedRewardIndividualDone
from mate.wrappers.single_team import MultiCamera, SingleTeamHelper
from mate.wrappers.typing import (
    MultiAgentEnvironmentType,
    WrapperMeta,
    assert_multi_agent_environment,
)


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class AuxiliaryTargetRewards(gym.Wrapper, metaclass=WrapperMeta):
    r"""Add additional auxiliary rewards for each individual target. (Not used in the evaluation script.)

    The auxiliary reward is a weighted sum of the following components:

        - ``raw_reward`` (the higher the better): team reward returned by the environment (shared, range in :math:`[0, +\infty)`).
        - ``coverage_rate`` (the lower the better): coverage rate of all targets in the environment (shared, range in :math:`[0, 1]`).
        - ``real_coverage_rate`` (the lower the better): coverage rate of targets with cargoes in the environment (shared, range in :math:`[0, 1]`).
        - ``mean_transport_rate`` (the higher the better): mean transport rate of the target team (shared, range in :math:`[0, 1]`).
        - ``normalized_goal_distance`` (the lower the better): the normalized value of the distance to destination, or the nearest non-empty warehouse when the target is not loaded (individual, range in :math:`[0, \sqrt{2}]`).
        - ``sparse_delivery`` (the higher the better): a boolean value that indicates whether the target reaches the destination (individual, range in :math:`{0, 1}`).
        - ``soft_coverage_score`` (the lower the better): soft coverage score is proportional to the distance from the target to the camera's boundary (individual, range in :math:`[-1, N_{\mathcal{C}}]`).
        - ``is_tracked`` (the lower the better): a boolean value that indicates whether the target is tracked by any camera or not. (individual, range in :math:`{0, 1}`).
        - ``is_colliding`` (the lower the better): a boolean value that indicates whether the target is colliding with obstacles, cameras' barriers of terrain boundary. (individual, range in :math:`{0, 1}`).
        - ``baseline``: constant :math:`1`.
    """  # pylint: disable=line-too-long
    ACCEPTABLE_KEYS = (
        'raw_reward',                # team reward
        'coverage_rate',             # team reward
        'real_coverage_rate',        # team reward
        'mean_transport_rate',       # team reward
        'normalized_goal_distance',  # individual reward
        'sparse_delivery',           # individual reward
        'soft_coverage_score',       # individual reward
        'is_tracked',                # individual reward
        'is_colliding',              # individual reward
        'baseline',                  # constant 1
    )  # fmt: skip
    REDUCERS = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
    }

    def __init__(
        self,
        env: MultiAgentEnvironmentType,
        coefficients: Dict[str, Union[float, Callable[[int, int, int, float, float], float]]],
        reduction: Literal['mean', 'sum', 'max', 'min', 'none'] = 'none',
    ) -> None:
        assert_multi_agent_environment(env)
        assert isinstance(env, RepeatedRewardIndividualDone), (
            f'You should use wrapper `{self.__class__}` with wrapper `RepeatedRewardIndividualDone`. '
            f'Please wrap the environment with wrapper `RepeatedRewardIndividualDone` first. '
            f'Got env = {env}.'
        )
        assert not isinstance(env, MultiCamera), (
            f'You should not use wrapper `{self.__class__}` with wrapper `CameraTarget`. '
            f'Got env = {env}.'
        )
        assert not isinstance(
            env, AuxiliaryTargetRewards
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'
        assert reduction in ('mean', 'sum', 'max', 'none'), (
            f'Invalid reduction method {reduction}. '
            f'The reduction method should be one of {("mean", "sum", "max", "min")} (for shared reward), '
            f'or "none" for no reduction (for individual reward).'
        )
        assert set(self.ACCEPTABLE_KEYS).issuperset(coefficients.keys()), (
            f'The coefficient mapping only accepts keys in {self.ACCEPTABLE_KEYS}. '
            f'Got list(coefficients.keys()) = {list(coefficients.keys())}.'
        )

        # The coefficient should be a function with signature:
        #   (agent_id: int, episode_id: int, episode_step: int, raw_reward: float, auxiliary_reward: float) -> float
        # or a constant float number.
        self.coefficients = {}
        for key, coefficient in coefficients.items():
            assert callable(coefficient) or isinstance(coefficient, (float, int)), (
                f'The argument `coefficient` should be a callable function or a float number. '
                f'Got coefficients[{key!r}] = {coefficient!r}.'
            )
            self.coefficients[key] = (
                coefficient if not isinstance(coefficient, int) else float(coefficient)
            )

        super().__init__(env)
        self.episode_id = -1

        self.reduction = reduction

        self.single_team = isinstance(env, SingleTeamHelper)
        self.soft_coverage_score_matrix = None

    def reset(self, **kwargs) -> np.ndarray:
        self.episode_id += 1
        self.soft_coverage_score_matrix = None

        return self.env.reset(**kwargs)

    # pylint: disable-next=too-many-locals,too-many-branches
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
        observations, rewards, dones, infos = self.env.step(action)

        if self.single_team:
            target_rewards, target_infos = list(rewards), infos
        else:
            target_rewards, target_infos = list(rewards[1]), infos[1]

        normalized_goal_distances = np.zeros((self.num_targets,), dtype=np.float64)
        sparse_delivery = self.target_dones.astype(np.float64)
        for t, target in enumerate(self.targets):
            goal = self.target_goals[t]
            warehouse_distances = np.maximum(
                self.target_warehouse_distances[t] - consts.WAREHOUSE_RADIUS, 0.0, dtype=np.float64
            )
            if goal >= 0:
                goal_distance = warehouse_distances[goal]
            elif not target.empty_bits.all():
                goal_distance = warehouse_distances[np.logical_not(target.empty_bits)].min()
            else:
                goal_distance = consts.TERRAIN_WIDTH / 2.0
            normalized_goal_distances[t] = goal_distance / consts.TERRAIN_WIDTH

        soft_coverage_scores = np.zeros((self.num_targets,), dtype=np.float64)
        if 'soft_coverage_score' in self.coefficients:
            self.soft_coverage_score_matrix = AuxiliaryCameraRewards.compute_soft_coverage_scores(
                self.unwrapped
            )
            camera_target_view_mask = self.camera_target_view_mask
            for t, target in enumerate(self.targets):
                if camera_target_view_mask[:, t].any():
                    scores = self.soft_coverage_score_matrix[camera_target_view_mask[:, t], t]
                    soft_coverage_scores[t] = scores.sum()
                else:
                    scores = self.soft_coverage_score_matrix[:, t]
                    soft_coverage_scores[t] = np.tanh(scores.max())

        for t, (raw_reward, info) in enumerate(zip(tuple(target_rewards), target_infos)):
            auxiliary_rewards = {
                'raw_reward': raw_reward,
                'coverage_rate': self.coverage_rate,
                'real_coverage_rate': self.real_coverage_rate,
                'mean_transport_rate': self.mean_transport_rate,
                'normalized_goal_distance': normalized_goal_distances[t],
                'sparse_delivery': sparse_delivery[t],
                'soft_coverage_score': soft_coverage_scores[t],
                'is_tracked': self.camera_target_view_mask[..., t].any(),
                'is_colliding': self.targets[t].is_colliding,
                'baseline': 1.0,
            }
            reward = 0.0
            for key, coefficient in self.coefficients.items():
                if callable(coefficient):
                    coefficient = coefficient(
                        t, self.episode_id, self.episode_step, raw_reward, auxiliary_rewards[key]
                    )
                reward += coefficient * auxiliary_rewards[key]
                info.setdefault(key, auxiliary_rewards[key])
                info[f'auxiliary_reward_{key}'] = auxiliary_rewards[key]
                info[f'reward_coefficient_{key}'] = coefficient

            info['reward'] = target_rewards[t] = reward

        reducer = self.REDUCERS.get(self.reduction, None)
        if reducer is not None:
            reducer = self.REDUCERS[self.reduction]
            shared_reward = reducer(target_rewards)
            target_rewards = [shared_reward] * self.num_targets
            for info in target_infos:
                info['shared_reward'] = shared_reward

        if not self.single_team:
            rewards = (rewards[0], target_rewards)
        else:
            rewards = target_rewards

        return observations, rewards, dones, infos
