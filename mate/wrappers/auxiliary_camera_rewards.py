# pylint: disable=missing-module-docstring

from typing import Callable, Dict, List, Tuple, Union

import gym
import numpy as np

from mate.utils import polar2cartesian, sin_deg
from mate.wrappers.repeated_reward_individual_done import RepeatedRewardIndividualDone
from mate.wrappers.single_team import MultiTarget, SingleTeamHelper
from mate.wrappers.typing import (
    MultiAgentEnvironmentType,
    WrapperMeta,
    assert_multi_agent_environment,
)


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class AuxiliaryCameraRewards(gym.Wrapper, metaclass=WrapperMeta):
    r"""Add additional auxiliary rewards for each individual camera. (Not used in the evaluation script.)

    The auxiliary reward is a weighted sum of the following components:

        - ``raw_reward`` (the higher the better): team reward returned by the environment (shared, range in :math:`(-\infty, 0]`).
        - ``coverage_rate`` (the higher the better): coverage rate of all targets in the environment (shared, range in :math:`[0, 1]`).
        - ``real_coverage_rate`` (the higher the better): coverage rate of targets with cargoes in the environment (shared, range in :math:`[0, 1]`).
        - ``mean_transport_rate`` (the lower the better): mean transport rate of the target team (shared, range in :math:`[0, 1]`).
        - ``soft_coverage_score`` (the higher the better): soft coverage score is proportional to the distance from the target to the camera's boundary (individual, range in :math:`[-1, N_{\mathcal{T}}]`).
        - ``num_tracked`` (the higher the better): number of targets tracked the camera (shared, range in :math:`[0, N_{\mathcal{T}}]`).
        - ``baseline``: constant :math:`1`.
    """  # pylint: disable=line-too-long
    ACCEPTABLE_KEYS = (
        'raw_reward',           # team reward
        'coverage_rate',        # team reward
        'real_coverage_rate',   # team reward
        'mean_transport_rate',  # team reward
        'soft_coverage_score',  # individual reward
        'num_tracked',          # individual reward
        'baseline',             # constant 1
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
        assert not isinstance(env, MultiTarget), (
            f'You should not use wrapper `{self.__class__}` with wrapper `MultiTarget`. '
            f'Got env = {env}.'
        )
        assert not isinstance(
            env, AuxiliaryCameraRewards
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'
        assert reduction in ('mean', 'sum', 'max', 'min', 'none'), (
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
            camera_rewards, camera_infos = list(rewards), infos
        else:
            camera_rewards, camera_infos = list(rewards[0]), infos[0]

        soft_coverage_scores = np.zeros((self.num_cameras,), dtype=np.float64)
        if 'soft_coverage_score' in self.coefficients:
            self.soft_coverage_score_matrix = self.compute_soft_coverage_scores(self.unwrapped)
            camera_target_view_mask = self.camera_target_view_mask
            for c in range(self.num_cameras):
                if camera_target_view_mask[c].any():
                    scores = self.soft_coverage_score_matrix[c, camera_target_view_mask[c]]
                    soft_coverage_scores[c] = scores.sum()
                else:
                    scores = self.soft_coverage_score_matrix[c, :]
                    soft_coverage_scores[c] = np.tanh(scores.max())

        for c, (raw_reward, info) in enumerate(zip(tuple(camera_rewards), camera_infos)):
            auxiliary_rewards = {
                'raw_reward': raw_reward,
                'coverage_rate': self.coverage_rate,
                'real_coverage_rate': self.real_coverage_rate,
                'mean_transport_rate': self.mean_transport_rate,
                'soft_coverage_score': soft_coverage_scores[c],
                'num_tracked': self.camera_target_view_mask[c, ...].sum(),
                'baseline': 1.0,
            }
            reward = 0.0
            for key, coefficient in self.coefficients.items():
                if callable(coefficient):
                    coefficient = coefficient(
                        c,
                        self.episode_id,
                        self.episode_step,
                        raw_reward,
                        auxiliary_rewards[key],
                    )
                reward += coefficient * auxiliary_rewards[key]
                info.setdefault(key, auxiliary_rewards[key])
                info[f'auxiliary_reward_{key}'] = auxiliary_rewards[key]
                info[f'reward_coefficient_{key}'] = coefficient

            info['reward'] = camera_rewards[c] = reward

        reducer = self.REDUCERS.get(self.reduction, None)
        if reducer is not None:
            shared_reward = reducer(camera_rewards)
            camera_rewards = [shared_reward] * self.num_cameras
            for info in camera_infos:
                info['shared_reward'] = shared_reward

        if not self.single_team:
            rewards = (camera_rewards, rewards[1])
        else:
            rewards = camera_rewards

        return observations, rewards, dones, infos

    @staticmethod
    def compute_soft_coverage_scores(env) -> np.ndarray:
        """Compute all soft coverage score for each individual camera."""

        auxiliary_reward_matrix = np.zeros((env.num_cameras, env.num_targets), dtype=np.float64)
        for c, camera in enumerate(env.cameras):
            tracked_bits = env.camera_target_view_mask[c]
            auxiliary_reward_matrix[c] = AuxiliaryCameraRewards.compute_soft_coverage_score(
                camera, env.targets, tracked_bits
            )

        return auxiliary_reward_matrix

    @staticmethod
    # pylint: disable-next=too-many-locals
    def compute_soft_coverage_score(camera, targets, tracked_bits: np.ndarray) -> List[float]:
        """The soft coverage score is proportional to the distance from the target to the camera's boundary."""

        if camera.viewing_angle < 180.0:
            dist_max = camera.sight_range / (1.0 + 1.0 / sin_deg(camera.viewing_angle / 2.0))
        else:
            dist_max = camera.sight_range / 2.0

        angle_left = camera.orientation - camera.viewing_angle / 2.0
        angle_right = camera.orientation + camera.viewing_angle / 2.0
        phis, rhos = camera.boundary_between(angle_left, angle_right, outer=True)

        phi_left, phi_right = phis[0], phis[-1]
        rho_left, rho_right = rhos[0], rhos[-1]

        phis = np.concatenate([[phi_left] * 16, phis, [phi_right] * 16])
        rhos = np.concatenate(
            [
                np.linspace(start=0.0, stop=rho_left, num=16, endpoint=False),
                rhos,
                np.linspace(start=0.0, stop=rho_right, num=16, endpoint=False),
            ]
        )

        xs, ys = polar2cartesian(rhos, phis)  # pylint: disable=invalid-name

        auxiliary_rewards = []
        for tracked, target in zip(tracked_bits, targets):
            direction = target - camera
            distances = np.hypot(direction.x - xs, direction.y - ys)
            dist = distances.min()
            if not tracked:
                dist = -dist

            auxiliary_reward = dist / dist_max
            auxiliary_rewards.append(auxiliary_reward)

        return auxiliary_rewards
