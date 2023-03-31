# pylint: disable=missing-module-docstring

import itertools
from typing import List, Tuple, Union

import gym
import numpy as np

from mate import constants as consts
from mate.wrappers.typing import BaseEnvironmentType, WrapperMeta, assert_base_environment


class MoreTrainingInformation(gym.Wrapper, metaclass=WrapperMeta):
    """Add more environment and agent information to the info field of step(),
    enabling full observability of the environment. (Not used in the evaluation script.)
    """

    def __init__(self, env: BaseEnvironmentType) -> None:
        assert_base_environment(env)
        assert not isinstance(
            env, MoreTrainingInformation
        ), f'You should not use wrapper `{self.__class__}` more than once.'

        super().__init__(env)

    # pylint: disable-next=too-many-locals
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
        (
            (camera_joint_observation, target_joint_observation),
            _,
            _,
            (camera_infos, target_infos),
        ) = results = self.env.step(action)

        offset = consts.PRESERVED_DIM
        camera_states_private = camera_joint_observation[
            ..., offset : offset + consts.CAMERA_STATE_DIM_PRIVATE
        ]
        target_states_private = target_joint_observation[
            ..., offset : offset + consts.TARGET_STATE_DIM_PRIVATE
        ]

        remaining_cargo_counts = self.remaining_cargoes.sum(axis=-1)

        # Information for cameras
        for c, camera_info in enumerate(camera_infos):
            camera_info.update(
                num_tracked=self.camera_target_view_mask[c, ...].sum(),
                is_sensed=self.target_camera_view_mask[..., c].any(),
            )

        # Information for targets
        for t, target_info in enumerate(target_infos):
            goal = self.target_goals[t]
            warehouse_distances = np.maximum(
                self.target_warehouse_distances[t] - consts.WAREHOUSE_RADIUS, 0.0, dtype=np.float64
            )
            goal_distance = warehouse_distances[goal] if goal >= 0 else consts.TERRAIN_WIDTH / 2.0
            target_info.update(
                goal=goal,
                goal_distance=goal_distance,
                warehouse_distances=warehouse_distances,
                individual_done=self.target_dones[t],
                is_tracked=self.camera_target_view_mask[..., t].any(),
                is_colliding=self.targets[t].is_colliding,
            )

        # Enable full observability
        state = self.state()
        for info in itertools.chain(camera_infos, target_infos):
            info.update(
                state=state.copy(),
                camera_states=camera_states_private.copy(),
                target_states=target_states_private.copy(),
                obstacle_states=self.obstacle_states.copy(),
                camera_target_view_mask=self.camera_target_view_mask.copy(),
                camera_obstacle_view_mask=self.camera_obstacle_view_mask.copy(),
                target_camera_view_mask=self.target_camera_view_mask.copy(),
                target_obstacle_view_mask=self.target_obstacle_view_mask.copy(),
                target_target_view_mask=self.target_target_view_mask.copy(),
                remaining_cargoes=self.remaining_cargoes.copy(),
                remaining_cargo_counts=remaining_cargo_counts.copy(),
                awaiting_cargo_counts=self.awaiting_cargo_counts.copy(),
            )

        return results
