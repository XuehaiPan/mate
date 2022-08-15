"""Utility functions and classes for agents."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from typing import Optional, Tuple, Union

import numpy as np
from gym import spaces

from mate.constants import (
    CAMERA_STATE_DIM_PRIVATE,
    CAMERA_STATE_DIM_PUBLIC,
    NUM_WAREHOUSES,
    OBSTACLE_STATE_DIM,
    PRESERVED_DIM,
    TARGET_STATE_DIM_PRIVATE,
    TARGET_STATE_DIM_PUBLIC,
    coordinate_mask_of,
    observation_indices_of,
    observation_slices_of,
    observation_space_of,
)
from mate.utils import Team, Vector2D


__all__ = [
    'convert_coordinates',
    'normalize_observation',
    'rescale_observation',
    'split_observation',
    'CameraStatePublic',
    'CameraStatePrivate',
    'TargetStatePublic',
    'TargetStatePrivate',
    'ObstacleState',
]


# pylint: disable-next=too-many-locals
def convert_coordinates(
    observation: np.ndarray,
    team: Team,
    num_cameras: int,
    num_targets: int,
    num_obstacles: int,
) -> np.ndarray:
    """Convert all locations of other entities in the observation to relative
    coordinates (exclude the current agent itself).
    """

    observation_space = observation_space_of(team, num_cameras, num_targets, num_obstacles)
    assert observation.shape[-1] >= observation_space.shape[-1], (
        f'The feature size of the observation must be not less than {observation_space.shape[-1]}. '
        f'Got observation.shape[-1] = {observation.shape[-1]}.'
    )

    converted = observation[..., : observation_space.shape[-1]].copy()

    slices = observation_slices_of(team, num_cameras, num_targets, num_obstacles)
    if team is Team.CAMERA:
        teammate_state_dim, opponent_state_dim = CAMERA_STATE_DIM_PUBLIC, TARGET_STATE_DIM_PUBLIC
    else:
        teammate_state_dim, opponent_state_dim = TARGET_STATE_DIM_PUBLIC, CAMERA_STATE_DIM_PUBLIC
    opponent_view_mask = converted[..., slices['opponent_mask']].astype(np.bool8)
    obstacle_view_mask = converted[..., slices['obstacle_mask']].astype(np.bool8)
    teammate_view_mask = converted[..., slices['teammate_mask']].astype(np.bool8)
    view_mask = np.hstack(
        [
            np.repeat(opponent_view_mask, repeats=opponent_state_dim + 1, axis=-1),
            np.repeat(obstacle_view_mask, repeats=OBSTACLE_STATE_DIM + 1, axis=-1),
            np.repeat(teammate_view_mask, repeats=teammate_state_dim + 1, axis=-1),
        ]
    )

    coordinate_mask = np.broadcast_to(
        coordinate_mask_of(team, num_cameras, num_targets, num_obstacles), shape=converted.shape
    ).copy()
    other_entities_size = view_mask.shape[-1]
    coordinate_mask[..., -other_entities_size:] = np.logical_and(
        coordinate_mask[..., -other_entities_size:], view_mask
    )

    origin = converted[..., PRESERVED_DIM : PRESERVED_DIM + 2]
    if converted.ndim == 1:
        converted[coordinate_mask] -= np.tile(origin, reps=coordinate_mask.sum() // 2)
    else:
        for i in range(converted.shape[0]):
            converted[i, coordinate_mask[i]] -= np.tile(
                origin[i], reps=coordinate_mask[i].sum() // 2
            )

    if observation.shape[-1] == observation_space.shape[-1]:
        return converted
    return np.hstack([converted, observation[..., observation_space.shape[-1] :]])


def normalize_observation(
    observation: np.ndarray,
    observation_space: spaces.Box,
    additional_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rescale all entity states in the observation to [-1., +1.]."""

    assert observation.shape[-1] >= observation_space.shape[-1], (
        f'The feature size of the observation must be not less than {observation_space.shape[-1]}. '
        f'Got observation.shape[-1] = {observation.shape[-1]}.'
    )

    rescaled = observation[..., : observation_space.shape[-1]].copy()

    bounded_below = observation_space.bounded_below
    bounded_above = observation_space.bounded_above
    bounded_both = np.logical_and(bounded_below, bounded_above)
    mask = np.logical_and(bounded_both, observation_space.high > observation_space.low)
    if additional_mask is not None:
        mask = np.logical_and(mask, additional_mask)

    rescaled[..., bounded_below] = (
        rescaled[..., bounded_below] - observation_space.low[bounded_below]
    )
    rescaled[..., mask] = (
        2.0 * rescaled[..., mask] / ((observation_space.high - observation_space.low)[mask]) - 1.0
    )

    if observation.shape[-1] == observation_space.shape[-1]:
        return rescaled
    return np.hstack([rescaled, observation[..., observation_space.shape[-1] :]])


def rescale_observation(
    observation: np.ndarray, team: Team, num_cameras: int, num_targets: int, num_obstacles: int
) -> np.ndarray:
    """Rescale all entity states in the observation to [-1., +1.]."""

    observation_space = observation_space_of(team, num_cameras, num_targets, num_obstacles)

    return normalize_observation(observation, observation_space)


def split_observation(
    observation: np.ndarray, team: Team, num_cameras: int, num_targets: int, num_obstacles: int
) -> Tuple[(np.ndarray,) * 5]:
    """Splits the concatenated observations into parts."""

    indices = observation_indices_of(team, num_cameras, num_targets, num_obstacles)
    assert observation.shape[-1] == indices[-1], (
        f'The feature size of the observation must be equal to {indices[-1]}. '
        f'Got observation.shape[-1] = {observation.shape[-1]}.'
    )
    return tuple(np.hsplit(observation, indices[1:-1]))


class StateBase:
    DIM = None

    def __init__(self, state: np.ndarray, index: int) -> None:
        assert len(state) == self.DIM

        self._state = state
        self._index = index
        self._location = None

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def index(self) -> int:
        return self._index

    @property
    def location(self) -> np.ndarray:
        if self._location is None:
            self._location = self.state[..., 0:2]
        return self._location

    def copy(self):
        return type(self)(self.state.copy(), self.index)

    def __array__(self):
        return self.state.copy()

    def __sub__(self, other):
        assert isinstance(other, StateBase)

        return Vector2D(vector=self.location - other.location, origin=other.location)


class CameraStatePublic(StateBase):
    DIM = CAMERA_STATE_DIM_PUBLIC

    def __init__(self, state: np.ndarray, index: int) -> None:
        super().__init__(state, index)

        self._radius = None
        self._sight_range = None
        self._orientation = None
        self._viewing_angle = None

    @property
    def radius(self) -> Union[float, np.ndarray]:
        if self._radius is None:
            self._radius = np.linalg.norm(self.state[..., 2])
        return self._radius

    @property
    def sight_range(self) -> Union[float, np.ndarray]:
        if self._sight_range is None:
            self._sight_range = np.linalg.norm(self.state[..., 3:5])
        return self._sight_range

    @property
    def orientation(self) -> Union[float, np.ndarray]:
        if self._orientation is None:
            self._orientation = np.rad2deg(np.arctan2(self.state[..., 4], self.state[..., 3]))
        return self._orientation

    @property
    def viewing_angle(self) -> Union[float, np.ndarray]:
        if self._viewing_angle is None:
            self._viewing_angle = self.state[..., 5]
        return self._viewing_angle


class CameraStatePrivate(CameraStatePublic):
    DIM = CAMERA_STATE_DIM_PRIVATE

    def __init__(self, state: np.ndarray, index: int) -> None:
        super().__init__(state, index)

        self._max_sight_range = None
        self._rotation_step = None
        self._zooming_step = None
        self._min_viewing_angle = None

    @property
    def max_sight_range(self) -> Union[float, np.ndarray]:
        if self._max_sight_range is None:
            self._max_sight_range = self.state[..., 6]
        return self._max_sight_range

    @property
    def min_viewing_angle(self) -> Union[float, np.ndarray]:
        if self._min_viewing_angle is None:
            self._min_viewing_angle = self.viewing_angle * np.square(
                self.sight_range / self.max_sight_range
            )
        return self._min_viewing_angle

    @property
    def rotation_step(self) -> Union[float, np.ndarray]:
        if self._rotation_step is None:
            self._rotation_step = self.state[..., 7]
        return self._rotation_step

    @property
    def zooming_step(self) -> Union[float, np.ndarray]:
        if self._zooming_step is None:
            self._zooming_step = self.state[..., 8]
        return self._zooming_step

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            # pylint: disable-next=invalid-unary-operand-type
            low=np.asarray([-self.rotation_step, -self.zooming_step]),
            high=np.asarray([self.rotation_step, self.zooming_step]),
            dtype=np.float64,
        )


class TargetStatePublic(StateBase):
    DIM = TARGET_STATE_DIM_PUBLIC

    def __init__(self, state: np.ndarray, index: int) -> None:
        super().__init__(state, index)

        self._sight_range = None
        self._is_loaded = None

    @property
    def sight_range(self) -> Union[float, np.ndarray]:
        if self._sight_range is None:
            self._sight_range = self.state[..., 2]
        return self._sight_range

    @property
    def is_loaded(self) -> Union[bool, np.ndarray]:
        if self._is_loaded is None:
            self._is_loaded = self.state[..., 3].astype(np.bool8)
            if self._is_loaded.ndim == 0:
                self._is_loaded = bool(self._is_loaded)
        return self._is_loaded


class TargetStatePrivate(StateBase):
    DIM = TARGET_STATE_DIM_PRIVATE

    def __init__(self, state: np.ndarray, index: int) -> None:
        super().__init__(state, index)

        self._step_size = None
        self._capacity = None
        self._goal_bits = None
        self._empty_bits = None

    @property
    def step_size(self) -> Union[float, np.ndarray]:
        if self._step_size is None:
            self._step_size = self.state[..., 4]
        return self._step_size

    @property
    def capacity(self) -> Union[float, np.ndarray]:
        if self._capacity is None:
            self._capacity = self.state[..., 5]
        return self._capacity

    @property
    def goal_bits(self) -> np.ndarray:
        if self._goal_bits is None:
            self._goal_bits = self.state[..., 6 : 6 + NUM_WAREHOUSES].astype(np.int64)
        return self._goal_bits

    @property
    def empty_bits(self) -> np.ndarray:
        if self._empty_bits is None:
            self._empty_bits = self.state[..., 6 + NUM_WAREHOUSES : 6 + 2 * NUM_WAREHOUSES].astype(
                np.bool8
            )
        return self._empty_bits

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            # pylint: disable-next=invalid-unary-operand-type
            low=np.asarray([-self.step_size, -self.step_size]),
            high=np.asarray([self.step_size, self.step_size]),
            dtype=np.float64,
        )


class ObstacleState(StateBase):
    DIM = OBSTACLE_STATE_DIM

    def __init__(self, state: np.ndarray, index: int) -> None:
        super().__init__(state, index)

        self._radius = None

    @property
    def radius(self) -> Union[float, np.ndarray]:
        if self._radius is None:
            self._radius = self.state[..., 2]
        return self._radius
