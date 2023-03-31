"""Constants of the Multi-Agent Tracking Environment."""

import functools
from typing import Dict

import numpy as np
from gym import spaces

from mate.utils import Team


__all__ = [
    'TERRAIN_SIZE',
    'TERRAIN_WIDTH',
    'TERRAIN_SPACE',
    'WAREHOUSES',
    'NUM_WAREHOUSES',
    'WAREHOUSE_RADIUS',
    'MAX_CAMERA_VIEWING_ANGLE',
    'TARGET_RADIUS',
    'PRESERVED_SPACE',
    'PRESERVED_DIM',
    'OBSERVATION_OFFSET',
    'CAMERA_STATE_DIM_PUBLIC',
    'CAMERA_STATE_SPACE_PUBLIC',
    'CAMERA_STATE_DIM_PRIVATE',
    'CAMERA_STATE_SPACE_PRIVATE',
    'TARGET_STATE_DIM_PUBLIC',
    'TARGET_STATE_SPACE_PUBLIC',
    'TARGET_STATE_DIM_PRIVATE',
    'TARGET_STATE_SPACE_PRIVATE',
    'OBSTACLE_STATE_DIM',
    'OBSTACLE_STATE_SPACE',
    'CAMERA_ACTION_DIM',
    'CAMERA_DEFAULT_ACTION',
    'TARGET_ACTION_DIM',
    'TARGET_DEFAULT_ACTION',
    'camera_observation_space_of',
    'target_observation_space_of',
    'observation_space_of',
    'camera_observation_indices_of',
    'target_observation_indices_of',
    'observation_indices_of',
    'camera_observation_slices_of',
    'target_observation_slices_of',
    'observation_slices_of',
    'camera_coordinate_mask_of',
    'target_coordinate_mask_of',
    'coordinate_mask_of',
]

TERRAIN_SIZE = 1000.0
"""Terrain size. The terrain is a 2000.0 by 2000.0 square."""

TERRAIN_WIDTH = 2.0 * TERRAIN_SIZE
"""Terrain width. The terrain is a 2000.0 by 2000.0 square."""

TERRAIN_SPACE = spaces.Box(
    low=np.array([-TERRAIN_SIZE, -TERRAIN_SIZE]),
    high=np.array([+TERRAIN_SIZE, +TERRAIN_SIZE]),
    dtype=np.float64,
)
"""The space object of the terrain. The terrain is a 2000.0 by 2000.0 square.
(i.e., a square within range of :math:`[-1000, +1000] \\times [-1000, +1000]` in cartesian coordinates.)
"""

WAREHOUSE_RADIUS = 0.075 * TERRAIN_SIZE
"""Half width of the squared warehouses."""

WAREHOUSES = (TERRAIN_SIZE - WAREHOUSE_RADIUS) * np.array(
    [[+1.0, +1.0], [-1.0, +1.0], [-1.0, -1.0], [+1.0, -1.0]]
)
"""Center locations of the warehouses."""

NUM_WAREHOUSES = len(WAREHOUSES)
"""Number of warehouses."""

MAX_CAMERA_VIEWING_ANGLE = 180.0
"""Maximum viewing angle of cameras **in degrees**."""

TARGET_RADIUS = 0.0
"""Radius of targets."""

PRESERVED_SPACE = spaces.Box(
    low=np.concatenate(
        [[0] * 4, 2.0 * np.tile(TERRAIN_SPACE.low, reps=NUM_WAREHOUSES), [0.0]]
    ).astype(np.float64),
    high=np.concatenate(
        [[+np.inf] * 4, 2.0 * np.tile(TERRAIN_SPACE.high, reps=NUM_WAREHOUSES), [TERRAIN_SIZE]]
    ).astype(np.float64),
    dtype=np.float64,
)
"""The space object of agent's preserved data."""

PRESERVED_DIM = 3 + 1 + 2 * NUM_WAREHOUSES + 1
"""Preserved observation dimension,
which holds the number of entities in the environment and the index of current agent.
"""

OBSERVATION_OFFSET = PRESERVED_DIM
"""Preserved observation dimension,
which holds the number of entities in the environment and the index of current agent.
"""

CAMERA_STATE_DIM_PUBLIC = 6
"""Dimension of camera's public state."""

CAMERA_STATE_SPACE_PUBLIC = spaces.Box(
    low=np.append(2.0 * TERRAIN_SPACE.low, [0.0, -TERRAIN_WIDTH, -TERRAIN_WIDTH, 0.0]).astype(
        np.float64
    ),
    high=np.append(
        2.0 * TERRAIN_SPACE.high,
        [TERRAIN_SIZE, TERRAIN_WIDTH, TERRAIN_WIDTH, MAX_CAMERA_VIEWING_ANGLE],
    ).astype(np.float64),
    dtype=np.float64,
)
"""The space object of camera's public state."""

CAMERA_STATE_DIM_PRIVATE = 9
"""Dimension of camera's private state."""

CAMERA_STATE_SPACE_PRIVATE = spaces.Box(
    low=np.append(CAMERA_STATE_SPACE_PUBLIC.low, [0.0, 0.0, 0.0]).astype(np.float64),
    high=np.append(
        CAMERA_STATE_SPACE_PUBLIC.high,
        [TERRAIN_WIDTH, MAX_CAMERA_VIEWING_ANGLE, MAX_CAMERA_VIEWING_ANGLE],
    ).astype(np.float64),
    dtype=np.float64,
)
"""The space object of camera's private state."""

TARGET_STATE_DIM_PUBLIC = 4
"""Dimension of target's public state."""

# Use space Box(low=-1, high=1) as `is_loaded` bit space (the actual range is [0, 1]).
# The boolean bit will remain unchanged after observation normalization.
TARGET_STATE_SPACE_PUBLIC = spaces.Box(
    low=np.append(2.0 * TERRAIN_SPACE.low, [0.0, -1.0]).astype(np.float64),
    high=np.append(2.0 * TERRAIN_SPACE.high, [TERRAIN_WIDTH, 1.0]).astype(np.float64),
    dtype=np.float64,
)
"""The space object of target's public state."""

TARGET_STATE_DIM_PRIVATE = 6 + NUM_WAREHOUSES * 2
"""Dimension of target's private state."""

# Use space Box(low=-1, high=1) as `empty_bits` space (the actual range is [0, 1]).
# The boolean bits will remain unchanged after observation normalization.
TARGET_STATE_SPACE_PRIVATE = spaces.Box(
    low=np.concatenate(
        [
            TARGET_STATE_SPACE_PUBLIC.low,
            [0.0, 1.0],
            [0.0] * NUM_WAREHOUSES,
            [-1.0] * NUM_WAREHOUSES,
        ]
    ).astype(np.float64),
    high=np.concatenate(
        [
            TARGET_STATE_SPACE_PUBLIC.high,
            [TERRAIN_WIDTH, 2.0],
            [+np.inf] * NUM_WAREHOUSES,
            [1.0] * NUM_WAREHOUSES,
        ]
    ).astype(np.float64),
    dtype=np.float64,
)
"""The space object of target's private state."""

OBSTACLE_STATE_DIM = 3
"""Dimension of obstacle's state."""

OBSTACLE_STATE_SPACE = spaces.Box(
    low=np.append(2.0 * TERRAIN_SPACE.low, 0.0).astype(np.float64),
    high=np.append(2.0 * TERRAIN_SPACE.high, TERRAIN_SIZE).astype(np.float64),
    dtype=np.float64,
)
"""The space object of obstacle's state."""

CAMERA_ACTION_DIM = 2
"""Dimension of camera's action."""

CAMERA_DEFAULT_ACTION = np.asarray([0.0, 0.0], dtype=np.float64)
"""Default action of cameras."""

TARGET_ACTION_DIM = 2
"""Dimension of target's action."""

TARGET_DEFAULT_ACTION = np.asarray([0.0, 0.0], dtype=np.float64)
"""Default action of targets."""


@functools.lru_cache(maxsize=None)
def camera_observation_space_of(
    num_cameras: int, num_targets: int, num_obstacles: int
) -> spaces.Box:
    """Get the space object of a single camera's observation from the given number of entities."""

    # Use space Box(low=-1, high=1) as flag space (the actual range is [0, 1]).
    # The boolean flags will remain unchanged after observation normalization.
    return spaces.Box(
        low=np.concatenate(
            [
                PRESERVED_SPACE.low,
                CAMERA_STATE_SPACE_PRIVATE.low,
                np.tile(np.append(TARGET_STATE_SPACE_PUBLIC.low, -1), reps=num_targets),
                np.tile(np.append(OBSTACLE_STATE_SPACE.low, -1), reps=num_obstacles),
                np.tile(np.append(CAMERA_STATE_SPACE_PUBLIC.low, -1), reps=num_cameras),
            ]
        ).astype(np.float64),
        high=np.concatenate(
            [
                PRESERVED_SPACE.high,
                CAMERA_STATE_SPACE_PRIVATE.high,
                np.tile(np.append(TARGET_STATE_SPACE_PUBLIC.high, 1), reps=num_targets),
                np.tile(np.append(OBSTACLE_STATE_SPACE.high, 1), reps=num_obstacles),
                np.tile(np.append(CAMERA_STATE_SPACE_PUBLIC.high, 1), reps=num_cameras),
            ]
        ).astype(np.float64),
        dtype=np.float64,
    )


@functools.lru_cache(maxsize=None)
def target_observation_space_of(
    num_cameras: int, num_targets: int, num_obstacles: int
) -> spaces.Box:
    """Get the space object of a single target's observation from the given number of entities."""

    # Use space Box(low=-1, high=1) as flag space (the actual range is [0, 1]).
    # The boolean flags will remain unchanged after observation normalization.
    return spaces.Box(
        low=np.concatenate(
            [
                PRESERVED_SPACE.low,
                TARGET_STATE_SPACE_PRIVATE.low,
                np.tile(np.append(CAMERA_STATE_SPACE_PUBLIC.low, -1), reps=num_cameras),
                np.tile(np.append(OBSTACLE_STATE_SPACE.low, -1), reps=num_obstacles),
                np.tile(np.append(TARGET_STATE_SPACE_PUBLIC.low, -1), reps=num_targets),
            ]
        ).astype(np.float64),
        high=np.concatenate(
            [
                PRESERVED_SPACE.high,
                TARGET_STATE_SPACE_PRIVATE.high,
                np.tile(np.append(CAMERA_STATE_SPACE_PUBLIC.high, 1), reps=num_cameras),
                np.tile(np.append(OBSTACLE_STATE_SPACE.high, 1), reps=num_obstacles),
                np.tile(np.append(TARGET_STATE_SPACE_PUBLIC.high, 1), reps=num_targets),
            ]
        ).astype(np.float64),
        dtype=np.float64,
    )


@functools.lru_cache(maxsize=None)
def observation_space_of(
    team: Team, num_cameras: int, num_targets: int, num_obstacles: int
) -> spaces.Box:
    """Get the space object of a single agent's observation of the given team from the given number of entities."""

    return (camera_observation_space_of, target_observation_space_of)[team.value](
        num_cameras, num_targets, num_obstacles
    )


@functools.lru_cache(maxsize=None)
def camera_observation_indices_of(
    num_cameras: int, num_targets: int, num_obstacles: int
) -> np.ndarray:
    """The start indices of each part of the camera observation."""

    return np.cumsum(
        [
            0,
            PRESERVED_DIM,
            CAMERA_STATE_DIM_PRIVATE,
            num_targets * (TARGET_STATE_DIM_PUBLIC + 1),
            num_obstacles * (OBSTACLE_STATE_DIM + 1),
            num_cameras * (CAMERA_STATE_DIM_PUBLIC + 1),
        ]
    )


@functools.lru_cache(maxsize=None)
def target_observation_indices_of(
    num_cameras: int, num_targets: int, num_obstacles: int
) -> np.ndarray:
    """The start indices of each part of the target observation."""

    return np.cumsum(
        [
            0,
            PRESERVED_DIM,
            TARGET_STATE_DIM_PRIVATE,
            num_cameras * (CAMERA_STATE_DIM_PUBLIC + 1),
            num_obstacles * (OBSTACLE_STATE_DIM + 1),
            num_targets * (TARGET_STATE_DIM_PUBLIC + 1),
        ]
    )


@functools.lru_cache(maxsize=None)
def observation_indices_of(
    team: Team, num_cameras: int, num_targets: int, num_obstacles: int
) -> np.ndarray:
    """The start indices of each part of the observation."""

    return (camera_observation_indices_of, target_observation_indices_of)[team.value](
        num_cameras, num_targets, num_obstacles
    )


@functools.lru_cache(maxsize=None)
def camera_observation_slices_of(
    num_cameras: int, num_targets: int, num_obstacles: int
) -> Dict[str, slice]:
    """The slices of each part of the camera observation."""

    indices = camera_observation_indices_of(num_cameras, num_targets, num_obstacles)
    return {
        'preserved_data': slice(indices[0], indices[1]),
        'self_state': slice(indices[1], indices[2]),
        'opponent_states_with_mask': slice(indices[2], indices[3]),
        'opponent_mask': slice(
            indices[2] + TARGET_STATE_DIM_PUBLIC, indices[3], TARGET_STATE_DIM_PUBLIC + 1
        ),
        'obstacle_states_with_mask': slice(indices[3], indices[4]),
        'obstacle_mask': slice(indices[3] + OBSTACLE_STATE_DIM, indices[4], OBSTACLE_STATE_DIM + 1),
        'teammate_states_with_mask': slice(indices[4], indices[5]),
        'teammate_mask': slice(
            indices[4] + CAMERA_STATE_DIM_PUBLIC, indices[5], CAMERA_STATE_DIM_PUBLIC + 1
        ),
    }


@functools.lru_cache(maxsize=None)
def target_observation_slices_of(
    num_cameras: int, num_targets: int, num_obstacles: int
) -> Dict[str, slice]:
    """The slices of each part of the target observation."""

    indices = target_observation_indices_of(num_cameras, num_targets, num_obstacles)
    return {
        'preserved_data': slice(indices[0], indices[1]),
        'self_state': slice(indices[1], indices[2]),
        'opponent_states_with_mask': slice(indices[2], indices[3]),
        'opponent_mask': slice(
            indices[2] + CAMERA_STATE_DIM_PUBLIC, indices[3], CAMERA_STATE_DIM_PUBLIC + 1
        ),
        'obstacle_states_with_mask': slice(indices[3], indices[4]),
        'obstacle_mask': slice(indices[3] + OBSTACLE_STATE_DIM, indices[4], OBSTACLE_STATE_DIM + 1),
        'teammate_states_with_mask': slice(indices[4], indices[5]),
        'teammate_mask': slice(
            indices[4] + TARGET_STATE_DIM_PUBLIC, indices[5], TARGET_STATE_DIM_PUBLIC + 1
        ),
    }


@functools.lru_cache(maxsize=None)
def observation_slices_of(
    team: Team, num_cameras: int, num_targets: int, num_obstacles: int
) -> np.ndarray:
    """The slices of each part of the observation."""

    return (camera_observation_slices_of, target_observation_slices_of)[team.value](
        num_cameras, num_targets, num_obstacles
    )


@functools.lru_cache(maxsize=None)
def camera_coordinate_mask_of(num_cameras: int, num_targets: int, num_obstacles: int) -> np.ndarray:
    """Get the bit mask from the given number of entities.
    The bit values is true if the corresponding entry in a single target's observation
    is a coordinate value (exclude the current target itself).
    """

    preserved_mask = np.zeros(PRESERVED_DIM, dtype=np.bool8)
    preserved_mask[-1 - 2 * NUM_WAREHOUSES : -1] = True  # the warehouse locations

    camera_mask = np.zeros(CAMERA_STATE_DIM_PRIVATE, dtype=np.bool8)

    target_mask = np.zeros(TARGET_STATE_DIM_PUBLIC + 1, dtype=np.bool8)
    target_mask[:2] = True  # first two elements (x, y)
    target_mask = np.tile(target_mask, reps=num_targets)

    obstacle_mask = np.zeros(OBSTACLE_STATE_DIM + 1, dtype=np.bool8)
    obstacle_mask[:2] = True  # first two elements (x, y)
    obstacle_mask = np.tile(obstacle_mask, reps=num_obstacles)

    other_camera_mask = np.zeros(CAMERA_STATE_DIM_PUBLIC + 1, dtype=np.bool8)
    other_camera_mask[:2] = True  # first two elements (x, y)
    other_camera_mask = np.tile(other_camera_mask, reps=num_cameras)

    return np.concatenate(
        [preserved_mask, camera_mask, target_mask, obstacle_mask, other_camera_mask]
    ).astype(np.bool8)


@functools.lru_cache(maxsize=None)
def target_coordinate_mask_of(num_cameras: int, num_targets: int, num_obstacles: int) -> np.ndarray:
    """Get the bit mask from the given number of entities.
    The bit values is true if the corresponding entry in a single target's observation
    is a coordinate value (exclude the current target itself).
    """

    preserved_mask = np.zeros(PRESERVED_DIM, dtype=np.bool8)
    preserved_mask[-1 - 2 * NUM_WAREHOUSES : -1] = True  # the warehouse locations

    target_mask = np.zeros(TARGET_STATE_DIM_PRIVATE, dtype=np.bool8)

    camera_mask = np.zeros(CAMERA_STATE_DIM_PUBLIC + 1, dtype=np.bool8)
    camera_mask[:2] = True  # first two elements (x, y)
    camera_mask = np.tile(camera_mask, reps=num_cameras)

    obstacle_mask = np.zeros(OBSTACLE_STATE_DIM + 1, dtype=np.bool8)
    obstacle_mask[:2] = True  # first two elements (x, y)
    obstacle_mask = np.tile(obstacle_mask, reps=num_obstacles)

    other_target_mask = np.zeros(TARGET_STATE_DIM_PUBLIC + 1, dtype=np.bool8)
    other_target_mask[:2] = True  # first two elements (x, y)
    other_target_mask = np.tile(other_target_mask, reps=num_targets)

    return np.concatenate(
        [preserved_mask, target_mask, camera_mask, obstacle_mask, other_target_mask]
    ).astype(np.bool8)


@functools.lru_cache(maxsize=None)
def coordinate_mask_of(
    team: Team, num_cameras: int, num_targets: int, num_obstacles: int
) -> np.ndarray:
    """Get the bit mask from the given number of entities.
    The bit values is true if the corresponding entry in a single agent's observation
    is a boolean value (include the current agent itself).
    """

    return (camera_coordinate_mask_of, target_coordinate_mask_of)[team.value](
        num_cameras, num_targets, num_obstacles
    )
