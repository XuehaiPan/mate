# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from abc import ABC, abstractmethod

import numpy as np
from gym import spaces
from scipy.interpolate import interp1d

from mate import constants as consts
from mate.utils import SpatialHashmap, Vector2D, arcsin_deg, normalize_angle, polar2cartesian


__all__ = ['Camera', 'Target', 'Obstacle']

DEFAULT_SIGHT_RANGE = 500.0
DEFAULT_CAMERA_RADIUS = 40.0
DEFAULT_CAMERA_MIN_VIEWING_ANGLE = 90.0
DEFAULT_CAMERA_ROTATION_STEP = 5.0
DEFAULT_CAMERA_ZOOMING_STEP = 2.5
DEFAULT_TARGET_STEP_SIZE = 10.0
DEFAULT_OBSTACLE_TRANSMITTANCE = 0.0

SPATIAL_GRID_RESOLUTION = 80


class Entity(ABC):
    COLOR = (0.5, 0.5, 0.5)

    STATE_DIM = STATE_DIM_PUBLIC = 2
    STATE_DIM_PRIVATE = 2
    state_space = consts.TERRAIN_SPACE
    action_space = None

    def __init__(self, location=None, location_random_range=None, radius=1.0):
        assert (location is None, location_random_range is None).count(True) == 1, (
            f'You should specify either a fixed location or a random range for the location. '
            f'Got (location, location_random_range) = {(location, location_random_range)}.'
        )
        assert (
            radius >= 0.0
        ), f'The argument `radius` should be a non-negative number. Got radius = {radius}.'

        if location is not None:
            location = np.asarray(location, dtype=np.float64)

        if location_random_range is None:
            location_random_range = spaces.Box(low=location, high=location, dtype=np.float64)

        self.location = location
        self.location_random_range = location_random_range
        self.radius = radius

        self.seed(0)
        self.reset()

    def state(self, private=False):  # pylint: disable=unused-argument
        return self.location.copy()

    def reset(self):
        location = self.location_random_range.sample()
        self.location = location.clip(
            min=consts.TERRAIN_SPACE.low + 1.2 * self.radius,
            max=consts.TERRAIN_SPACE.high - 1.2 * self.radius,
        )

    def simulate(self, action):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.location_random_range.seed(seed)

    @property
    def np_random(self):
        return self.location_random_range.np_random

    @property
    def x(self):
        return self.location[0]

    @property
    def y(self):
        return self.location[1]

    def __sub__(self, other):
        if not isinstance(other, Entity):
            raise NotImplementedError

        return Vector2D(vector=self.location - other.location, origin=other.location)

    def distance(self, other):
        if isinstance(other, Entity):
            return np.linalg.norm(self.location - other.location)
        return np.linalg.norm(self.location - other)

    def overlap(self, other, min_distance=0.0):
        if not isinstance(other, Entity):
            raise NotImplementedError

        return self.distance(other) * (1 + 1e-6) < self.radius + other.radius + min_distance


class Obstacle(Entity):
    COLOR = (0.3, 0.3, 0.3)

    STATE_DIM = STATE_DIM_PUBLIC = consts.OBSTACLE_STATE_DIM  # 3
    STATE_DIM_PRIVATE = consts.OBSTACLE_STATE_DIM  # 3
    state_space_public = state_space_private = state_space = consts.OBSTACLE_STATE_SPACE

    DEFAULTS = {
        'transmittance': DEFAULT_OBSTACLE_TRANSMITTANCE,
    }

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        location=None,
        location_random_range=None,
        radius=None,
        radius_random_range=None,
        transmittance=DEFAULT_OBSTACLE_TRANSMITTANCE,
    ):
        assert (radius is None, radius_random_range is None).count(True) == 1, (
            f'You should specify either a fixed radius or a random range for the radius. '
            f'Got (radius, radius_random_range) = {(radius, radius_random_range)}.'
        )
        assert 0.0 <= transmittance <= 1.0, (
            f'The argument `transmittance` within the range of [0.0, 1.0]. '
            f'Got transmittance = { transmittance}.'
        )

        if radius is None:
            radius = radius_random_range.sample()

        if radius_random_range is None:
            radius_random_range = spaces.Box(
                low=np.asarray(radius), high=np.asarray(radius), dtype=np.float64
            )

        self.radius_random_range = radius_random_range
        self.transmittance = transmittance

        super().__init__(
            location=location, location_random_range=location_random_range, radius=radius
        )

    def state(self, private=False):
        return np.append(self.location, self.radius).astype(np.float64)

    def reset(self):
        self.radius = self.radius_random_range.sample()
        super().reset()

    def seed(self, seed=None):
        self.radius_random_range.seed(seed)
        return super().seed(seed)

    def obstruct(self, ray, keep_tangential=False, outer=False):
        relative = Vector2D(vector=self.location - ray.origin)
        norm = ray.norm
        if norm == 0.0 or relative.norm < self.radius:
            return -ray
        if relative.norm >= norm + self.radius:
            return ray

        inner = np.inner(relative.vector, ray.vector)
        if inner >= 0.0:
            cos = min(1.0, inner / (relative.norm * norm))
            perpendicular = relative.norm * np.sqrt(1.0 - np.square(cos))
            if self.radius > perpendicular:
                half_chord = np.sqrt(np.square(self.radius) - np.square(perpendicular))
                if not outer:
                    new_norm = max(0.0, relative.norm * cos - half_chord)
                else:
                    new_norm = max(0.0, relative.norm * cos + half_chord)
                if new_norm < norm:
                    old_ray = ray.vector
                    ray.norm = new_norm
                    if keep_tangential:
                        radius = ray.endpoint - self.location
                        ray.vector = old_ray + radius * (
                            (norm - new_norm) * half_chord / np.square(self.radius)
                        )
        return ray


class Sensor(Entity):
    STATE_DIM = STATE_DIM_PUBLIC = 3
    STATE_DIM_PRIVATE = 3

    def __init__(
        self, location=None, location_random_range=None, radius=1.0, sight_range=DEFAULT_SIGHT_RANGE
    ):
        assert sight_range > 0.0, (
            f'The argument `sight_range` should be a positive number. '
            f'Got sight_range = {sight_range}.'
        )

        self.sight_range = sight_range

        super().__init__(
            location=location, location_random_range=location_random_range, radius=radius
        )

        self.state_space = self.state_space_public = spaces.Box(
            low=consts.TERRAIN_SPACE.low, high=consts.TERRAIN_SPACE.high, dtype=np.float64
        )
        self.state_space_private = spaces.Box(
            low=np.append(consts.TERRAIN_SPACE.low, 0).astype(np.float64),
            high=np.append(consts.TERRAIN_SPACE.high, consts.TERRAIN_SIZE).astype(np.float64),
            dtype=np.float64,
        )

    def state(self, private=False):
        return np.append(self.location, self.sight_range).astype(np.float64)

    @abstractmethod
    def simulate(self, action):
        raise NotImplementedError

    @abstractmethod
    def add_obstacles(self, *obstacles):
        raise NotImplementedError

    @abstractmethod
    def clear_obstacles(self):
        raise NotImplementedError

    def perceive(self, other):
        assert isinstance(other, Entity)

        return self.distance(other) <= self.sight_range + other.radius


class Camera(Sensor, Obstacle):  # pylint: disable=too-many-instance-attributes
    COLOR_UNPERCEIVED = (0.1, 0.2, 0.6)
    COLOR_PERCEIVED = (0.6, 0.2, 0.1)
    COLOR = COLOR_UNPERCEIVED

    STATE_DIM = STATE_DIM_PUBLIC = consts.CAMERA_STATE_DIM_PUBLIC  # 6
    STATE_DIM_PRIVATE = consts.CAMERA_STATE_DIM_PRIVATE  # 9
    state_space_private = consts.CAMERA_STATE_DIM_PRIVATE
    state_space = state_space_public = consts.CAMERA_STATE_SPACE_PUBLIC

    ACTION_DIM = consts.CAMERA_ACTION_DIM  # 2
    DEFAULT_ACTION = consts.CAMERA_DEFAULT_ACTION  # np.array([0.0, 0.0], dtype=np.float64)

    DEFAULTS = {
        'radius': DEFAULT_CAMERA_RADIUS,
        'min_viewing_angle': DEFAULT_CAMERA_MIN_VIEWING_ANGLE,
        'max_sight_range': DEFAULT_SIGHT_RANGE,
        'rotation_step': DEFAULT_CAMERA_ROTATION_STEP,
        'zooming_step': DEFAULT_CAMERA_ZOOMING_STEP,
    }

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        location=None,
        location_random_range=None,
        radius=DEFAULT_CAMERA_RADIUS,
        min_viewing_angle=DEFAULT_CAMERA_MIN_VIEWING_ANGLE,
        max_sight_range=DEFAULT_SIGHT_RANGE,
        rotation_step=DEFAULT_CAMERA_ROTATION_STEP,
        zooming_step=DEFAULT_CAMERA_ZOOMING_STEP,
    ):
        assert 0.0 < min_viewing_angle <= consts.MAX_CAMERA_VIEWING_ANGLE, (
            f'The argument `min_viewing_angle` within the range of (0.0, {consts.MAX_CAMERA_VIEWING_ANGLE}]. '
            f'Got min_viewing_angle = {min_viewing_angle}.'
        )
        assert rotation_step > 0.0, (
            f'The argument `rotation_step` should be a positive number. '
            f'Got rotation_step = {rotation_step}.'
        )
        assert zooming_step > 0.0, (
            f'The argument `zooming_step` should be a positive number. '
            f'Got zooming_step = {zooming_step}.'
        )

        self.rotation_step = rotation_step
        self.zooming_step = zooming_step
        self.viewing_angle = min_viewing_angle
        self.min_viewing_angle = min_viewing_angle
        self.max_sight_range = max_sight_range
        self.area_product = min_viewing_angle * np.square(max_sight_range)
        self.sight_range_func = None
        self.sight_range_outer_func = None
        self.boundary = []
        self.boundary_outer = []
        self.obstacles = set()

        super().__init__(
            location=location,
            location_random_range=location_random_range,
            radius=radius,
            sight_range=max_sight_range,
        )

        self.action_space = spaces.Box(
            low=np.asarray([-self.rotation_step, -self.zooming_step]),
            high=np.asarray([self.rotation_step, self.zooming_step]),
            dtype=np.float64,
        )

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = normalize_angle(value)

    def state(self, private=False):
        state = np.concatenate(
            [
                self.location,
                [self.radius],
                polar2cartesian(self.sight_range, self.orientation),
                [self.viewing_angle],
            ]
        )
        if private:
            state = np.append(state, [self.max_sight_range, self.rotation_step, self.zooming_step])
        return state.astype(np.float64)

    def reset(self):
        super().reset()
        self.orientation = self.rotation_step * self.np_random.randint(
            low=0, high=360 / self.rotation_step
        )
        self.viewing_angle = self.np_random.uniform(
            self.min_viewing_angle, consts.MAX_CAMERA_VIEWING_ANGLE
        )
        self.sight_range = np.sqrt(self.area_product / self.viewing_angle)

        self.boundary = [
            Vector2D(norm=self.max_sight_range, angle=angle, origin=self.location)
            for angle in np.linspace(-180.0, +180.0, num=360, endpoint=False)
        ]
        self.boundary_outer = [
            Vector2D(norm=self.max_sight_range, angle=angle, origin=self.location)
            for angle in np.linspace(-180.0, +180.0, num=360, endpoint=False)
        ]
        phis = np.linspace(-180.0, +180.0, num=361, endpoint=True)
        self.sight_range_func = interp1d(phis, self.max_sight_range * np.ones_like(phis))

    def simulate(self, action):
        assert len(action) == consts.CAMERA_ACTION_DIM, f'Got unexpected camera action {action}.'

        delta_angle, delta_viewing_angle = np.clip(
            action, a_min=self.action_space.low, a_max=self.action_space.high
        )

        self.orientation = self.orientation + delta_angle
        self.viewing_angle = np.clip(
            self.viewing_angle + delta_viewing_angle,
            a_min=self.min_viewing_angle,
            a_max=consts.MAX_CAMERA_VIEWING_ANGLE,
        )
        self.sight_range = np.sqrt(self.area_product / self.viewing_angle)

    def add_obstacles(self, *obstacles):  # pylint: disable=too-many-locals
        obstacles = set(
            filter(
                lambda obstacle: self.distance(obstacle) < self.max_sight_range + obstacle.radius,
                filter(lambda obstacle: obstacle is not self, obstacles),
            )
        )
        self.obstacles.update(obstacles)

        boundary = self.boundary
        boundary_outer = self.boundary_outer
        for obstacle in obstacles:
            if obstacle.transmittance == 1.0:
                continue

            relative = obstacle - self
            if obstacle.radius > relative.norm:
                boundary = [
                    Vector2D(norm=0, angle=angle, origin=self.location)
                    for angle in range(-180, 180, 90)
                ]
                boundary_outer = [
                    Vector2D(norm=0, angle=angle, origin=self.location)
                    for angle in range(-180, 180, 90)
                ]
                break

            half_opening_angle = arcsin_deg(obstacle.radius / relative.norm)
            max_rho = min(self.max_sight_range, relative.norm + obstacle.radius)
            angle_left = relative.angle - half_opening_angle
            angle_right = relative.angle + half_opening_angle
            boundary.extend(
                [
                    Vector2D(
                        norm=self.max_sight_range, angle=angle_left - 0.01, origin=self.location
                    ),
                    Vector2D(
                        norm=self.max_sight_range, angle=angle_left + 0.01, origin=self.location
                    ),
                    Vector2D(
                        norm=self.max_sight_range, angle=angle_right - 0.01, origin=self.location
                    ),
                    Vector2D(
                        norm=self.max_sight_range, angle=angle_right + 0.01, origin=self.location
                    ),
                ]
                + [
                    Vector2D(norm=max_rho, angle=angle, origin=self.location)
                    for angle in np.linspace(
                        angle_left,
                        angle_right,
                        num=max(16, int(2 * half_opening_angle)) + 1,
                        endpoint=True,
                    )
                ]
            )

            boundary_outer.extend(
                [
                    Vector2D(norm=max_rho, angle=angle, origin=self.location)
                    for angle in np.linspace(
                        angle_left,
                        angle_right,
                        num=max(16, int(2 * half_opening_angle)) + 1,
                        endpoint=True,
                    )
                ]
            )

            near_rho = min(
                self.max_sight_range, np.sqrt(np.square(relative.norm) + np.square(obstacle.radius))
            )
            far_rho = self.max_sight_range

            near = Vector2D(norm=near_rho, angle=angle_left, origin=self.location)
            far = Vector2D(norm=far_rho, angle=angle_left - 0.01, origin=self.location)
            for t in np.linspace(start=0.0, stop=1.0, num=21, endpoint=True):
                x = (1.0 - t) * near.x + t * far.x
                y = (1.0 - t) * near.y + t * far.y
                boundary_outer.append(Vector2D(vector=(x, y), origin=self.location))

            near = Vector2D(norm=near_rho, angle=angle_right, origin=self.location)
            far = Vector2D(norm=far_rho, angle=angle_right + 0.01, origin=self.location)
            for t in np.linspace(start=0.0, stop=1.0, num=21, endpoint=True):
                x = (1.0 - t) * near.x + t * far.x
                y = (1.0 - t) * near.y + t * far.y
                boundary_outer.append(Vector2D(vector=(x, y), origin=self.location))

        for obstacle in obstacles:
            if obstacle.transmittance == 1.0:
                continue

            boundary = [obstacle.obstruct(b) for b in boundary]
            boundary_outer = [obstacle.obstruct(b, outer=True) for b in boundary_outer]

        def interpolate(boundary):
            boundary.sort(key=lambda ray: ray.angle)

            boundary, unfiltered = [], boundary
            for ray in unfiltered:
                if len(boundary) > 0 and boundary[-1].angle == ray.angle:
                    if boundary[-1].norm > ray.norm:
                        boundary[-1] = ray
                else:
                    boundary.append(ray)

            rhos = [ray.norm for ray in boundary]
            phis = [ray.angle for ray in boundary]
            rhos.append(rhos[0])
            phis.append(phis[0] + 360)

            rhos = np.asarray(rhos, dtype=np.float64)
            phis = np.asarray(phis, dtype=np.float64)

            return interp1d(phis, rhos), boundary

        self.sight_range_func, self.boundary = interpolate(boundary)
        self.sight_range_outer_func, self.boundary_outer = interpolate(boundary_outer)

    def clear_obstacles(self):
        self.obstacles.clear()

    def overlap(self, other, min_distance=0.0):
        if super().overlap(other, min_distance):
            return True
        if isinstance(other, Camera):
            return self.distance(other) < 0.1 * min(self.sight_range, other.sight_range)
        return False

    def perceive(self, other, transmittance=0.0):  # pylint: disable=arguments-differ
        assert isinstance(other, (Target, Camera))

        relative = Vector2D(vector=other.location - self.location)
        if relative.norm > self.sight_range:
            return False

        relative_angle = abs(self.orientation - relative.angle)
        relative_angle = min(relative_angle, 360 - relative_angle)
        if relative_angle * 2.0 > self.viewing_angle:
            return False

        if self.np_random.binomial(1, transmittance) != 0:
            return True
        return relative.norm <= self.sight_range_at(relative.angle) * (1 + 1e-6)

    def sight_range_at(self, angle, outer=False):
        angle = normalize_angle(angle)
        if outer:
            return self.sight_range_outer_func(angle)
        return self.sight_range_func(angle)

    def boundary_between(self, angle_left, angle_right, outer=False):
        assert 0.0 < angle_right - angle_left <= 360.0

        normalized_angle_left = normalize_angle(angle_left)
        angle_left, angle_right = normalized_angle_left, normalized_angle_left + (
            angle_right - angle_left
        )

        if outer:
            phis_all = self.sight_range_outer_func.x
            rhos_all = self.sight_range_outer_func.y
        else:
            phis_all = self.sight_range_func.x
            rhos_all = self.sight_range_func.y

        if angle_right <= +180.0:
            mask = np.logical_and(angle_left < phis_all, phis_all < angle_right)
            phis = phis_all[mask]
            rhos = rhos_all[mask]
        else:
            mask1 = np.logical_and(angle_left < phis_all, phis_all <= +180.0)
            mask2 = np.logical_and(phis_all > -180.0, phis_all < angle_right - 360.0)
            phis = np.concatenate([phis_all[mask1], phis_all[mask2]])
            rhos = np.concatenate([rhos_all[mask1], rhos_all[mask2]])

        phis = np.concatenate([[angle_left], phis, [angle_right]])
        rhos = np.concatenate(
            [[self.sight_range_at(angle_left)], rhos, [self.sight_range_at(angle_right)]]
        )

        return phis.astype(np.float64), rhos.astype(np.float64)


class Target(Sensor):  # pylint: disable=too-many-instance-attributes
    COLOR_UNTRACKED = (1.0, 0.0, 0.0)
    COLOR_TRACKED = (1.0, 1.0, 0.0)
    COLOR_NO_LOAD = (0.2, 0.6, 0.2)
    COLOR = COLOR_UNTRACKED

    STATE_DIM = STATE_DIM_PUBLIC = consts.TARGET_STATE_DIM_PUBLIC  # 4
    STATE_DIM_PRIVATE = consts.TARGET_STATE_DIM_PUBLIC  # 6 + 2 * NUM_WAREHOUSES = 14
    state_space = state_space_public = consts.TARGET_STATE_SPACE_PUBLIC
    state_space_private = consts.TARGET_STATE_SPACE_PRIVATE

    ACTION_DIM = consts.TARGET_ACTION_DIM  # 2
    DEFAULT_ACTION = consts.TARGET_DEFAULT_ACTION  # np.array([0.0, 0.0], dtype=np.float64)

    SPATIAL_HASHMAP = SpatialHashmap(step=consts.TERRAIN_WIDTH / SPATIAL_GRID_RESOLUTION)
    OBSTACLES = set()

    DEFAULTS = {
        'sight_range': DEFAULT_SIGHT_RANGE,
        'step_size': DEFAULT_TARGET_STEP_SIZE,
    }

    def __init__(
        self,
        location=None,
        location_random_range=None,
        sight_range=DEFAULT_SIGHT_RANGE,
        step_size=DEFAULT_TARGET_STEP_SIZE,
    ):
        assert step_size > 0.0, (
            f'The argument `step_size` should be a positive number. '
            f'Got step_size = {step_size}.'
        )

        self.transport_product = step_size

        self.obstacles = []
        self.capacity = 1
        self.goal_bits = np.zeros(consts.NUM_WAREHOUSES, dtype=np.int64)
        self.empty_bits = np.zeros(consts.NUM_WAREHOUSES, dtype=np.bool8)

        self.is_colliding = False

        super().__init__(
            location=location,
            location_random_range=location_random_range,
            radius=consts.TARGET_RADIUS,
            sight_range=sight_range,
        )

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        assert (
            value > 0.0
        ), f'The argument `step_size` should be a positive number. Got step_size = {value}.'
        self._step_size = value
        self._action_space = None

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = value
        self.step_size = self.transport_product / value

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = spaces.Box(
                low=np.asarray([-self.step_size, -self.step_size]),
                high=np.asarray([self.step_size, self.step_size]),
                dtype=np.float64,
            )
        return self._action_space

    @property
    def is_loaded(self):
        return self.goal_bits.any()

    def state(self, private=False):
        state = np.append(self.location, [self.sight_range, self.is_loaded])
        if private:
            state = np.concatenate(
                [state, [self.step_size, self.capacity], self.goal_bits, self.empty_bits]
            )
        return state.astype(np.float64)

    def reset(self):
        super().reset()
        self.goal_bits.fill(0)
        self.empty_bits.fill(False)
        self.is_colliding = False

    def simulate(self, action):
        assert len(action) == consts.TARGET_ACTION_DIM, f'Got unexpected target action: {action}.'

        step = Vector2D(vector=action, origin=self.location)
        if step.norm > self.step_size:
            step.norm = self.step_size

        desired_location = step.endpoint.copy()

        if len(self.OBSTACLES) > 0:
            obstacles = set()
            ix_low, iy_low = self.SPATIAL_HASHMAP.hash_key(self.location - self.step_size)
            ix_high, iy_high = self.SPATIAL_HASHMAP.hash_key(self.location + self.step_size)
            for ix in range(ix_low, ix_high + 1):
                for iy in range(iy_low, iy_high + 1):
                    obstacles.update(self.SPATIAL_HASHMAP[ix, iy])
            for obstacle in obstacles:
                step = obstacle.obstruct(step, keep_tangential=True)

        self.location = step.endpoint.clip(
            min=consts.TERRAIN_SPACE.low, max=consts.TERRAIN_SPACE.high
        )

        self.is_colliding = not np.allclose(self.location, desired_location, rtol=0.0, atol=1e-6)

    @classmethod
    def add_obstacles(cls, *obstacles, epsilon=1e-5):  # pylint: disable=arguments-differ
        for obstacle in obstacles:
            if obstacle in cls.OBSTACLES:
                continue

            ix_low, iy_low = cls.SPATIAL_HASHMAP.hash_key(
                obstacle.location - obstacle.radius - epsilon
            )
            ix_high, iy_high = cls.SPATIAL_HASHMAP.hash_key(
                obstacle.location + obstacle.radius + epsilon
            )
            for ix in range(ix_low, ix_high + 1):
                for iy in range(iy_low, iy_high + 1):
                    cls.SPATIAL_HASHMAP[ix, iy].add(obstacle)

            cls.OBSTACLES.add(obstacle)

    @classmethod
    def clear_obstacles(cls):  # pylint: disable=arguments-differ
        cls.SPATIAL_HASHMAP.clear()
        cls.OBSTACLES.clear()
