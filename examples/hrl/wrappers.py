import re

import gym
import numpy as np
from gym import spaces

import mate
from examples.utils import CustomMetricCallback, MetricCollector


__all__ = [
    'HierarchicalCamera',
    'MultiDiscrete2DiscreteActionMapper',
    'FlattenMultiDiscrete',
    'DiscreteMultiSelection',
]


class HierarchicalCamera(gym.Wrapper, metaclass=mate.WrapperMeta):
    INFO_KEYS = {
        'raw_reward': 'sum',
        'normalized_raw_reward': 'sum',
        re.compile(r'^auxiliary_reward(\w*)$'): 'sum',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
        'num_selected_targets': 'mean',
        'num_valid_selected_targets': 'mean',
        'num_invalid_selected_targets': 'mean',
        'invalid_target_selection_rate': 'mean',
    }

    def __init__(self, env, multi_selection=True, frame_skip=1, custom_metrics=None):
        assert isinstance(env, mate.MultiCamera), (
            f'You should use wrapper `{self.__class__}` with wrapper `MultiCamera`. '
            f'Please wrap the environment with wrapper `MultiCamera` first. '
            f'Got env = {env}.'
        )
        assert not isinstance(
            env, HierarchicalCamera
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'

        super().__init__(env)

        self.multi_selection = multi_selection
        if self.multi_selection:
            self.camera_action_space = spaces.MultiDiscrete((2,) * env.num_targets)
            self.action_mask_space = spaces.MultiBinary(2 * env.num_targets)
        else:
            self.camera_action_space = spaces.Discrete(env.num_targets + 1)
            self.action_mask_space = spaces.MultiBinary(env.num_targets + 1)
        self.action_space = spaces.Tuple(spaces=(self.camera_action_space,) * env.num_cameras)
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

        self.observation_slices = mate.camera_observation_slices_of(
            env.num_cameras, env.num_targets, env.num_obstacles
        )
        self.target_view_mask_slice = self.observation_slices['opponent_mask']

        self.index2onehot = np.eye(env.num_targets + 1, env.num_targets, dtype=np.bool8)
        self.last_observations = None

        self.frame_skip = frame_skip
        self.custom_metrics = custom_metrics or CustomMetricCallback.DEFAULT_CUSTOM_METRICS
        self.custom_metrics.update(
            {
                'num_selected_targets': 'mean',
                'num_valid_selected_targets': 'mean',
                'num_invalid_selected_targets': 'mean',
                'invalid_target_selection_rate': 'mean',
            }
        )

    def load_config(self, config=None):
        self.env.load_config(config=config)

        self.__init__(
            self.env,
            multi_selection=self.multi_selection,
            frame_skip=self.frame_skip,
            custom_metrics=self.custom_metrics,
        )

    def reset(self, **kwargs):
        self.last_observations = observations = self.env.reset(**kwargs)

        return observations

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if self.multi_selection:
            action = action.reshape(self.num_cameras, self.num_targets)
        else:
            action = action.reshape(self.num_cameras)
        assert self.camera_joint_action_space.contains(
            tuple(action)
        ), f'Joint action {tuple(action)} outside given joint action space {self.camera_joint_action_space}.'

        if not self.multi_selection:
            action = self.index2onehot[action]
        else:
            action = action.astype(np.bool8)

        fragment_rewards = []
        if self.frame_skip > 1:
            metric_collectors = [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
        else:
            metric_collectors = []

        observations = self.last_observations
        for f in range(self.frame_skip):
            observations, rewards, dones, infos = self.env.step(
                self.joint_executor(action, observations)
            )

            for c in range(self.num_cameras):
                target_selection = action[c].astype(np.bool8)
                target_view_mask = observations[c, self.target_view_mask_slice].astype(np.bool8)
                num_selected_targets = target_selection.sum()
                num_valid_selected_targets = np.logical_and(
                    target_selection, target_view_mask
                ).sum()
                num_invalid_selected_targets = np.logical_and(
                    target_selection, np.logical_not(target_view_mask)
                ).sum()
                invalid_target_selection_rate = num_invalid_selected_targets / max(
                    1, num_selected_targets
                )
                infos[c]['num_selected_targets'] = num_selected_targets
                infos[c]['num_valid_selected_targets'] = num_valid_selected_targets
                infos[c]['num_invalid_selected_targets'] = num_invalid_selected_targets
                infos[c]['invalid_target_selection_rate'] = invalid_target_selection_rate

            if self.frame_skip > 1:
                fragment_rewards.append(rewards)
                for collector, info in zip(metric_collectors, infos):
                    collector.add(info)

            if all(dones):
                break

        self.last_observations = observations
        if self.frame_skip > 1:
            rewards = np.sum(fragment_rewards, axis=0).tolist()
            for collector, info in zip(metric_collectors, infos):
                info.update(collector.collect())

        return observations, rewards, dones, infos

    def joint_executor(self, joint_action, joint_observation):
        actions = []
        for camera, target_selection_bits, observation in zip(
            self.cameras, joint_action, joint_observation
        ):
            target_view_mask = observation[self.target_view_mask_slice].astype(np.bool8)
            actions.append(
                self.executor(camera, self.targets, target_selection_bits, target_view_mask)
            )

        return np.asarray(actions, dtype=np.float64)

    def action_mask(self, observation):
        target_view_mask = observation[self.target_view_mask_slice].ravel().astype(np.bool8)

        if self.multi_selection:
            action_mask = np.repeat(target_view_mask, repeats=2)
            action_mask[::2] = True
        else:
            action_mask = np.append(target_view_mask, True)

        return action_mask

    @staticmethod
    def executor(camera, targets, target_selection_bits, target_view_mask):
        target_bits = np.logical_and(target_selection_bits, target_view_mask)
        targets = [targets[t] for t in np.flatnonzero(target_bits)]
        return HierarchicalCamera.track(camera, targets)

    @staticmethod
    def track(camera, targets):
        if len(targets) == 0:
            return camera.action_space.low

        center = np.mean([target.location for target in targets], axis=0)

        def best_orientation():
            direction = center - camera.location
            return mate.arctan2_deg(direction[-1], direction[0])

        def best_viewing_angle():
            distance = np.linalg.norm(center - camera.location)

            if (
                distance * (1.0 + mate.sin_deg(camera.min_viewing_angle / 2.0))
                >= camera.max_sight_range
            ):
                return camera.min_viewing_angle

            area_product = camera.viewing_angle * np.square(camera.sight_range)
            if distance <= np.sqrt(area_product / 180.0) / 2.0:
                return min(180.0, mate.MAX_CAMERA_VIEWING_ANGLE)

            best = min(180.0, mate.MAX_CAMERA_VIEWING_ANGLE)
            for _ in range(20):
                sight_range = distance * (1.0 + mate.sin_deg(min(best / 2.0, 90.0)))
                best = area_product / np.square(sight_range)
            return np.clip(
                best, a_min=camera.min_viewing_angle, a_max=mate.MAX_CAMERA_VIEWING_ANGLE
            )

        return np.asarray(
            [
                mate.normalize_angle(best_orientation() - camera.orientation),
                best_viewing_angle() - camera.viewing_angle,
            ]
        ).clip(min=camera.action_space.low, max=camera.action_space.high)

    @staticmethod
    def render_selection_callback(unwrapped, mode, selections):
        import mate.assets.pygletrendering as rendering

        geoms = []
        for c, selection, mask in selections:
            camera = unwrapped.cameras[c]
            valid_selection = np.logical_and(selection, mask)

            if valid_selection.any():
                valid_selected = [unwrapped.targets[t] for t in np.flatnonzero(valid_selection)]
                center = np.mean([target.location for target in valid_selected], axis=0)
                line = rendering.Line(camera.location, center)
                line.set_color(1.0, 0.0, 0.0, 0.5)
                line.linewidth.stroke = 2
                geoms.append(line)
            else:
                center = camera.location

            for t in np.flatnonzero(selection):
                target = unwrapped.targets[t]
                line = rendering.Line(camera.location, target.location)
                if mask[t]:
                    line.linewidth.stroke = 2
                    center_line = rendering.Line(center, target.location)
                    center_line.linewidth.stroke = 1.5
                    center_line.set_color(1.0, 0.0, 0.0, 0.5)
                    geoms.append(center_line)
                else:
                    line.set_color(0.0, 0.0, 0.0, 0.3)
                    line.add_attr(rendering.LineStyle(0x0F0F))
                geoms.append(line)

        unwrapped.viewer.onetime_geoms[:] = geoms + unwrapped.viewer.onetime_geoms[:]


class MultiDiscrete2DiscreteActionMapper:
    def __init__(self, original_space):
        assert isinstance(original_space, spaces.MultiDiscrete)
        self.nvec = original_space.nvec
        self.original_space = original_space
        self.original_mask_space = spaces.MultiBinary(np.sum(self.nvec))

        self.n = np.prod(self.nvec)
        self.space = spaces.Discrete(self.n)
        self.mask_space = spaces.MultiBinary(self.n)

        self.strides = np.asarray(
            list(reversed(np.cumprod(list(reversed(self.nvec.ravel())))))[1:] + [1],
            dtype=self.space.dtype,
        )

        self._mask_mapping = None

    @property
    def mask_table(self):
        if self._mask_mapping is None:
            self._mask_mapping = np.zeros((self.n, np.sum(self.nvec)), dtype=np.bool8)
            all_multi_discrete_actions = self.multi_discrete_action_batched(
                list(range(self.n)), strict=False
            )
            offsets = np.cumsum([0, *self.nvec.ravel()[:-1]], dtype=np.int64)
            indices = all_multi_discrete_actions.reshape(self.n, -1) + offsets[np.newaxis, :]
            for n, index in enumerate(indices):
                self._mask_mapping[n, index] = True

        return self._mask_mapping

    def multi_discrete_action_batched(self, discrete_action_batch, strict=True):
        discrete_action_batch = np.asarray(discrete_action_batch, dtype=self.space.dtype)

        assert discrete_action_batch.ndim == 1
        if strict:
            for discrete_action in discrete_action_batch:
                assert self.space.contains(discrete_action), (
                    f'Discrete action {discrete_action} outside given '
                    f'discrete action space {self.space}.'
                )

        multi_discrete_action_batch = []
        for s in self.strides:
            multi_discrete_action_batch.append(discrete_action_batch // s)
            discrete_action_batch = discrete_action_batch % s

        multi_discrete_action_batch = np.stack(multi_discrete_action_batch, axis=-1)
        return multi_discrete_action_batch.reshape(-1, *self.original_space.shape).astype(
            self.original_space.dtype
        )

    def multi_discrete_action(self, discrete_action):
        return self.multi_discrete_action_batched([discrete_action])[0]

    def discrete_action_batched(self, multi_discrete_action_batch, strict=True):
        multi_discrete_action_batch = np.asarray(
            multi_discrete_action_batch, dtype=self.original_space.dtype
        )

        assert multi_discrete_action_batch.shape[1:] == self.nvec.shape
        if strict:
            for multi_discrete_action in multi_discrete_action_batch:
                assert self.original_space.contains(multi_discrete_action), (
                    f'Multi-discrete action {multi_discrete_action} outside given '
                    f'multi-discrete action space {self.original_space}.'
                )

        batch_size = multi_discrete_action_batch.shape[0]
        multi_discrete_action_batch = multi_discrete_action_batch.reshape(batch_size, -1)

        discrete_action_batch = (self.strides[np.newaxis, :] * multi_discrete_action_batch).sum(
            axis=-1
        )
        return discrete_action_batch.astype(self.space.dtype).ravel()

    def discrete_action(self, multi_discrete_action):
        return self.discrete_action_batched([multi_discrete_action])[0]

    def discrete_action_mask(self, multi_discrete_action_mask):
        multi_discrete_action_mask = np.asarray(multi_discrete_action_mask, dtype=np.bool8)

        assert self.original_mask_space.contains(multi_discrete_action_mask), (
            f'Multi-discrete action mask {multi_discrete_action_mask} outside given '
            f'Multi-discrete action mask space {self.original_mask_space}.'
        )

        return (multi_discrete_action_mask >= self.mask_table).all(axis=-1)


class FlattenMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.action_mapper = MultiDiscrete2DiscreteActionMapper(original_space=env.action_space)
        self.action_space = self.action_mapper.space

    def action(self, action):
        return self.action_mapper.multi_discrete_action(action)

    def reverse_action(self, action):
        return self.action_mapper.discrete_action(action)


class DiscreteMultiSelection(gym.ActionWrapper, metaclass=mate.WrapperMeta):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env, HierarchicalCamera)
        assert env.multi_selection

        self.action_mapper = MultiDiscrete2DiscreteActionMapper(
            original_space=env.camera_action_space
        )

        self.camera_action_space = self.action_mapper.space
        self.action_mask_space = self.action_mapper.mask_space
        self.action_space = spaces.Tuple(spaces=(self.camera_action_space,) * env.num_cameras)
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

    def action(self, action):
        return self.action_mapper.multi_discrete_action_batched(action)

    def reverse_action(self, action):
        return self.action_mapper.discrete_action_batched(action)

    def action_mask(self, observation):
        action_mask = self.env.action_mask(observation)

        return self.action_mapper.discrete_action_mask(action_mask)
