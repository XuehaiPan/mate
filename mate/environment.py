"""The Multi-Agent Tracking Environment."""

# pylint: disable=too-many-lines

import copy
import itertools
import os
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import gym
import numpy as np
from gym import spaces
from gym.utils import EzPickle, seeding

from mate import constants as consts
from mate.entities import Camera, Obstacle, Target
from mate.utils import Message, Team, arctan2_deg, normalize_angle, polar2cartesian


__all__ = ['ASSETS_DIR', 'DEFAULT_CONFIG_FILE', 'read_config', 'EnvMeta', 'MultiAgentTracking']

ASSETS_DIR = Path(__file__).absolute().parent / 'assets'
"""The asset directory path."""

DEFAULT_CONFIG_FILE = ASSETS_DIR / 'MATE-4v8-9.yaml'
"""The default configuration file."""

DEFAULT_WINDOW_SIZE = 800
TARGET_RENDER_RADIUS = 27.5

WAREHOUSE_COLORS = [
    (52 / 255, 127 / 255, 212 / 255),
    (255 / 255, 34 / 255, 34 / 255),
    (149 / 255, 117 / 255, 205 / 255),
    (134 / 255, 110 / 255, 68 / 255),
]

assert len(WAREHOUSE_COLORS) >= consts.NUM_WAREHOUSES >= 2

NUM_RESET_RETRIES = 500

if TYPE_CHECKING:
    from mate.agents import AgentType


def _did_you_mean(path: Union[str, os.PathLike]) -> Tuple[str, ...]:
    path = str(path)

    def edit_distance(str1: str, str2: str) -> int:
        dis = {
            **{(i, 0): i for i in range(len(str1) + 1)},
            **{(0, j): j for j in range(len(str2) + 1)},
        }
        for i, j in itertools.product(range(1, len(str1) + 1), range(1, len(str2) + 1)):
            dis[i, j] = min(
                dis[i - 1, j - 1] + int(str1[i - 1] != str2[j - 1]),
                dis[i - 1, j] + 1,
                dis[i, j - 1] + 1,
            )
        return dis[len(str1), len(str2)]

    candidates = tuple(
        itertools.starmap(
            os.path.join,
            sorted(
                map(
                    os.path.split,
                    itertools.chain.from_iterable(
                        DIR.glob(pattern)
                        for pattern in ('*.yaml', '*.yml', '*.json')
                        for DIR in (Path(os.getcwd()), ASSETS_DIR)
                    ),
                ),
                key=lambda split: (
                    edit_distance(split[1], path),
                    split[0] == str(ASSETS_DIR),
                    split[1],
                ),
            ),
        )
    )

    return candidates


def _deep_update(dict1: Dict[str, Any], dict2: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    dict1, dict2 = copy.deepcopy(dict1), copy.deepcopy(dict2)
    for key, value in dict2.items():
        if isinstance(dict1.get(key, None), dict) and isinstance(value, dict):
            value = _deep_update(dict1[key], value, prefix=f'{key}/')
        elif key in dict1:
            gym.logger.warn(f'Override configuration "{prefix}{key}" with `{value!r}`.')
        else:
            gym.logger.warn(f'Set configuration "{prefix}{key}" with `{value!r}`.')
        dict1[key] = value
    return dict1


# pylint: disable-next=too-many-branches
def read_config(
    config_or_path: Optional[Union[Dict[str, Any], str]] = None, **kwargs
) -> Dict[str, Any]:
    """Load configuration from a dictionary mapping or a JSON/YAML file."""

    if isinstance(config_or_path, str) and not os.path.exists(config_or_path):
        for candidate in (Path(os.getcwd()) / config_or_path, ASSETS_DIR / config_or_path):
            if candidate.is_file():
                gym.logger.info(
                    'Found configuration file "%s" in assets directory.', config_or_path
                )
                config_or_path = candidate
                break
        else:
            candidates = _did_you_mean(config_or_path)
            raise ValueError(
                f'Cannot found the configuration file "{config_or_path}". '
                f'Did you mean: "{candidates[0]}"?'
            )

    if config_or_path is None:
        config = {}
    elif not isinstance(config_or_path, Mapping):
        config = None
        if isinstance(config_or_path, os.PathLike):
            config_or_path = str(config_or_path)
        if isinstance(config_or_path, str) and os.path.exists(config_or_path):
            file_ext = os.path.splitext(config_or_path)[1].lower()
            if file_ext in ('.json', '.yaml', '.yml'):
                with open(config_or_path, encoding='UTF-8') as file:
                    if file_ext == '.json':
                        import json  # pylint: disable=import-outside-toplevel

                        config = json.load(file)
                    else:
                        import yaml  # pylint: disable=import-outside-toplevel

                        config = yaml.load(file, yaml.SafeLoader)
        if config is None:
            raise ValueError(
                f'The configuration should be a dictionary mapping '
                f'or a path to a readable JSON/YAML file. '
                f'Got {config_or_path!r}.'
            )
    else:
        config = dict(config_or_path)

    config = _deep_update(config, kwargs)
    validate_config(config)

    def to_box(random_range):
        if isinstance(random_range, spaces.Box):
            low, high = random_range.low, random_range.high
        elif isinstance(random_range, dict):
            low, high = random_range['low'], random_range['high']
        else:
            low, high = random_range[0::2], random_range[1::2]
            if len(low) == 1 and len(high) == 1:
                low, high = low[0], high[0]

        return spaces.Box(
            low=np.array(low, dtype=np.float64, copy=True),
            high=np.array(high, dtype=np.float64, copy=True),
            dtype=np.float64,
        )

    for entity in ('camera', 'obstacle', 'target'):
        config.setdefault(entity, {})
        subconfig = config[entity]
        if 'location' in subconfig:
            subconfig['location'] = [
                np.asarray(array, dtype=np.float64) for array in subconfig['location']
            ]
        if 'location_random_range' in subconfig:
            subconfig['location_random_range'] = list(
                map(to_box, subconfig['location_random_range'])
            )
        if 'radius_random_range' in subconfig:
            subconfig['radius_random_range'] = to_box(subconfig['radius_random_range'])

    return config


def validate_config(config: Dict[str, Any]) -> None:  # pylint: disable=too-many-branches
    """Validate configuration."""

    if 'max_episode_steps' not in config:
        gym.logger.warn('Missing key "max_episode_steps", set to 10000.')
        config['max_episode_steps'] = 10000
    if config['max_episode_steps'] <= 0:
        raise ValueError('`max_episode_steps` must be a positive integer.')

    if 'reward_type' not in config:
        gym.logger.warn('Missing key "reward_type", set to "dense".')
        config['reward_type'] = 'dense'
    if config['reward_type'] not in ('dense', 'sparse'):
        raise ValueError(
            f'Invalid reward type {config["reward_type"]}. Expect one of {("dense", "sparse")}'
        )

    if 'target' not in config:
        raise ValueError(
            'Missing key "target". There must be at least one target in the environment.'
        )

    target = config['target']
    num_targets = len(target.get('location', [])) + len(target.get('location_random_range', []))
    if num_targets == 0:
        raise ValueError('There must be at least one target in the environment.')

    if 'num_cargoes_per_target' not in config:
        raise ValueError('Missing key "num_cargoes_per_target".')
    if config['num_cargoes_per_target'] < consts.NUM_WAREHOUSES:
        raise ValueError(
            f'`num_cargoes_per_target` should be no less than {consts.NUM_WAREHOUSES}. '
            f'Got {config["num_cargoes_per_target"]}.'
        )

    if 'high_capacity_target_split' not in config:
        gym.logger.warn('Missing key "high_capacity_target_split", set to 0.5.')
        config['high_capacity_target_split'] = 0.5
    if not 0.0 <= config['high_capacity_target_split'] <= 1.0:
        raise ValueError(
            f'`high_capacity_target_split` must be between 0 and 1. '
            f'Got {config["high_capacity_target_split"]}.'
        )

    if 'targets_start_with_cargoes' not in config:
        gym.logger.warn('Missing key "targets_start_with_cargoes", set to True.')
        config['targets_start_with_cargoes'] = True
    config['targets_start_with_cargoes'] = bool(config['targets_start_with_cargoes'])

    if 'bounty_factor' not in config:
        gym.logger.warn('Missing key "bounty_factor", set to 1.0.')
        config['bounty_factor'] = 1.0
    if not config['bounty_factor'] >= 0.0:
        raise ValueError(
            f'`bounty_factor` must be a non-negative number. Got {config["bounty_factor"]}.'
        )

    if 'shuffle_entities' not in config:
        gym.logger.warn('Missing key "shuffle_entities", set to True.')
        config['shuffle_entities'] = True
    config['shuffle_entities'] = bool(config['shuffle_entities'])

    for Entity in (Camera, Target):
        entity = Entity.__name__.lower()
        if entity in config:
            for key, default in Entity.DEFAULTS.items():
                if key not in config[entity]:
                    gym.logger.warn(f'Missing key "{entity}/{key}", set to {default}.')
                    config[entity][key] = default
                if not config[entity][key] > 0.0:
                    raise ValueError(
                        f'`{entity}/{key}` must be a positive number. '
                        f'Got {config[entity][key]}.'
                    )


class EnvMeta(type(gym.Env)):
    """Helper metaclass for instance check."""

    def __instancecheck__(cls, instance):
        if super().__instancecheck__(instance):
            return True

        while issubclass(type(instance), gym.Wrapper):
            instance = instance.env
            if super().__instancecheck__(instance):
                return True

        return False


# pylint: disable-next=too-many-instance-attributes,too-many-public-methods
class MultiAgentTracking(gym.Env, EzPickle, metaclass=EnvMeta):
    """The main class of the Multi-Agent Tracking Environment. It encapsulates
    an environment with arbitrary behind-the-scenes dynamics. This environment
    is partially observed for both teams.

    The main API methods that users of this class need to know are:

        - step
        - reset
        - render
        - close
        - seed
        - send_messages     <- new method
        - receive_messages  <- new method
        - load_config       <- new method

    And set the following attributes:

        action_space: A tuple of two Space objects corresponding to valid joint actions of cameras and targets
        camera_action_space: The Space object corresponding to a single camera's valid actions
        camera_joint_action_space: The Space object corresponding to valid joint actions of all cameras
        target_action_space: The Space object corresponding to a single target's valid actions
        target_joint_action_space: The Space object corresponding to valid joint actions of all targets
        observation_space: A tuple of two Space objects corresponding to valid joint observations of cameras and targets
        camera_observation_space: The Space object corresponding to a single camera's valid observations
        camera_joint_observation_space: The Space object corresponding to valid joint observations of all cameras
        target_observation_space: The Space object corresponding to a single target's valid observations
        target_joint_observation_space: The Space object corresponding to valid joint observations of all targets

    The methods are accessed publicly as "step", "reset", etc...
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60,
        'video.output_frames_per_second': 60,
    }

    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_FILE
    """The default configuration file."""

    # pylint: disable-next=too-many-statements
    def __init__(self, config: Optional[Union[Dict[str, Any], str]] = None, **kwargs) -> None:
        """Initialize the Multi-Agent Tracking Environment from a dictionary
        mapping or a JSON/YAML file.

        Parameters:
            config (Optional[Union[Dict[str, Any], str]]): a dictionary mapping or a path to a readable JSON/YAML file
        """

        if config is None:
            config = {} if len(kwargs) > 0 else self.DEFAULT_CONFIG_FILE

        config = read_config(config, **kwargs)

        EzPickle.__init__(self, config, **kwargs)

        self.config = config

        assert self.num_cargoes_per_target >= self.num_warehouses, (
            f'The number of cargoes per target must be no less than {self.num_warehouses}. '
            f'Got num_cargoes_per_target = {self.num_cargoes_per_target}.'
        )
        self._num_cameras = None
        self._num_targets = None
        self._num_obstacles = None
        self._camera_state_dim = None
        self._target_state_dim = None
        self._obstacle_state_dim = None
        self._camera_observation_dim = None
        self._target_observation_dim = None
        self._target_step_size = None
        self._high_capacity_target_split = None
        self._num_high_capacity_targets = None
        self._num_low_capacity_targets = None
        self._targets_start_with_cargoes = None
        self._bounty_factor = None
        self._obstacle_transmittance = None
        self._shuffle_entities = None
        self._state = None

        def merge_space(iterable):
            space_list = list(iterable)
            if len(space_list) == 0 or any(space is None for space in space_list):
                low = high = np.zeros(0, dtype=np.float64)
            else:
                low = np.min([space.low for space in space_list], axis=0)
                high = np.min([space.high for space in space_list], axis=0)
            return spaces.Box(
                low=low.astype(np.float64), high=high.astype(np.float64), dtype=np.float64
            )

        def make_from_config(entity_class):
            common_kwargs = config.get(entity_class.__name__.lower(), {}).copy()
            locations = common_kwargs.pop('location', [])
            location_random_ranges = common_kwargs.pop('location_random_range', [])
            entities = []
            for location in locations:
                entities.append(entity_class(location=location, **common_kwargs))
            for location_random_range in location_random_ranges:
                entities.append(
                    entity_class(location_random_range=location_random_range, **common_kwargs)
                )

            state_space_public = entity_class.state_space_public
            state_space_private = entity_class.state_space_private
            action_space = merge_space(r.action_space for r in entities)

            return entities, state_space_public, state_space_private, action_space

        (
            self.cameras_ordered,
            self.camera_state_space_public,
            self.camera_state_space_private,
            self.camera_action_space,
        ) = make_from_config(Camera)
        (
            self.targets_ordered,
            self.target_state_space_public,
            self.target_state_space_private,
            self.target_action_space,
        ) = make_from_config(Target)
        self.obstacles_ordered, self.obstacle_state_space, _, _ = make_from_config(Obstacle)

        self.cameras = list(self.cameras_ordered)
        self.targets = list(self.targets_ordered)
        self.obstacles = list(self.obstacles_ordered)

        assert self.num_targets > 0, (
            f'There must be at least one target in the environment. '
            f'Got num_targets = {self.num_targets}.'
        )
        if self.num_cameras == 0:
            self.camera_action_space = spaces.Box(
                low=np.zeros(consts.CAMERA_ACTION_DIM, dtype=np.float64),
                high=np.zeros(consts.CAMERA_ACTION_DIM, dtype=np.float64),
                dtype=np.float64,
            )

        self.camera_joint_action_space = spaces.Tuple(
            spaces=(self.camera_action_space,) * self.num_cameras
        )
        self.target_joint_action_space = spaces.Tuple(
            spaces=(self.target_action_space,) * self.num_targets
        )
        self.action_space = spaces.Tuple(
            spaces=(self.camera_joint_action_space, self.target_joint_action_space)
        )

        numbers = (self.num_cameras, self.num_targets, self.num_obstacles)
        self.camera_observation_space = consts.camera_observation_space_of(*numbers)
        self.target_observation_space = consts.target_observation_space_of(*numbers)
        self.camera_joint_observation_space = spaces.Tuple(
            spaces=(self.camera_observation_space,) * self.num_cameras
        )
        self.target_joint_observation_space = spaces.Tuple(
            spaces=(self.target_observation_space,) * self.num_targets
        )
        self.observation_space = spaces.Tuple(
            spaces=(self.camera_joint_observation_space, self.target_joint_observation_space)
        )

        self.state_space = spaces.Box(
            low=np.concatenate(
                [consts.PRESERVED_SPACE.low]
                + [consts.CAMERA_STATE_SPACE_PRIVATE.low] * self.num_cameras
                + [consts.TARGET_STATE_SPACE_PRIVATE.low] * self.num_targets
                + [consts.OBSTACLE_STATE_SPACE.low] * self.num_obstacles
                + [[0.0] * (2 * self.num_targets + self.num_warehouses * self.num_warehouses)]
            ).astype(np.float64),
            high=np.concatenate(
                [consts.PRESERVED_SPACE.high]
                + [consts.CAMERA_STATE_SPACE_PRIVATE.high] * self.num_cameras
                + [consts.TARGET_STATE_SPACE_PRIVATE.high] * self.num_targets
                + [consts.OBSTACLE_STATE_SPACE.high] * self.num_obstacles
                + [[+np.inf] * (2 * self.num_targets + self.num_warehouses * self.num_warehouses)]
            ).astype(np.float64),
            dtype=np.float64,
        )

        self.obstacle_states = np.zeros(
            (self.num_obstacles, consts.OBSTACLE_STATE_DIM), dtype=np.float64
        )
        self.obstacle_states_flagged = np.zeros(
            (self.num_obstacles, consts.OBSTACLE_STATE_DIM + 1), dtype=np.float64
        )

        self.camera_target_view_mask = np.zeros(
            (self.num_cameras, self.num_targets), dtype=np.bool8
        )
        self.tracked_bits = np.zeros(self.num_targets, dtype=np.bool8)
        self.target_camera_view_mask = np.zeros(
            (self.num_targets, self.num_cameras), dtype=np.bool8
        )

        self.camera_obstacle_view_mask = np.zeros(
            (self.num_cameras, self.num_obstacles), dtype=np.bool8
        )
        self.camera_camera_view_mask = np.zeros(
            (self.num_cameras, self.num_cameras), dtype=np.bool8
        )
        self.target_obstacle_view_mask = np.zeros(
            (self.num_targets, self.num_obstacles), dtype=np.bool8
        )
        self.target_target_view_mask = np.zeros(
            (self.num_targets, self.num_targets), dtype=np.bool8
        )
        self.camera_obstacle_observations = np.zeros(
            (self.num_cameras, self.obstacle_states_flagged.size), dtype=np.float64
        )

        self.preserved_data = np.concatenate(
            [numbers, [0], consts.WAREHOUSES.ravel(), [consts.WAREHOUSE_RADIUS]]
        ).astype(np.float64)

        self.target_capacities = np.ones(self.num_targets, dtype=np.int64)
        self.remaining_cargoes = np.zeros(
            (self.num_warehouses, self.num_warehouses), dtype=np.int64
        )
        self.awaiting_cargo_counts = np.zeros(self.num_warehouses, dtype=np.int64)
        self.num_delivered_cargoes = 0
        self.target_team_episode_reward = 0.0
        self.delayed_target_team_episode_reward = 0.0
        self.target_warehouse_distances = np.zeros(
            (self.num_targets, self.num_warehouses), dtype=np.float64
        )
        self.target_goal_bits = np.zeros((self.num_targets, self.num_warehouses), dtype=np.int64)
        self.target_goals = np.zeros(self.num_targets, dtype=np.int64)
        self.target_goals.fill(-1)
        self.target_dones = np.zeros(self.num_targets, dtype=np.bool8)
        self.target_steps = np.zeros(self.num_targets, dtype=np.int64)
        self.tracked_steps = np.zeros(self.num_targets, dtype=np.int64)

        self.freight_scale = np.ceil(consts.TERRAIN_WIDTH / self.target_step_size)
        self.bounty_scale = np.ceil(self.freight_scale * self.bounty_factor)
        self.reward_scale = self.freight_scale + self.bounty_scale
        self.freights = np.zeros(self.num_targets, dtype=np.int64)
        self.bounties = np.zeros(self.num_targets, dtype=np.int64)
        self._sparse_reward = self.config['reward_type'] == 'sparse'
        self.max_target_team_episode_reward = (
            self.reward_scale * self.num_cargoes_per_target * self.num_targets
        )

        self.coverage_rate = 0.0
        self.real_coverage_rate = 0.0
        self.mean_transport_rate = 0.0

        self.episode_step = 0
        self.viewer = None
        self.render_callbacks = OrderedDict()
        self.target_orientations = np.zeros(self.num_targets, dtype=np.float64)

        self.camera_message_buffer = defaultdict(list)
        self.target_message_buffer = defaultdict(list)
        self.message_buffers = (self.camera_message_buffer, self.target_message_buffer)

        self.camera_message_queue = defaultdict(deque)
        self.target_message_queue = defaultdict(deque)
        self.message_queues = (self.camera_message_queue, self.target_message_queue)

        self.camera_communication_edges = np.zeros(
            (self.num_cameras, self.num_cameras), dtype=np.int64
        )
        self.target_communication_edges = np.zeros(
            (self.num_targets, self.num_targets), dtype=np.int64
        )
        self.camera_total_communication_edges = self.camera_communication_edges.copy()
        self.target_total_communication_edges = self.target_communication_edges.copy()
        self.communication_edges = (
            self.camera_communication_edges,
            self.target_communication_edges,
        )

        self._np_random = None
        self.seed(seed=0)

    def load_config(self, config: Optional[Union[Dict[str, Any], str]] = None) -> None:
        """Reinitialize the Multi-Agent Tracking Environment from a dictionary
        mapping or a JSON/YAML file.

        Parameters:
            config (Optional[Union[Dict[str, Any], str]]): a dictionary mapping or a path to a readable JSON/YAML file

        Examples:
            You can change the environment configuration without creating a new
            environment, and this will keep the wrappers you add.

            >>> env = mate.make('MultiAgentTracking-v0', config='MATE-4v8-9.yaml')
            >>> env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
            >>> print(env)
            <MultiCamera<MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 9 obstacles)>
            >>> env.load_config('MATE-4v2-9.yaml')
            >>> print(env)
            <MultiCamera<MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 2 targets, 9 obstacles)>
        """

        seed = self.np_random.randint(np.iinfo(int).max)

        self.__init__(config=config)  # pylint: disable=unnecessary-dunder-call

        self.seed(seed)

    def step(
        self, action: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, Tuple[List[dict], List[dict]]
    ]:
        """Run one timestep of the environment's dynamics. When end of episode
        is reached, you are responsible for calling `reset()` to reset this
        environment's state.

        Accepts a tuple of cameras' joint action and targets' joint action,
        and returns a tuple (observation, reward, done, info).

        Parameters:
            action (Tuple[np.ndarray, np.ndarray]): a tuple of joint actions provided by the camera agents and the target agents

        Returns:
            observation (Tuple[np.ndarray, np.ndarray]): a tuple of agent's observation of the current environment
            reward (Tuple[float, float]): a tuple of the amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (Tuple[List[dict], List[dict]]): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """  # pylint: disable=line-too-long

        self._simulate(action)
        target_team_reward, delayed_target_team_reward = self._assign_goals()
        self.target_team_episode_reward += target_team_reward
        self.delayed_target_team_episode_reward += delayed_target_team_reward
        camera_joint_observation, target_joint_observation = self.joint_observation()

        if self._sparse_reward:
            target_team_reward = delayed_target_team_reward

        camera_team_reward = -target_team_reward

        normalized_target_team_reward = target_team_reward / self.max_target_team_episode_reward
        normalized_camera_team_reward = -normalized_target_team_reward

        self.target_steps += 1
        self.tracked_steps += self.tracked_bits

        self.episode_step += 1
        done = not (
            self.episode_step <= self.max_episode_steps and self.awaiting_cargo_counts.any()
        )

        common_info = {
            'coverage_rate': self.coverage_rate,
            'real_coverage_rate': self.real_coverage_rate,
            'mean_transport_rate': self.mean_transport_rate,
            'num_delivered_cargoes': self.num_delivered_cargoes,
        }
        camera_infos = [
            {
                'raw_reward': camera_team_reward,
                'normalized_raw_reward': normalized_camera_team_reward,
                'messages': self.camera_message_buffer[c],
                'out_communication_edges': self.camera_communication_edges[c, :].sum(),
                'in_communication_edges': self.camera_communication_edges[:, c].sum(),
                **common_info,
            }
            for c in range(self.num_cameras)
        ]
        target_infos = [
            {
                'raw_reward': target_team_reward,
                'normalized_raw_reward': normalized_target_team_reward,
                'messages': self.target_message_buffer[t],
                'out_communication_edges': self.target_communication_edges[t, :].sum(),
                'in_communication_edges': self.target_communication_edges[:, t].sum(),
                **common_info,
            }
            for t in range(self.num_targets)
        ]
        self.camera_total_communication_edges += self.camera_communication_edges
        self.target_total_communication_edges += self.target_communication_edges
        self.camera_communication_edges.fill(0)
        self.target_communication_edges.fill(0)
        self.camera_message_buffer.clear()
        self.target_message_buffer.clear()
        self.camera_message_queue.clear()
        self.target_message_queue.clear()

        return (
            (camera_joint_observation, target_joint_observation),
            (camera_team_reward, target_team_reward),
            done,
            (camera_infos, target_infos),
        )

    # pylint: disable-next=arguments-differ,too-many-locals,too-many-branches,too-many-statements
    def reset(self, *, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Resets the environment to an initial state and returns an initial
        observation. The entities (cameras, targets and obstacles) may be
        shuffled if not explicitly disabled in configuration.

        Note that unless an explicit seed is provided, this function would not
        reset the environment's random number generator(s). Random variables in
        the environment's state should be sampled independently between multiple
        calls to `reset()`. In other words, each call of `reset()` should yield
        an environment suitable for a new episode, independent of previous
        episodes.

        Args:
            seed (int): the seed for the random number generator(s)

        Returns:
            observations (Tuple[numpy.ndarray, np.ndarray]): the initial observations of all cameras and targets.
        """

        self._destroy()

        if seed is not None:
            self.seed(seed)

        self.cameras = list(self.cameras_ordered)
        self.targets = list(self.targets_ordered)
        self.obstacles = list(self.obstacles_ordered)

        if self.shuffle_entities:
            self.np_random.shuffle(self.cameras)
            self.np_random.shuffle(self.targets)
            self.np_random.shuffle(self.obstacles)

        self.target_capacities.fill(1)
        if self.num_high_capacity_targets > 0:
            if self.shuffle_entities:
                slices = self.np_random.choice(
                    self.num_targets, size=self.num_high_capacity_targets, replace=False
                )
            else:
                slices = slice(0, self.num_high_capacity_targets)
            self.target_capacities[slices] = 2
            for capacity, target in zip(self.target_capacities, self.targets):
                target.capacity = capacity

        reset = [
            Obstacle(location=warehouse, radius=0.75 * consts.WAREHOUSE_RADIUS)
            for warehouse in consts.WAREHOUSES
        ]
        for entity in itertools.chain(self.cameras, self.obstacles, self.targets):
            min_distance = 0.0 if isinstance(entity, Target) else self.target_step_size
            for _ in range(NUM_RESET_RETRIES):
                entity.reset()
                if all(not entity.overlap(r, min_distance) for r in reset):
                    break
            else:
                if entity.__class__ is Obstacle:
                    entity.radius = 0.0
            reset.append(entity)

        for camera in self.cameras:
            camera.clear_obstacles()
            camera.add_obstacles(*self.obstacles)
        Target.clear_obstacles()
        Target.add_obstacles(*self.obstacles, *self.cameras)

        if self.num_obstacles > 0:
            self.obstacle_states = np.vstack(list(map(Obstacle.state, self.obstacles)))
            self.obstacle_states_flagged = np.hstack(
                [self.obstacle_states, np.ones((self.num_obstacles, 1))]
            )
            self.camera_obstacle_view_mask.fill(False)
            self.target_obstacle_view_mask.fill(False)
            for c, camera in enumerate(self.cameras):
                for o, obstacle in enumerate(self.obstacles):
                    if obstacle in camera.obstacles:
                        self.camera_obstacle_view_mask[c, o] = True

            if self.num_cameras > 0:
                camera_obstacle_observations = []
                for c in range(self.num_cameras):
                    obstacle_mask = self.camera_obstacle_view_mask[c, :, np.newaxis]
                    camera_obstacle_observations.append(
                        np.where(obstacle_mask, self.obstacle_states_flagged, 0.0).ravel()
                    )
                self.camera_obstacle_observations = np.vstack(camera_obstacle_observations)

        self._update_view()

        self.remaining_cargoes.fill(0)
        while not self.remaining_cargoes.any(axis=-1).all():
            for _ in range(self.num_cargoes_per_target * self.num_targets):
                sender, recipient = self.np_random.choice(
                    self.num_warehouses, size=2, replace=False
                )
                self.remaining_cargoes[sender, recipient] += 1
            self.awaiting_cargo_counts = self.remaining_cargoes.sum(axis=0)

        self.target_warehouse_distances.fill(0.0)
        self.target_goals.fill(-1)
        self.target_goal_bits.fill(False)
        self.target_steps.fill(0)
        self.tracked_steps.fill(0)
        self.freights.fill(0)
        self.bounties.fill(0)
        self._assign_goals()
        self.target_dones.fill(False)
        self.num_delivered_cargoes = 0
        self.target_team_episode_reward = 0.0
        self.delayed_target_team_episode_reward = 0.0
        if self.targets_start_with_cargoes:
            for t in np.flatnonzero(self.target_goals < 0):
                target = self.targets[t]
                capacity = self.target_capacities[t]
                for warehouse in self.np_random.permutation(self.num_warehouses):
                    if self.remaining_cargoes[warehouse].any():
                        goal = self.np_random.choice(
                            np.flatnonzero(self.remaining_cargoes[warehouse] > 0)
                        )
                        remaining = self.remaining_cargoes[warehouse, goal]
                        cargo_weight = min(capacity, remaining)
                        self.remaining_cargoes[warehouse, goal] -= cargo_weight
                        self.target_goal_bits[t, goal] = cargo_weight
                        self.freights[t] = cargo_weight * self.freight_scale
                        self.bounties[t] = cargo_weight * self.bounty_scale

                        target.goal_bits[goal] = cargo_weight
                        self.target_goals[t] = goal
                        break

            assert (self.target_goals >= 0).all(), (
                f'Internal error: not all targets have been assigned with cargoes. '
                f'Got target_goals: {self.target_goals}.'
            )

        self.target_orientations.fill(0.0)
        for t, (goal, target) in enumerate(zip(self.target_goals, self.targets)):
            if goal >= 0:
                self.target_orientations[t] = arctan2_deg(
                    *reversed(consts.WAREHOUSES[goal] - target.location)
                )
            else:
                self.target_orientations[t] = normalize_angle(360.0 * self.np_random.random())

        self.camera_total_communication_edges.fill(0)
        self.target_total_communication_edges.fill(0)
        self.camera_communication_edges.fill(0)
        self.target_communication_edges.fill(0)
        self.camera_message_buffer.clear()
        self.target_message_buffer.clear()
        self.camera_message_queue.clear()
        self.target_message_queue.clear()

        self.episode_step = 0

        return self.joint_observation()

    def send_messages(self, messages: Union[Message, Iterable[Message]]) -> None:
        """Buffer the messages from an agent to others in the same team.

        The environment will send the messages to recipients' through method
        receive_messages(), and also info field of step() results.
        """

        if isinstance(messages, Message):
            messages = (messages,)

        messages = list(messages)
        assert (
            len({m.team for m in messages}) <= 1
        ), f'All messages must be from the same team. Got messages = {messages}.'

        for message in self.route_messages(messages):
            self.message_queues[message.team.value][message.recipient].append(message)
            self.message_buffers[message.team.value][message.recipient].append(message)
            self.communication_edges[message.team.value][message.sender, message.recipient] += 1

    def receive_messages(
        self, agent_id: Optional[Tuple[Team, int]] = None, agent: Optional['AgentType'] = None
    ) -> Union[Tuple[List[List[Message]], List[List[Message]]], List[Message]]:
        """Retrieve the messages to recipients. If no agent is specified, this
        method will return all the messages to all agents in the environment.

        The environment will also put the messages to recipients' info field of
        step() results.
        """

        if agent_id is None and agent is None:
            messages = (
                [list(self.camera_message_queue[c]) for c in range(self.num_cameras)],
                [list(self.target_message_queue[t]) for t in range(self.num_targets)],
            )

            self.camera_message_queue.clear()
            self.target_message_queue.clear()
        else:
            from mate.agents.base import AgentBase  # pylint: disable=import-outside-toplevel

            if isinstance(agent_id, AgentBase) and agent is None:
                agent_id, agent = agent, agent_id

            if agent is not None:
                assert agent_id is None, (
                    f'You should specify either `agent_id` or `agent`, not both.'
                    f'Got (agent_id, agent) = {(agent_id, agent)}.'
                )
                team, index = agent.TEAM, agent.index
            else:
                team, index = agent_id

            messages = list(self.message_queues[team.value][index])
            del self.message_queues[team.value][index]

        return messages

    def state(self) -> np.ndarray:
        """The global state of the environment."""

        if self._state is None:
            self._state = np.concatenate(
                [self.preserved_data]
                + [camera.state(private=True) for camera in self.cameras]
                + [target.state(private=True) for target in self.targets]
                + [obstacle.state() for obstacle in self.obstacles]
                + [self.freights, self.bounties, self.remaining_cargoes.ravel()]
            ).astype(np.float64)

        return self._state.copy()

    def joint_observation(self) -> Tuple[np.ndarray, np.ndarray]:  # pylint: disable=too-many-locals
        """Joint observations of both teams."""

        if self.num_cameras > 0:
            camera_states_public = np.vstack(list(map(Camera.state, self.cameras)))
        else:
            camera_states_public = np.zeros(
                (self.num_cameras, consts.CAMERA_STATE_DIM_PUBLIC), dtype=np.float64
            )
        camera_states_public_flagged = np.hstack(
            [camera_states_public, np.ones((self.num_cameras, 1), dtype=np.float64)]
        )

        target_states_public = np.vstack(list(map(Target.state, self.targets)))
        target_states_public_flagged = np.hstack(
            [target_states_public, np.ones((self.num_targets, 1), dtype=np.float64)]
        )

        if self.num_cameras > 0:
            camera_joint_observation = []
            for c, camera in enumerate(self.cameras):
                camera_observation = [self.preserved_data, camera.state(private=True)]
                target_mask = self.camera_target_view_mask[c, :, np.newaxis]
                camera_observation.append(
                    np.where(target_mask, target_states_public_flagged, 0.0).ravel()
                )
                camera_observation.append(self.camera_obstacle_observations[c])
                camera_mask = self.camera_camera_view_mask[c, :, np.newaxis]
                camera_observation.append(
                    np.where(camera_mask, camera_states_public_flagged, 0.0).ravel()
                )
                camera_joint_observation.append(np.concatenate(camera_observation))
            camera_joint_observation = np.vstack(camera_joint_observation)
            camera_joint_observation[:, 3] = np.arange(self.num_cameras, dtype=np.float64)
        else:
            camera_joint_observation = np.zeros(
                (self.num_cameras, self.camera_observation_dim), dtype=np.float64
            )

        target_joint_observation = []
        for t, target in enumerate(self.targets):
            target_observation = [self.preserved_data, target.state(private=True)]
            camera_mask = self.target_camera_view_mask[t, :, np.newaxis]
            target_observation.append(
                np.where(camera_mask, camera_states_public_flagged, 0.0).ravel()
            )
            obstacle_mask = self.target_obstacle_view_mask[t, :, np.newaxis]
            target_observation.append(
                np.where(obstacle_mask, self.obstacle_states_flagged, 0.0).ravel()
            )
            target_mask = self.target_target_view_mask[t, :, np.newaxis]
            target_observation.append(
                np.where(target_mask, target_states_public_flagged, 0.0).ravel()
            )
            target_joint_observation.append(np.concatenate(target_observation))
        target_joint_observation = np.vstack(target_joint_observation)
        target_joint_observation[:, 3] = np.arange(self.num_targets, dtype=np.float64)

        with_bounty_bits = self.bounties > 0
        num_with_bounty = with_bounty_bits.sum()
        self.coverage_rate = self.tracked_bits.sum() / self.num_targets
        if num_with_bounty > 0:
            self.real_coverage_rate = (self.tracked_bits * with_bounty_bits).sum() / num_with_bounty
        else:
            self.real_coverage_rate = 0.0

        if self.num_delivered_cargoes > 0:
            self.mean_transport_rate = self.delayed_target_team_episode_reward / (
                self.reward_scale * self.num_delivered_cargoes
            )
        else:
            self.mean_transport_rate = 0.0

        return camera_joint_observation.astype(np.float64), target_joint_observation.astype(
            np.float64
        )

    # pylint: disable-next=arguments-differ,too-many-locals,too-many-branches,too-many-statements
    def render(
        self,
        mode: str = 'human',
        window_size: int = DEFAULT_WINDOW_SIZE,
        onetime_callbacks: Iterable[Callable[['MultiAgentTracking', str], None]] = (),
    ) -> Union[bool, np.ndarray]:
        """Render the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and return nothing.
          Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        Parameters:
            mode (str): the mode to render with
            window_size (int): the width and height of the render window (only valid for the first call)
            onetime_callbacks (Iterable[callable]): callback functions for the rendering results
        """

        if mode not in self.metadata['render.modes']:
            return super().render(mode=mode)

        import mate.assets.pygletrendering as rendering  # pylint: disable=import-outside-toplevel

        if self.viewer is None:
            self.viewer = rendering.Viewer(window_size, window_size)
            bound = 1.05 * consts.TERRAIN_SIZE
            self.viewer.set_bounds(-bound, bound, -bound, bound)

            self.viewer.warehouse_images = {}
            for key in ((True, True), (True, False), (False, True), (False, False)):
                base = rendering.make_polygon(
                    consts.WAREHOUSE_RADIUS
                    * np.array([(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)])
                )
                image = rendering.Image(
                    ASSETS_DIR / 'images' / f'warehouse-{key[0]:d}{key[1]:d}.png',
                    1.8 * consts.WAREHOUSE_RADIUS,
                    1.8 * consts.WAREHOUSE_RADIUS,
                )
                self.viewer.warehouse_images[key] = image

        if len(self.viewer.geoms) == 0:
            margin = rendering.make_polygon(
                consts.TERRAIN_SIZE * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]), filled=False
            )
            margin.set_linewidth(3)
            self.viewer.add_geom(margin)

            self.viewer.warehouse = []
            for color, warehouse in zip(WAREHOUSE_COLORS, consts.WAREHOUSES):
                base = rendering.make_polygon(
                    consts.WAREHOUSE_RADIUS
                    * np.array([(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)])
                )
                image = rendering.Compound([base, self.viewer.warehouse_images[(True, True)]])
                base.attrs[:] = [base.color]
                base.set_color(*color)
                image.base = base
                image.transform = rendering.Transform(translation=warehouse)
                image.add_attr(image.transform)
                self.viewer.warehouse.append(image)
                self.viewer.add_geom(image)

            self.viewer.obstacles = []
            for obstacle in self.obstacles:
                image = rendering.make_circle(radius=obstacle.radius, res=72, filled=True)
                image.add_attr(rendering.Transform(translation=obstacle.location))
                image.set_color(*obstacle.COLOR)
                self.viewer.obstacles.append(image)
                self.viewer.add_geom(image)

            self.viewer.cameras = []
            for c, camera in enumerate(self.cameras):
                base = rendering.make_circle(radius=camera.radius, res=72, filled=True)
                body = rendering.make_polygon(
                    camera.radius * np.array([(0.8, 0.6), (-0.8, 0.6), (-0.8, -0.6), (0.8, -0.6)])
                )
                lens = rendering.make_polygon(
                    camera.radius * np.array([(0.7, 0.3), (1.2, 0.3), (1.2, -0.3), (0.7, -0.3)])
                )
                image = rendering.Compound([base, body, lens])
                for geom in image.gs:
                    geom.attrs[:] = [geom.color]
                body.set_color(1.0, 1.0, 1.0, 0.75)
                lens.set_color(0.1, 0.1, 0.1, 0.75)
                image.base = base
                image.transform = rendering.Transform(translation=camera.location)
                image.add_attr(image.transform)
                self.viewer.cameras.append(image)

            self.viewer.targets = []
            self.viewer.markers = []
            for capacity, target in zip(self.target_capacities, self.targets):
                if capacity == 1:
                    image = rendering.make_polygon(
                        TARGET_RENDER_RADIUS
                        * np.array(
                            [
                                (1.0, 0.0),
                                (-0.2, 0.6),
                                (-0.8, 0.6),
                                (-0.4, 0.0),
                                (-0.8, -0.6),
                                (-0.2, -0.6),
                            ]
                        )
                    )
                else:
                    image = rendering.make_polygon(
                        TARGET_RENDER_RADIUS
                        * np.array([(1.0, 0.0), (0.3, 0.6), (-0.8, 0.6), (-0.8, -0.6), (0.3, -0.6)])
                    )

                image.transform = rendering.Transform(translation=target.location)
                image.add_attr(image.transform)

                marker = rendering.make_circle(
                    radius=1.2 * TARGET_RENDER_RADIUS, res=15, filled=True
                )
                marker.transform = rendering.Transform(translation=target.location)
                marker.add_attr(marker.transform)
                marker.set_color(*target.COLOR_TRACKED)

                self.viewer.targets.append(image)
                self.viewer.markers.append(marker)

        remaining_cargo_counts = self.remaining_cargoes.sum(axis=-1)
        for w, color in enumerate(WAREHOUSE_COLORS):
            remaining, awaiting = (remaining_cargo_counts[w] > 0, self.awaiting_cargo_counts[w] > 0)

            warehouse = self.viewer.warehouse[w]
            warehouse.gs[-1] = self.viewer.warehouse_images[(remaining, awaiting)]
            warehouse.base.set_color(
                *warehouse.base.color.vec4[:3], (0.6 if remaining or awaiting else 0.3)
            )

        for c, camera in enumerate(self.cameras):
            phis, rhos = camera.boundary_between(
                camera.orientation - camera.viewing_angle / 2.0,
                camera.orientation + camera.viewing_angle / 2.0,
            )
            rhos = rhos.clip(min=camera.radius, max=camera.sight_range)
            vertices = polar2cartesian(rhos, phis).transpose()
            vertices = camera.location + np.concatenate([[[0.0, 0.0]], vertices, [[0.0, 0.0]]])
            boundary = polar2cartesian(camera.sight_range, phis).transpose()
            boundary = camera.location + np.concatenate([[[0.0, 0.0]], boundary, [[0.0, 0.0]]])

            polygon = rendering.make_polygon(vertices, filled=True)
            sector = rendering.make_polygon(boundary, filled=True)
            if self.camera_target_view_mask[c].any():
                polygon.set_color(0.0, 0.6, 0.0, 0.25)
            else:
                polygon.set_color(0.6, 0.6, 0.0, 0.25)
            sector.set_color(0.0, 0.6, 0.8, 0.1)
            self.viewer.add_onetime(sector)
            self.viewer.add_onetime(polygon)

        for c, (camera, image) in enumerate(zip(self.cameras, self.viewer.cameras)):
            perceived_by_targets = self.target_camera_view_mask[:, c].any()

            image.base.set_color(
                *(Camera.COLOR_PERCEIVED if perceived_by_targets else Camera.COLOR_UNPERCEIVED)
            )
            image.transform.set_rotation(np.deg2rad(camera.orientation))
            self.viewer.add_onetime(image)

        for t in np.flatnonzero(self.tracked_bits):
            marker = self.viewer.markers[t]
            marker.transform.set_translation(*self.targets[t].location)
            self.viewer.add_onetime(marker)

        for t, (goal, target, image) in enumerate(
            zip(self.target_goals, self.targets, self.viewer.targets)
        ):
            image.set_color(*(WAREHOUSE_COLORS[goal] if goal >= 0 else target.COLOR_NO_LOAD))
            image.transform.set_translation(*target.location)
            image.transform.set_rotation(np.deg2rad(self.target_orientations[t]))
            self.viewer.add_onetime(image)
            if goal >= 0 and self.bounties[t] == 0:
                new_image = copy.deepcopy(image)
                new_image.set_color(1.0, 1.0, 1.0, 0.66)
                new_image.transform.set_scale(0.4, 0.4)
                self.viewer.add_onetime(new_image)

        for callback in itertools.chain(self.render_callbacks.values(), onetime_callbacks):
            callback(self, mode)

        # pylint: disable-next=superfluous-parens
        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def add_render_callback(
        self, name: str, callback: Callable[['MultiAgentTracking', str], None]
    ) -> None:
        """Add a callback function to the render function.

        This is useful to add additional elements to the rendering results.
        """

        self.render_callbacks[name] = callback

    def close(self) -> None:
        """Perform necessary cleanup.

        Environments will automatically close() themselves when garbage
        collected or when the program exits.
        """

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for this environment's random number generators.

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: the list of seeds used in this environment's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """

        self._np_random, seed = seeding.np_random(seed)

        seeds, int_max = [seed], np.iinfo(int).max
        for entity in itertools.chain(
            self.cameras_ordered, self.targets_ordered, self.obstacles_ordered
        ):
            seeds.append(entity.seed(self.np_random.randint(int_max))[0])

        return seeds

    @property
    def np_random(self) -> np.random.RandomState:  # pylint: disable=no-member
        """The main random number generator of the environment."""

        if self._np_random is None:
            self.seed()
        return self._np_random

    def __str__(self) -> str:
        # pylint: disable-next=consider-using-f-string
        return '{}({} camera{}, {} target{}, {} obstacle{})'.format(
            super().__str__(),
            self.num_cameras,
            's' if self.num_cameras > 1 else '',
            self.num_targets,
            's' if self.num_targets > 1 else '',
            self.num_obstacles,
            's' if self.num_obstacles > 1 else '',
        )

    def route_messages(self, messages: List[Message]) -> List[Message]:
        """Convert broadcast messages to peer-to-peer forms."""

        processed_messages = []
        for message in messages:
            if message.recipient is None:  # broadcasting
                num_teammates = [self.num_cameras, self.num_targets][message.team.value]
                for recipient in range(num_teammates):
                    processed_messages.append(
                        Message(
                            sender=message.sender,
                            recipient=recipient,
                            content=copy.deepcopy(message.content),
                            team=message.team,
                            broadcasting=True,
                        )
                    )
            else:
                processed_messages.append(message)

        return processed_messages

    def _assign_goals(self) -> float:  # pylint: disable=too-many-locals
        old_target_goals = self.target_goals.copy()

        delayed_target_team_reward = 0.0
        target_team_reward = -float(np.logical_and(self.tracked_bits, self.bounties > 0).sum())
        self.bounties = np.maximum(self.bounties - self.tracked_bits, 0).astype(np.int64)

        for t, (goal, capacity, target) in enumerate(
            zip(self.target_goals, self.target_capacities, self.targets)
        ):
            directions = target.location - consts.WAREHOUSES
            self.target_warehouse_distances[t] = np.linalg.norm(directions, axis=-1)
            supremum = np.linalg.norm(directions, ord=np.inf, axis=-1)
            for warehouse in np.flatnonzero(supremum <= consts.WAREHOUSE_RADIUS):
                if goal >= 0:
                    if goal == warehouse:
                        cargo_weight = self.target_goal_bits[t, goal]
                        total_bounty = cargo_weight * self.bounty_scale
                        reward = self.freights[t] + self.bounties[t]
                        target_team_reward += reward
                        delayed_target_team_reward += reward - (total_bounty - self.bounties[t])
                        self.num_delivered_cargoes += cargo_weight
                        self.awaiting_cargo_counts[goal] -= cargo_weight
                    else:
                        continue
                self.freights[t] = self.bounties[t] = 0
                self.tracked_steps[t] = self.target_steps[t] = 0
                self.target_goal_bits[t].fill(0)
                target.goal_bits.fill(0)
                self.target_goals[t] = -1

                if self.remaining_cargoes[warehouse].any():
                    new_goal = self.np_random.choice(
                        np.flatnonzero(self.remaining_cargoes[warehouse] > 0)
                    )
                    remaining = self.remaining_cargoes[warehouse, new_goal]
                    cargo_weight = min(capacity, remaining)
                    self.remaining_cargoes[warehouse, new_goal] -= cargo_weight
                    self.target_goal_bits[t, new_goal] = cargo_weight
                    self.freights[t] = cargo_weight * self.freight_scale
                    self.bounties[t] = cargo_weight * self.bounty_scale

                    target.goal_bits[new_goal] = cargo_weight
                    self.target_goals[t] = new_goal
                    break

            for warehouse in np.flatnonzero(supremum <= consts.WAREHOUSE_RADIUS):
                target.empty_bits[warehouse] = not self.remaining_cargoes[warehouse].any()

        self.target_dones = np.logical_and(
            self.target_goals != old_target_goals, old_target_goals >= 0
        )

        return target_team_reward, delayed_target_team_reward

    def _simulate(self, action: Tuple[np.ndarray, np.ndarray]) -> None:
        camera_joint_action, target_joint_action = action

        camera_joint_action = np.asarray(camera_joint_action, dtype=np.float64)
        target_joint_action = np.asarray(target_joint_action, dtype=np.float64)
        camera_joint_action = camera_joint_action.reshape(
            self.num_cameras, consts.CAMERA_ACTION_DIM
        )
        target_joint_action = target_joint_action.reshape(
            self.num_targets, consts.TARGET_ACTION_DIM
        )
        assert np.isfinite(
            camera_joint_action
        ).all(), f'Got unexpected joint action {camera_joint_action}.'
        assert np.isfinite(
            target_joint_action
        ).all(), f'Got unexpected joint action {target_joint_action}.'

        for camera, camera_action in zip(self.cameras, camera_joint_action):
            camera.simulate(camera_action)
        for t, (target, target_action) in enumerate(zip(self.targets, target_joint_action)):
            previous_location = target.location.copy()
            target.simulate(target_action)
            if np.any(previous_location != target.location):
                self.target_orientations[t] = arctan2_deg(
                    *reversed(target.location - previous_location)
                )

        self._update_view()

    def _update_view(self) -> None:  # pylint: disable=too-many-branches
        self._state = None
        self.camera_target_view_mask.fill(False)
        self.target_camera_view_mask.fill(False)
        self.target_obstacle_view_mask.fill(False)
        self.camera_camera_view_mask.fill(False)
        self.target_target_view_mask.fill(False)

        for t, target in enumerate(self.targets):
            for c, camera in enumerate(self.cameras):
                if camera.perceive(target, transmittance=self.obstacle_transmittance):
                    self.camera_target_view_mask[c, t] = True
                if target.perceive(camera):
                    self.target_camera_view_mask[t, c] = True

            for o, obstacle in enumerate(self.obstacles):
                if target.perceive(obstacle):
                    self.target_obstacle_view_mask[t, o] = True

            for t_other, target_other in enumerate(self.targets):
                if t == t_other:
                    self.target_target_view_mask[t, t] = True
                elif target.perceive(target_other):
                    self.target_target_view_mask[t, t_other] = True

        for c, camera in enumerate(self.cameras):
            for c_other, camera_other in enumerate(self.cameras):
                if c == c_other:
                    self.camera_camera_view_mask[c, c] = True
                elif camera.perceive(camera_other):
                    self.camera_camera_view_mask[c, c_other] = True

        self.tracked_bits = self.camera_target_view_mask.any(axis=0)

    def _destroy(self) -> None:
        if self.viewer is not None:
            self.viewer.geoms.clear()
        self.camera_message_buffer.clear()
        self.target_message_buffer.clear()

    @property
    def name(self) -> str:
        """Name of the environment."""

        return self.config['name']

    @property
    def max_episode_steps(self) -> int:
        """Maximum number of episode steps."""

        return self.config['max_episode_steps']

    @property
    def camera_min_viewing_angle(self) -> float:
        """Minimum viewing angle of cameras **in degrees**."""

        return self.config['camera']['min_viewing_angle']

    @property
    def camera_max_sight_range(self) -> float:
        """Maximum sight range of cameras."""

        return self.config['camera']['max_sight_range']

    @property
    def camera_rotation_step(self) -> float:
        """Maximum rotation step of cameras **in degrees**."""

        return self.config['camera']['rotation_step']

    @property
    def camera_zooming_step(self) -> float:
        """Maximum zooming step of cameras **in degrees**."""

        return self.config['camera']['zooming_step']

    @property
    def target_step_size(self) -> float:
        """Maximum step size of targets."""

        if self._target_step_size is None:
            self._target_step_size = self.config['target']['step_size']
        return self._target_step_size

    @property
    def target_sight_range(self) -> float:
        """Sight range of targets."""

        return self.config['target']['sight_range']

    @property
    def num_cargoes_per_target(self) -> int:
        """Average number of cargoes per target."""

        return self.config['num_cargoes_per_target']

    @property
    def targets_start_with_cargoes(self) -> bool:
        """Always assign cargoes to the target at the beginning of an episode."""

        if self._targets_start_with_cargoes is None:
            self._targets_start_with_cargoes = self.config.get('targets_start_with_cargoes', True)
        return self._targets_start_with_cargoes

    @property
    def bounty_factor(self) -> float:
        """The ratio of the maximum bounty reward over the freight reward."""

        if self._bounty_factor is None:
            bounty_factor = self.config.get('bounty_factor', 1.0)
            self._bounty_factor = max(0.0, bounty_factor)
        return self._bounty_factor

    @property
    def obstacle_transmittance(self) -> float:
        """Transmittance coefficient of obstacles."""

        if self._obstacle_transmittance is None:
            transmittance = self.config.get('obstacle', {}).get('transmittance', 0.0)
            self._obstacle_transmittance = min(max(0.0, transmittance), 1.0)
        return self._obstacle_transmittance

    @property
    def shuffle_entities(self) -> bool:
        """Whether or not to shuffle entity IDs when reset the environment."""

        if self._shuffle_entities is None:
            self._shuffle_entities = self.config.get('shuffle_entities', True)
        return self._shuffle_entities

    @property
    def num_warehouses(self) -> int:
        """Number of warehouses."""

        return consts.NUM_WAREHOUSES

    @property
    def num_cameras(self) -> int:
        """Number of camera(s) in the environment."""

        if self._num_cameras is None:
            self._num_cameras = len(self.cameras)
        return self._num_cameras

    @property
    def num_targets(self) -> int:
        """Number of target(s) in the environment."""

        if self._num_targets is None:
            self._num_targets = len(self.targets)
        return self._num_targets

    @property
    def num_obstacles(self) -> int:
        """Number of obstacle(s) in the environment."""

        if self._num_obstacles is None:
            self._num_obstacles = len(self.obstacles)
        return self._num_obstacles

    @property
    def high_capacity_target_split(self) -> float:
        """Population ratio of the high-capacity target in the target team."""

        if self._high_capacity_target_split is None:
            split_ratio = self.config.get('high_capacity_target_split', 0.5)
            self._high_capacity_target_split = min(max(0.0, split_ratio), 1.0)

        return self._high_capacity_target_split

    @property
    def num_high_capacity_targets(self) -> float:
        """Number of high-capacity target(s) in the target team."""

        if self._num_high_capacity_targets is None:
            self._num_high_capacity_targets = int(
                self.num_targets * self.high_capacity_target_split
            )

        return self._num_high_capacity_targets

    @property
    def num_low_capacity_targets(self) -> float:
        """Number of low-capacity target(s) in the target team."""

        if self._num_low_capacity_targets is None:
            self._num_low_capacity_targets = self.num_targets - self.num_high_capacity_targets

        return self._num_low_capacity_targets

    @property
    def camera_observation_dim(self) -> int:
        """Dimension of single camera observation."""

        if self._camera_observation_dim is None:
            self._camera_observation_dim = self.camera_observation_space.shape[-1]
        return self._camera_observation_dim

    @property
    def target_observation_dim(self) -> int:
        """Dimension of single target observation."""

        if self._target_observation_dim is None:
            self._target_observation_dim = self.target_observation_space.shape[-1]
        return self._target_observation_dim
