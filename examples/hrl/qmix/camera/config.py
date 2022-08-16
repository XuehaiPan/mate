import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.qmix import qmix
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hrl.wrappers import DiscreteMultiSelection, HierarchicalCamera
from examples.utils import (
    CustomMetricCallback,
    RLlibMultiAgentAPI,
    RLlibMultiAgentCentralizedTraining,
)


def target_agent_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    base_env = mate.make(
        env_id, config=env_config.get('config'), **env_config.get('config_overrides', {})
    )
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])

    target_agent = env_config.get('opponent_agent_factory', target_agent_factory)()
    env = mate.MultiCamera(base_env, target_agent=target_agent)

    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)

    if 'reward_coefficients' in env_config:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=env_config['reward_coefficients'],
            reduction=env_config.get('reward_reduction', 'none'),
        )

    multi_selection = env_config.get('multi_selection', False)
    env = HierarchicalCamera(
        env, multi_selection=multi_selection, frame_skip=env_config.get('frame_skip', 1)
    )
    if multi_selection:
        env = DiscreteMultiSelection(env)

    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))

    env = env.with_agent_groups(
        groups={'camera': env.agent_ids}, obs_space=observation_space, act_space=action_space
    )
    return env


tune.register_env('mate-hrl.qmix.camera', make_env)

config = {
    **qmix.DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-hrl.qmix.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},  # override env's raw reward
        'reward_reduction': 'mean',  # shared reward
        'multi_selection': True,
        'frame_skip': 5,
        'enhanced_observation': 'none',
        'opponent_agent_factory': target_agent_factory,
    },
    'disable_env_checking': True,
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    # === Model ====================================================================================
    'normalize_actions': True,
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [512, 256],  # not used
        'fcnet_activation': 'tanh',
        'lstm_cell_size': 256,
        'max_seq_len': 10000,  # for complete episode
    },
    'mixer': 'qmix',
    'mixing_embed_dim': 128,
    # === Policy ===================================================================================
    'gamma': 0.99,
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.0,
        'final_epsilon': 0.02,
        'epsilon_timesteps': 50000,  # trained environment steps
    },
    # === Replay Buffer & Optimization =============================================================
    'batch_mode': 'complete_episodes',
    'rollout_fragment_length': 0,  # send sampled episodes to the replay buffer immediately
    'buffer_size': 2000,  # each item contains `num_workers` episodes (will be updated in train.py)
    'timesteps_per_iteration': 5120,  # environment steps
    'learning_starts': 5000,  # environment steps
    'train_batch_size': 1024,  # environment steps
    'target_network_update_freq': 500,  # environment steps
    'metrics_num_episodes_for_smoothing': 25,
    'grad_norm_clipping': 1000.0,
    'lr': 1e-4,
}
