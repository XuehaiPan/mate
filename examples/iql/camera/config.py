from ray import tune
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

import mate
from examples.utils import (
    SHARED_POLICY_ID,
    CustomMetricCallback,
    FrameSkip,
    RLlibMultiAgentAPI,
    shared_policy_mapping_fn,
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

    discrete_levels = env_config.get('discrete_levels', None)
    assert discrete_levels is not None, 'DQN only supports discrete actions.'
    base_env = mate.DiscreteCamera(base_env, levels=discrete_levels)

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

    frame_skip = env_config.get('frame_skip', 1)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip=frame_skip)

    env = RLlibMultiAgentAPI(env)
    return env


tune.register_env('mate-iql.camera', make_env)

config = {
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-iql.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},  # override env's raw reward
        'reward_reduction': 'mean',  # shared reward
        'discrete_levels': 5,
        'frame_skip': 5,
        'enhanced_observation': 'none',
        'opponent_agent_factory': target_agent_factory,
    },
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    # === Model ====================================================================================
    'normalize_actions': True,
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [512, 512, 256],
        'fcnet_activation': 'tanh',
        'max_seq_len': 25,
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    'dueling': True,
    'double_q': True,
    'n_step': 1,
    'multiagent': {
        'policies': {
            SHARED_POLICY_ID: PolicySpec(observation_space=None, action_space=None, config=None)
        },
        'policy_mapping_fn': shared_policy_mapping_fn,
    },
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.0,
        'final_epsilon': 0.02,
        'epsilon_timesteps': 50000,  # trained environment steps
    },
    # === Replay Buffer & Optimization =============================================================
    'batch_mode': 'truncate_episodes',
    'prioritized_replay': True,
    'replay_buffer_config': {
        'type': 'MultiAgentReplayBuffer',
        'capacity': 500000,
    },
    'timesteps_per_iteration': 5120,
    'learning_starts': 5000,
    'rollout_fragment_length': 1,
    'train_batch_size': 1024,
    'target_network_update_freq': 500,
    'metrics_num_episodes_for_smoothing': 25,
    'grad_clip': None,
    'lr': 5e-4,
    'lr_schedule': [
        (0, 5e-4),
        (4e6, 5e-4),
        (4e6, 1e-4),
        (8e6, 1e-4),
        (8e6, 5e-5),
    ],
}
