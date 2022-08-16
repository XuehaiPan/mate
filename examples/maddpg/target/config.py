from ray import tune
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

import mate
from examples.maddpg.models import MADDPGModel
from examples.utils import (
    CustomMetricCallback,
    FrameSkip,
    RLlibMultiAgentAPI,
    RLlibMultiAgentCentralizedTraining,
    RLlibMultiCallbacks,
    ShiftAgentActionTimestep,
    independent_policy_mapping_fn,
)


def camera_agent_factory():
    return mate.agents.GreedyCameraAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    base_env = mate.make(
        env_id, config=env_config.get('config'), **env_config.get('config_overrides', {})
    )
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])

    discrete_levels = env_config.get('discrete_levels', None)
    assert discrete_levels is None, 'DDPG/TD3 only supports continuous actions.'

    camera_agent = env_config.get('opponent_agent_factory', camera_agent_factory)()
    env = mate.MultiTarget(base_env, camera_agent=camera_agent)

    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)

    if 'reward_coefficients' in env_config:
        env = mate.AuxiliaryTargetRewards(
            env,
            coefficients=env_config['reward_coefficients'],
            reduction=env_config.get('reward_reduction', 'none'),
        )

    frame_skip = env_config.get('frame_skip', 1)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip=frame_skip)

    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    return env


tune.register_env('mate-maddpg.target', make_env)

config = {
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-maddpg.target',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-2v4-0.yaml',
        'config_overrides': {'reward_type': 'dense', 'shuffle_entities': False},
        'frame_skip': 10,
        'enhanced_observation': 'none',
        'opponent_agent_factory': camera_agent_factory,
    },
    'horizon': 500,
    'callbacks': RLlibMultiCallbacks([CustomMetricCallback, ShiftAgentActionTimestep]),
    # === Model ====================================================================================
    'normalize_actions': False,  # required by the model and exploration
    'model': {
        'max_seq_len': 25,
        'custom_model': MADDPGModel,
        'custom_model_config': {
            **MODEL_DEFAULTS,
            'actor_hiddens': [512, 256],
            'actor_hidden_activation': 'tanh',
            'critic_hiddens': [512, 256],
            'critic_hidden_activation': 'tanh',
            'max_seq_len': 25,
            'vf_share_layers': False,
        },
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    'twin_q': True,
    'policy_delay': 2,
    'smooth_target_policy': True,
    'target_noise': 0.2,
    'target_noise_clip': 0.5,
    'n_step': 1,
    'multiagent': {},  # independent policies defined in below
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'GaussianNoise',
        'random_timesteps': 10000,  # trained environment steps
        'stddev': 0.1,
        'initial_scale': 1.0,
        'final_scale': 1.0,
        'scale_timesteps': 1,  # do not anneal over time (fixed 1.0)
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
    'rollout_fragment_length': 25,
    'train_batch_size': 1024,
    'metrics_num_episodes_for_smoothing': 25,
    'actor_lr': 1e-4,
    'critic_lr': 1e-4,
    'tau': 0.01,
    'target_network_update_freq': 0,
    'use_huber': True,
    'huber_threshold': 10.0,
    'l2_reg': 0.0,
}

# Independent policy for each agent (no parameter sharing)
_dummy_env = make_env(config['env_config'])
config['multiagent'].update(
    policies={
        agent_id: PolicySpec(observation_space=None, action_space=None, config=None)
        for agent_id in _dummy_env.agent_ids
    },
    policy_mapping_fn=independent_policy_mapping_fn,
)
del _dummy_env
