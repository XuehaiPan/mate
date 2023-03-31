import random
import threading
from collections import defaultdict

import nashpy as nash
import numpy as np
import ray
import torch
import tqdm
from setproctitle import setproctitle

import mate


@ray.remote(max_restarts=3)
class Evaluator:
    def __init__(
        self,
        entry,
        camera_agent,
        target_agent,
        env_config,
        deterministic=None,
        seed=0,
        horizon=+np.inf,
    ):
        self.entry = entry
        self.camera_agent = camera_agent.clone()
        self.target_agent = target_agent.clone()
        self.env_config = env_config

        self.deterministic = deterministic
        self.seed = seed
        self.horizon = horizon
        self.identifier = (self.entry, self.seed)

    def make_evaluation_env(self, seed=None):
        env_id = self.env_config.get('env_id', 'MultiAgentTracking-v0')
        base_env = mate.make(
            env_id,
            config=self.env_config.get('config'),
            **self.env_config.get('config_overrides', {}),
        )

        enhance_team = str(self.env_config.get('enhanced_observation', None)).lower()
        if enhance_team is not None:
            base_env = mate.EnhancedObservation(base_env, team=enhance_team)

        base_env.seed(seed=seed)
        return base_env

    def set_seed(self, seed=None):
        if seed is not None:
            self.seed = seed

        mate.seed_everything(self.seed)

    def evaluate(self):
        name = f'Evaluator(entry={self.entry}, seed={self.seed})'
        setproctitle(f'ray::{name}.evaluate()')
        print(f'{name}: {self.camera_agent} vs. {self.target_agent}')

        self.set_seed()

        env = self.make_evaluation_env(seed=self.seed)

        self.camera_agent.seed(seed=env.np_random.randint(np.iinfo(int).max))
        self.target_agent.seed(seed=env.np_random.randint(np.iinfo(int).max))
        camera_agents = self.camera_agent.spawn(env.num_cameras)
        target_agents = self.target_agent.spawn(env.num_targets)

        camera_team_episode_reward, target_team_episode_reward = 0.0, 0.0
        coverage_rates = []

        camera_joint_observation, target_joint_observation = env.reset()
        camera_infos, target_infos = None, None
        mate.group_reset(camera_agents, camera_joint_observation)
        mate.group_reset(target_agents, target_joint_observation)

        while env.episode_step < self.horizon:
            camera_joint_action = mate.group_step(
                env, camera_agents, camera_joint_observation, camera_infos, self.deterministic
            )
            target_joint_action = mate.group_step(
                env, target_agents, target_joint_observation, target_infos, self.deterministic
            )
            (
                (camera_joint_observation, target_joint_observation),
                (camera_team_reward, target_team_reward),
                done,
                (camera_infos, target_infos),
            ) = env.step((camera_joint_action, target_joint_action))

            camera_team_episode_reward += camera_team_reward
            target_team_episode_reward += target_team_reward
            coverage_rates.append(env.coverage_rate)

            if done:
                break

        payoff_camera = camera_team_episode_reward / env.max_target_team_episode_reward
        payoff_target = target_team_episode_reward / env.max_target_team_episode_reward
        mean_coverage_rate = np.mean(coverage_rates)

        return self.identifier, (self.entry, (payoff_camera, payoff_target), mean_coverage_rate)


def as_remote_ref(getitem):
    cache = {}
    lock = threading.RLock()

    def wrapped(key):
        with lock:
            try:
                return cache[key]
            except KeyError:
                remote_ref = ray.put(getitem(key))
                cache[key] = remote_ref
                return remote_ref

    wrapped.cache = cache
    wrapped.lock = lock
    return wrapped


def expand_size(array, shape, fill_value=np.nan, dtype=np.float64):
    if array is None:
        return np.full(shape, fill_value=fill_value, dtype=dtype)

    old_array = np.asarray(array, dtype=np.float64)
    assert len(shape) == old_array.ndim and shape >= old_array.shape

    array = np.full(shape, fill_value=fill_value, dtype=dtype)
    array[tuple(slice(None, s) for s in old_array.shape)] = old_array

    return array


def evaluate(
    payoff_matrices,
    coverage_rate_matrix,
    camera_agent_pool,
    target_agent_pool,
    env_config,
    deterministic=None,
    num_workers=None,
    num_episodes=100,
    horizon=+np.inf,
):
    m, n = len(camera_agent_pool), len(target_agent_pool)
    payoff_matrices = expand_size(
        payoff_matrices, shape=(2, m, n), fill_value=np.nan, dtype=np.float64
    )
    coverage_rate_matrix = expand_size(
        coverage_rate_matrix, shape=(m, n), fill_value=np.nan, dtype=np.float64
    )

    if num_workers is None:
        num_workers = round(ray.cluster_resources()['CPU'])

    camera_agent_remote_get = as_remote_ref(camera_agent_pool.__getitem__)
    target_agent_remote_get = as_remote_ref(target_agent_pool.__getitem__)

    pending = {}
    for c, t in np.argwhere(np.isnan(payoff_matrices).any(axis=0)):
        for e in range(num_episodes):
            pending[((c, t), e)] = {
                'entry': (c, t),
                'camera_agent': camera_agent_remote_get(c),
                'target_agent': target_agent_remote_get(t),
                'env_config': env_config,
                'deterministic': deterministic,
                'seed': e,
                'horizon': horizon,
            }

    not_ready = []
    evaluators = {}
    results = []
    with tqdm.tqdm(
        desc=f'Update payoffs{(len(camera_agent_pool), len(target_agent_pool))}',
        total=len(pending),
        unit='episode',
    ) as pbar:
        while len(not_ready) > 0 or len(pending) > 0:
            while len(not_ready) < num_workers and len(pending) > 0:
                identifier = next(iter(pending))
                kwargs = pending.pop(identifier)
                evaluator = Evaluator.remote(**kwargs)
                evaluators[identifier] = evaluator
                not_ready.append(evaluator.evaluate.remote())

            ready, not_ready = ray.wait(not_ready, timeout=1)
            for identifier, result in ray.get(ready):
                results.append(result)
                evaluator = evaluators.pop(identifier)
                ray.kill(evaluator)

            pbar.update(len(ready))

    payoff_results = defaultdict(list)
    coverage_rate_results = defaultdict(list)
    for entry, payoff, mean_coverage_rate in results:
        payoff_results[entry].append(payoff)
        coverage_rate_results[entry].append(mean_coverage_rate)

    for (c, t), payoffs in payoff_results.items():
        payoff_matrices[:, c, t] = np.mean(payoffs, axis=0)
    for (c, t), mean_coverage_rates in coverage_rate_results.items():
        coverage_rate_matrix[c, t] = np.mean(mean_coverage_rates)

    return payoff_matrices, coverage_rate_matrix, payoff_results, coverage_rate_results


def calculate_exploitability(payoff_matrices, sigma_row, sigma_col):
    payoff_matrices = np.asarray(payoff_matrices)
    m, n = len(sigma_row), len(sigma_col)
    assert payoff_matrices.shape[1] > m and payoff_matrices.shape[2] > n

    subgame = nash.Game(*payoff_matrices[:, :m, :n])

    payoff_row, payoff_col = subgame[sigma_row, sigma_col]
    payoff_row_br = np.dot(payoff_matrices[0, : m + 1, :n], sigma_col).max()
    payoff_col_br = np.dot(payoff_matrices[1, :m, : n + 1].T, sigma_row).max()

    exploitability_row = payoff_row_br - payoff_row
    exploitability_col = payoff_col_br - payoff_col
    exploitability_all = (exploitability_row + exploitability_col) / 2.0

    return exploitability_all, exploitability_row, exploitability_col
