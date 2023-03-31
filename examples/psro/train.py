#!/usr/bin/env python3

# Run: python3 -m examples.psro.train

import argparse
import copy
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import ray
import torch

import mate
from examples.hrl.mappo.camera import CameraAgent
from examples.hrl.mappo.camera.train import experiment as camera_base_experiment
from examples.hrl.mappo.camera.train import train as camera_train
from examples.mappo.target import TargetAgent
from examples.mappo.target.train import experiment as target_base_experiment
from examples.mappo.target.train import train as target_train
from examples.psro.evaluator import calculate_exploitability, evaluate
from examples.psro.meta_solvers import META_SOLVERS
from examples.psro.trainer import PlayerTrainer


camera_base_experiment = copy.deepcopy(camera_base_experiment)
camera_base_experiment.name = f'camera.{camera_base_experiment.name}'
camera_base_experiment.spec['config']['env_config']['config'] = 'MATE-2v4-0.yaml'
camera_base_experiment.spec['config']['env_config']['config_overrides'] = {'reward_type': 'dense'}
camera_base_experiment.spec['config']['env_config']['reward_coefficients'] = {
    'coverage_rate': 0.5,
    'real_coverage_rate': 0.5,
    'raw_reward': 0.1,
}
camera_base_experiment.spec['config']['env_config']['reward_reduction'] = 'mean'
camera_base_experiment.spec['config']['env_config']['enhanced_observation'] = 'target'

target_base_experiment = copy.deepcopy(target_base_experiment)
target_base_experiment.name = f'target.{target_base_experiment.name}'
target_base_experiment.spec['config']['env_config']['config'] = 'MATE-2v4-0.yaml'
target_base_experiment.spec['config']['env_config']['config_overrides'] = {'reward_type': 'dense'}
target_base_experiment.spec['config']['env_config'].pop('reward_coefficients', None)
target_base_experiment.spec['config']['env_config'].pop('reward_reduction', None)
target_base_experiment.spec['config']['env_config']['enhanced_observation'] = 'target'


BOOTSTRAP_CAMERA_AGENT_CLASSES = [mate.RandomCameraAgent]
BOOTSTRAP_TARGET_AGENT_CLASSES = [mate.RandomTargetAgent]


DEBUG = getattr(sys, 'gettrace', lambda: None)() is not None

HERE = Path(__file__).absolute().parent
LOCAL_DIR = HERE / 'ray_results'
if DEBUG:
    print(f'DEBUG MODE: {DEBUG}')
    LOCAL_DIR = LOCAL_DIR / 'debug'


# Node resources
SLURM_CPUS_ON_NODE = int(os.getenv('SLURM_CPUS_ON_NODE', str(os.cpu_count())))
NUM_NODE_CPUS = max(1, min(os.cpu_count(), SLURM_CPUS_ON_NODE))
assert NUM_NODE_CPUS >= 3
NUM_NODE_GPUS = torch.cuda.device_count()

# Training resources
PRESERVED_NUM_CPUS = 1  # for raylet
NUM_CPUS_FOR_TRAINER = 1
NUM_GPUS_FOR_TRAINER = min(NUM_NODE_GPUS / 2.0, 0.25)  # can be overridden by command line arguments

MAX_NUM_CPUS_FOR_WORKER = max(0, (NUM_NODE_CPUS - PRESERVED_NUM_CPUS) // 2 - NUM_CPUS_FOR_TRAINER)
MAX_NUM_WORKERS = min(32, MAX_NUM_CPUS_FOR_WORKER)  # use at most 32 workers
NUM_WORKERS = MAX_NUM_WORKERS if not DEBUG else 0  # can be overridden by command line arguments


def camera_agent_factory(agent_specs, weights, seed=0):
    assert len(agent_specs) == len(weights)

    candidates = [agent_spec() for agent_spec in agent_specs]
    return mate.MixtureCameraAgent(
        candidates=candidates, weights=weights, mixture_seed=seed, seed=seed
    )


def target_agent_factory(agent_specs, weights, seed=0):
    assert len(agent_specs) == len(weights)

    candidates = [agent_spec() for agent_spec in agent_specs]
    return mate.MixtureTargetAgent(
        candidates=candidates, weights=weights, mixture_seed=seed, seed=seed
    )


def train(
    iterations,
    project=None,
    group=None,
    local_dir=LOCAL_DIR,
    num_gpus=NUM_GPUS_FOR_TRAINER,
    num_workers=NUM_WORKERS,
    num_envs_per_worker=8,
    timesteps_total=5e6,
    meta_solver='NashEquilibrium',
    num_evaluation_episodes=100,
    seed=0,
):
    if not ray.is_initialized():
        ray.init(num_cpus=NUM_NODE_CPUS, local_mode=DEBUG)
    num_ray_cpus = round(ray.cluster_resources()['CPU'])
    num_ray_gpus = ray.cluster_resources().get('GPU', 0.0)
    num_workers = max(0, min(num_workers, num_ray_cpus // 2 - NUM_CPUS_FOR_TRAINER))
    num_gpus = min(num_gpus, num_ray_gpus / 2.0)

    project = project or 'mate-psro'
    if isinstance(meta_solver, str):
        meta_solver = META_SOLVERS[meta_solver]
    group = (
        group
        or f'{meta_solver.ABBREVIATED_NAME}-{camera_base_experiment.name}-vs.-{target_base_experiment.name}'
    )

    evaluation_env_config = target_base_experiment.spec['config']['env_config']

    camera_agent_specs = [
        partial(agent_class, seed=seed) for agent_class in BOOTSTRAP_CAMERA_AGENT_CLASSES
    ]
    target_agent_specs = [
        partial(agent_class, seed=seed) for agent_class in BOOTSTRAP_TARGET_AGENT_CLASSES
    ]
    camera_agent_pool = [agent_spec() for agent_spec in camera_agent_specs]
    target_agent_pool = [agent_spec() for agent_spec in target_agent_specs]
    payoff_matrices = np.full((2, 0, 0), fill_value=np.nan, dtype=np.float64)
    coverage_rate_matrix = np.full((0, 0), fill_value=np.nan, dtype=np.float64)
    exploitabilities = np.zeros((0, 3), dtype=np.float64)

    local_dir = Path(local_dir)
    camera_local_dir = target_local_dir = None
    last_camera_checkpoint_path = last_target_checkpoint_path = None
    for i in range(0, iterations + 1):
        prev_camera_local_dir = camera_local_dir
        prev_target_local_dir = target_local_dir
        camera_local_dir = local_dir / group / 'camera' / f'{i:05d}'
        target_local_dir = local_dir / group / 'target' / f'{i:05d}'
        camera_local_dir.mkdir(parents=True, exist_ok=True)
        target_local_dir.mkdir(parents=True, exist_ok=True)

        if i > 0:
            camera_trainer = PlayerTrainer.remote(
                iteration=i,
                player='camera',
                train_fn=camera_train,
                base_experiment=camera_base_experiment,
                opponent_agent_factory=partial(
                    target_agent_factory,
                    agent_specs=list(target_agent_specs),
                    weights=sigma_target,
                    seed=seed,
                ),
                from_checkpoint=last_camera_checkpoint_path,
                timesteps_total=timesteps_total,
                local_dir=camera_local_dir,
                project=project,
                group=group,
            )

            target_trainer = PlayerTrainer.remote(
                iteration=i,
                player='target',
                train_fn=target_train,
                base_experiment=target_base_experiment,
                opponent_agent_factory=partial(
                    camera_agent_factory,
                    agent_specs=list(camera_agent_specs),
                    weights=sigma_camera,
                    seed=seed,
                ),
                from_checkpoint=last_target_checkpoint_path,
                timesteps_total=timesteps_total,
                local_dir=target_local_dir,
                project=project,
                group=group,
            )

            player_mapping = {
                'camera': (
                    camera_trainer,
                    CameraAgent,
                    camera_agent_specs,
                    camera_agent_pool,
                    camera_local_dir,
                ),
                'target': (
                    target_trainer,
                    TargetAgent,
                    target_agent_specs,
                    target_agent_pool,
                    target_local_dir,
                ),
            }

            train_kwargs = {
                'num_workers': num_workers,
                'num_gpus': num_gpus,
                'num_envs_per_worker': num_envs_per_worker,
                'seed': seed,
            }
            not_ready = [
                camera_trainer.result.remote(skip_train_if_exists=True, **train_kwargs),
                target_trainer.result.remote(skip_train_if_exists=True, **train_kwargs),
            ]
            while len(not_ready) > 0:
                ready, not_ready = ray.wait(not_ready, timeout=10)

                if len(ready) > 0:
                    for player, best_checkpoint_path in ray.get(ready):
                        print(f'Player {player}({i:05d}) ready.')
                        (
                            trainer,
                            agent_class,
                            agent_specs,
                            agent_pool,
                            experiment_local_dir,
                        ) = player_mapping[player]
                        ray.kill(trainer)

                        new_agent_spec = partial(agent_class, checkpoint_path=best_checkpoint_path)
                        new_agent = new_agent_spec()
                        agent_specs.append(new_agent_spec)
                        agent_pool.append(new_agent)
                        (experiment_local_dir / 'checkpoint_path.txt').write_text(
                            str(best_checkpoint_path), encoding='UTF-8'
                        )

                    payoff_matrices, coverage_rate_matrix, *_ = evaluate(
                        payoff_matrices,
                        coverage_rate_matrix,
                        camera_agent_pool,
                        target_agent_pool,
                        env_config=evaluation_env_config,
                        deterministic=None,
                        num_episodes=num_evaluation_episodes,
                        horizon=+np.inf,
                    )

            last_camera_checkpoint_path = camera_agent_pool[-1].checkpoint_path
            last_target_checkpoint_path = target_agent_pool[-1].checkpoint_path
            exploitabilities = np.append(
                exploitabilities,
                [calculate_exploitability(payoff_matrices, sigma_camera, sigma_target)],
                axis=0,
            )
        else:
            payoff_matrices, coverage_rate_matrix, *_ = evaluate(
                payoff_matrices,
                coverage_rate_matrix,
                camera_agent_pool,
                target_agent_pool,
                env_config=evaluation_env_config,
                deterministic=None,
                num_episodes=num_evaluation_episodes,
                horizon=+np.inf,
            )

        payoff_camera, payoff_target = payoff_matrices
        print(f'payoff_matrices: {payoff_matrices}')
        np.savetxt(camera_local_dir / 'payoff.txt', payoff_camera, fmt='%+.18E')
        np.savetxt(target_local_dir / 'payoff.txt', payoff_target, fmt='%+.18E')
        np.savetxt(camera_local_dir / 'coverage_rate.txt', coverage_rate_matrix, fmt='%.18f')

        sigma_camera, sigma_target = meta_solver(payoff_matrices).solve()
        print(f'sigma_camera: {sigma_camera}')
        print(f'sigma_target: {sigma_target}')
        np.savetxt(camera_local_dir / 'sigma.txt', sigma_camera, fmt='%.18f')
        np.savetxt(target_local_dir / 'sigma.txt', sigma_target, fmt='%.18f')

        print(f'exploitabilities: {exploitabilities}')
        if prev_camera_local_dir is not None:
            np.savetxt(prev_camera_local_dir / 'exploitability.txt', exploitabilities, fmt='%+.18E')
        if prev_target_local_dir is not None:
            np.savetxt(prev_target_local_dir / 'exploitability.txt', exploitabilities, fmt='%+.18E')


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--iterations',
        type=int,
        metavar='ITER',
        default=40,
        help='total number of iterations (default: %(default)d)',
    )
    parser.add_argument(
        '--project', type=str, metavar='PROJECT', default=None, help='W&B project name'
    )
    parser.add_argument('--group', type=str, metavar='GROUP', default=None, help='W&B group name')
    parser.add_argument(
        '--local-dir',
        type=str,
        metavar='DIR',
        default=LOCAL_DIR,
        help='Local directory for the experiment (default: %(default)s)',
    )
    parser.add_argument(
        '--num-gpus',
        type=float,
        metavar='GPU',
        default=NUM_GPUS_FOR_TRAINER,
        help='number of GPUs for trainer for each player (default: %(default)g)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        metavar='WORKER',
        default=NUM_WORKERS,
        help='number of rollout workers for each player (default: %(default)d)',
    )
    parser.add_argument(
        '--num-envs-per-worker',
        type=int,
        metavar='ENV',
        default=8,
        help='number of environments per rollout worker (default: %(default)d)',
    )
    parser.add_argument(
        '--timesteps-total',
        type=float,
        metavar='STEP',
        default=5e6,
        help='number of environment steps for each iteration (default: %(default).1e)',
    )
    parser.add_argument(
        '--meta-solver',
        type=str,
        metavar='SOLVER',
        default='NashEquilibrium',
        choices=list(META_SOLVERS.keys()),
        help='meta-strategy solver for the meta-game (default: %(default)s)',
    )
    parser.add_argument(
        '--num-evaluation-episodes',
        type=int,
        metavar='EPISODE',
        default=100,
        help='number of episodes to evaluate for each payoff entry (default: %(default)d)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', default=0, help='the global seed (default: %(default)d)'
    )

    args = parser.parse_args()

    train(**vars(args))


if __name__ == '__main__':
    main()
