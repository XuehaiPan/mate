import copy
import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import ray
from ray import tune
from setproctitle import setproctitle

from examples.utils import RLlibMultiCallbacks, SymlinkCheckpointCallback, TrainFromCheckpoint


@ray.remote(max_restarts=1)
class PlayerTrainer:
    def __init__(
        self,
        iteration,
        player,
        train_fn,
        base_experiment,
        opponent_agent_factory,
        from_checkpoint,
        timesteps_total,
        local_dir,
        project=None,
        group=None,
        **kwargs,
    ):
        self.iteration = iteration
        self.player = player
        self.name = f'PSRO-{base_experiment.name}'

        run = base_experiment.spec['run']
        self.config = copy.deepcopy(base_experiment.spec['config'])
        self.config['env_config'].update(
            player=player,
            iteration=iteration,
            opponent_agent_factory=opponent_agent_factory,
        )
        self.config.update(
            lr=5e-4,
            lr_schedule=[
                (0, 5e-4),
                (0.4 * timesteps_total, 5e-4),
                (0.4 * timesteps_total, 1e-4),
                (0.8 * timesteps_total, 1e-4),
                (0.8 * timesteps_total, 5e-5),
            ],
            entropy_coeff=0.05,
            entropy_coeff_schedule=[
                (0, 0.05),
                (0.2 * timesteps_total, 0.01),
                (0.4 * timesteps_total, 0.001),
                (timesteps_total, 0.0),
            ],
        )
        if from_checkpoint is not None:
            self.config['callbacks'] = RLlibMultiCallbacks(
                [
                    partial(TrainFromCheckpoint, checkpoint_path=from_checkpoint),
                    self.config['callbacks'],
                ]
            )

        self.train_fn = train_fn
        self.local_dir = Path(local_dir)
        self.timesteps_total = timesteps_total
        self.experiment = tune.Experiment(
            name=self.name,
            run=run,
            config=self.config,
            stop={'timesteps_total': self.timesteps_total},
            local_dir=self.local_dir,
            trial_name_creator=lambda *_: f'{self.name}-{self.iteration:05d}',
            checkpoint_score_attr='episode_reward_mean',
            checkpoint_freq=20,
            checkpoint_at_end=True,
            max_failures=-1,
            **kwargs,
        )

        self.project = project
        self.group = group

        self._analysis = None

    @property
    def analysis(self):
        if self._analysis is None:
            self.train()
        return self._analysis

    def train(self, **kwargs):
        if self._analysis is None:
            print(
                f'{self.experiment.name}: Train {self.player} player for {self.timesteps_total} environment steps.'
            )
            self._analysis = self.train_fn(
                self.experiment,
                project=self.project,
                group=self.group,
                local_dir=self.local_dir,
                timesteps_total=self.timesteps_total,
                **kwargs,
            )
        return self._analysis

    def result(self, skip_train_if_exists=False, **kwargs):
        setproctitle(
            f'ray::{self.player.capitalize()}Trainer(name="{self.name}", iteration={self.iteration}).result()'
        )

        checkpoint_exists = False
        if skip_train_if_exists:
            result_file = self.local_dir / 'checkpoint_path.txt'
            if result_file.is_file():
                try:
                    best_checkpoint_path = Path(result_file.read_text(encoding='UTF-8').strip())
                except OSError:
                    pass
                else:
                    checkpoint_exists = best_checkpoint_path.is_file()

        if skip_train_if_exists and checkpoint_exists:
            print(
                f'{self.experiment.name}: Found existing checkpoint "{best_checkpoint_path}", skip training.'
            )
        else:
            self.train(**kwargs)

            best_trial = self.analysis.get_best_trial()
            if best_trial is None:
                best_trial = self.analysis.get_best_trial(filter_nan_and_inf=False)

            for metric in ('episode_reward_mean', 'training_iteration'):
                filtered_metric_checkpoints = [
                    (metric, checkpoint_path)
                    for checkpoint_path, metric in self.analysis.get_trial_checkpoints_paths(
                        best_trial, metric=metric
                    )
                    if np.isfinite(metric)
                ]
                if len(filtered_metric_checkpoints) > 0:
                    best_metric, best_checkpoint_path = max(filtered_metric_checkpoints)
                    break

        try:
            best_checkpoint_path = os.readlink(best_checkpoint_path)
        except OSError:
            pass
        best_checkpoint_path = Path(best_checkpoint_path)

        for target_dir in (self.experiment.local_dir, self.experiment.checkpoint_dir):
            SymlinkCheckpointCallback.symlink(
                source=best_checkpoint_path, target=os.path.join(target_dir, 'best-checkpoint')
            )

        # Delete checkpoints except the best and the latest ones
        best_checkpoint_dir = best_checkpoint_path.parent
        checkpoint_paths = sorted(
            map(str, best_checkpoint_dir.parent.glob('checkpoint_*')),
            key=lambda path: int(path.split('_')[-1]),
        )
        for checkpoint_path in checkpoint_paths[:-1]:
            if checkpoint_path != str(best_checkpoint_dir):
                shutil.rmtree(checkpoint_path, ignore_errors=True)

        return self.player, best_checkpoint_path
