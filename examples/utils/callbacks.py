import logging
import os
import platform
import re
from collections import OrderedDict, defaultdict
from operator import itemgetter
from pathlib import Path

import numpy as np
from gym import spaces
from ray.rllib.agents.callbacks import DefaultCallbacks as RLlibCallbackBase
from ray.rllib.agents.callbacks import MultiCallbacks as RLlibMultiCallbacksBase
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces import space_utils
from ray.tune.callback import Callback as TuneCallbackBase

from .rllib_policy import get_preprocessor, get_space_flat_size, load_checkpoint


logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    logger.error('Run `pip install wandb` to use WandbLoggerCallback.')
    wandb = None
    from ray.tune.logger import LoggerCallback as WandbLoggerCallbackBase
else:
    from ray.tune.integration.wandb import WandbLoggerCallback as WandbLoggerCallbackBase


__all__ = [
    'ShiftAgentActionTimestep',
    'MetricCollector',
    'CustomMetricCallback',
    'TrainFromCheckpoint',
    'SymlinkCheckpointCallback',
    'RLlibMultiCallbacks',
    'WandbLoggerCallback',
]


class ShiftAgentActionTimestep(RLlibCallbackBase):
    def __init__(self):
        super().__init__()

        self.agent_ids = []
        self.other_agent_ids = OrderedDict()

        self.observation_space = None
        self.obs_flat_dim = None

        self.others_joint_action_space = None
        self.others_joint_action_preprocessor = None
        self.others_joint_action_dim = 0
        self.others_joint_action_slice = slice(0, 0)

        self.identifier = None
        self.last_next_actions = {}

    def on_sub_environment_created(self, *, worker, sub_environment, env_context, **kwargs):
        if self.observation_space is not None:
            return

        from .wrappers import RLlibMultiAgentCentralizedTraining

        assert isinstance(sub_environment, RLlibMultiAgentCentralizedTraining)

        self.agent_ids = list(sub_environment.agent_ids)
        cycled_agent_ids = self.agent_ids + self.agent_ids
        self.other_agent_ids = OrderedDict(
            [
                (agent_id, cycled_agent_ids[i + 1 : i + len(self.agent_ids)])
                for i, agent_id in enumerate(self.agent_ids)
            ]
        )

        self.observation_space = sub_environment.observation_space
        self.obs_flat_dim = get_space_flat_size(self.observation_space)

        self.others_joint_action_space = self.observation_space['prev_others_joint_action']
        assert isinstance(self.others_joint_action_space, spaces.Tuple)
        self.others_joint_action_preprocessor = get_preprocessor(self.others_joint_action_space)
        self.others_joint_action_dim = self.others_joint_action_preprocessor.size
        self.others_joint_action_slice = slice(-self.others_joint_action_dim, None)

        self.identifier = None
        self.last_next_actions = {}

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
        batch_identifier = tuple(
            id(agent_batch) for _, agent_batch in map(original_batches.get, self.agent_ids)
        )
        identifier = hash((episode, batch_identifier))
        if identifier != self.identifier:
            self.last_next_actions.clear()
            self.identifier = identifier

        for aid in self.agent_ids:
            if aid in self.last_next_actions:
                continue  # cached

            policy = policies[episode.policy_for(aid)]
            action, *_ = policy.compute_single_action(
                episode.last_observation_for(aid),
                state=episode.rnn_state_for(aid),
                prev_action=episode.last_action_for(aid),
                prev_reward=episode.last_reward_for(aid),
                info=episode.last_info_for(aid),
            )
            if policy.config.get('normalize_actions', True):
                action = space_utils.unsquash_action(action, policy.action_space_struct)
            elif policy.config.get('clip_actions', False):
                action = space_utils.clip_action(action, policy.action_space_struct)
            self.last_next_actions[aid] = action

        postprocessed_batch[SampleBatch.CUR_OBS][
            :, self.others_joint_action_slice
        ] = postprocessed_batch[SampleBatch.NEXT_OBS][:, self.others_joint_action_slice]
        postprocessed_batch[SampleBatch.NEXT_OBS][
            :-1, self.others_joint_action_slice
        ] = postprocessed_batch[SampleBatch.NEXT_OBS][1:, self.others_joint_action_slice]

        last_others_next_actions = tuple(
            self.last_next_actions[aid] for aid in self.other_agent_ids[agent_id]
        )
        last_others_next_joint_action = self.others_joint_action_preprocessor.transform(
            last_others_next_actions
        )
        postprocessed_batch[SampleBatch.NEXT_OBS][
            -1, self.others_joint_action_slice
        ] = last_others_next_joint_action


class MetricCollector:
    REDUCERS = {
        'mean': np.mean,
        'sum': np.sum,
        'std': np.std,
        'last': itemgetter(-1)
    }  # fmt: skip

    def __init__(self, metrics):
        self.metrics = metrics
        self.accessed_patterns = defaultdict(set)
        self.data = defaultdict(list)

    def clear(self):
        self.accessed_patterns.clear()
        self.data.clear()

    def add(self, infos):
        if not isinstance(infos, (list, tuple)):
            infos = (infos,)

        values = defaultdict(list)
        for pattern in self.metrics:
            for info in infos:
                for key, value in info.items():
                    if self.match(pattern, key):
                        self.accessed_patterns[pattern].add(key)
                        values[key].append(value)

        for key in values:
            self.data[key].append(np.mean(values[key]))

    def collect(self):
        results = {}
        for pattern, keys in self.accessed_patterns.items():
            reduction = self.metrics[pattern]
            reducer = self.REDUCERS[reduction]
            for key in keys:
                if len(self.data[key]) > 0:
                    results[key] = float(reducer(self.data[key]))

        return results

    @staticmethod
    def match(pattern, string):
        if isinstance(pattern, re.Pattern):
            return pattern.match(string) is not None
        return string == pattern


class CustomMetricCallback(RLlibCallbackBase):
    DEFAULT_CUSTOM_METRICS = {
        'raw_reward': 'mean',
        'normalized_raw_reward': 'mean',
        re.compile(r'^auxiliary_reward(\w*)$'): 'mean',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
    }

    def __init__(self, custom_metrics=None):
        super().__init__()

        self.custom_metrics = custom_metrics or self.DEFAULT_CUSTOM_METRICS

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data['collector'] = MetricCollector(self.custom_metrics)

    def on_episode_step(self, *, episode, **kwargs):
        from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_INFO

        infos = []
        for info in map(episode.last_info_for, episode.get_agents()):
            infos.extend(info.get(GROUP_INFO, [info]))

        episode.user_data['collector'].add(infos)

    def on_episode_end(self, *, episode, **kwargs):
        collector = episode.user_data['collector']
        custom_metrics = collector.collect()
        for key in tuple(custom_metrics):
            if key.endswith('reward') and not key.startswith(('episode', 'reward_coefficient')):
                custom_metrics[f'episode_{key}'] = float(np.sum(collector.data[key]))

        episode.custom_metrics.update(custom_metrics)


class TrainFromCheckpoint(RLlibCallbackBase):
    def __init__(self, checkpoint_path=None):
        super().__init__()

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not (checkpoint_path.exists() and checkpoint_path.is_file()):
                raise FileNotFoundError(f'Checkpoint path "{checkpoint_path}" does not exist.')

            checkpoint_path = checkpoint_path.absolute()
            try:
                checkpoint_path = checkpoint_path.readlink()
            except OSError:
                pass

        self.checkpoint_path = checkpoint_path

    def on_trainer_init(self, *, trainer, **kwargs):
        if self.checkpoint_path is None:
            return

        _, worker, _ = load_checkpoint(self.checkpoint_path)

        weights = {
            policy_id: policy_state['weights']
            for policy_id, policy_state in worker['state'].items()
        }

        trainer.workers.local_worker().set_weights(weights)
        if trainer.workers.remote_workers():
            trainer.workers.sync_weights()


class SymlinkCheckpointCallback(TuneCallbackBase):
    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        source = checkpoint.value
        for target_dir in (trial.logdir, trial.local_dir):
            target = os.path.join(target_dir, 'latest-checkpoint')
            print(f'Symlink "{source}" to "{target}".')
            self.symlink(source, target)

    @staticmethod
    def symlink(source, target):
        temp_target = f'{target}.temp'

        os_symlink = getattr(os, 'symlink', None)
        if callable(os_symlink):
            os_symlink(source, temp_target)
        elif platform.system() == 'Windows':
            import ctypes

            csl = ctypes.windll.kernel32.CreateSymbolicLinkW
            csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
            csl.restype = ctypes.c_ubyte
            flags = 1 if os.path.isdir(source) else 0
            if csl(temp_target, source, flags) == 0:
                raise ctypes.WinError(f'Cannot create symlink "{source}" to "{target}".')
        else:
            raise OSError(f'Cannot create symlink "{source}" to "{target}".')

        os.replace(temp_target, target)


class RLlibMultiCallbacks(RLlibMultiCallbacksBase):  # add missing methods
    def on_sub_environment_created(self, *, worker, sub_environment, env_context, **kwargs):
        for callback in self._callback_list:
            callback.on_sub_environment_created(
                worker=worker,
                sub_environment=sub_environment,
                env_context=env_context,
                **kwargs,
            )

    def on_trainer_init(self, *, trainer, **kwargs):
        for callback in self._callback_list:
            callback.on_trainer_init(trainer=trainer, **kwargs)


class WandbLoggerCallback(WandbLoggerCallbackBase):
    WANDB_ENV_VAR = 'WANDB_API_KEY'

    def log_trial_start(self, trial):
        try:
            self.kwargs['notes'] = trial.logdir
        except AttributeError:
            pass
        super().log_trial_start(trial)

    @classmethod
    def set_api_key(cls, api_key_file=None, api_key=None):
        """Set WandB API key"""

        if api_key_file:
            if api_key:
                raise ValueError('Both WandB `api_key_file` and `api_key` set.')
            with open(api_key_file, encoding='UTF-8') as file:
                api_key = file.readline().strip()
        if api_key:
            os.environ[cls.WANDB_ENV_VAR] = api_key
        elif not os.environ.get(cls.WANDB_ENV_VAR):
            try:
                # Check if user is already logged into wandb.
                wandb.ensure_configured()
                if wandb.api.api_key:
                    logger.info('Already logged into W&B.')
                    return
            except AttributeError:
                pass
            raise ValueError(
                f'No WandB API key found. Either set the `{cls.WANDB_ENV_VAR}` '
                f'environment variable, pass `api_key` or `api_key_file` to '
                f'the `WandbLoggerCallback` class as arguments, '
                f'or run `wandb login` from the command line.'
            )

    @classmethod
    def is_available(cls, api_key_file=None, api_key=None):
        if wandb is None:
            return False

        try:
            cls.set_api_key(api_key_file, api_key)
        except ValueError as ex:
            logging.error(ex)
            return False

        return True
