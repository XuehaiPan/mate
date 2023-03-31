#!/usr/bin/env python3

"""Evaluation script for the Multi-Agent Tracking Environment."""

import argparse
import importlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Union

import gym
import numpy as np
import tqdm
from gym.utils import colorize
from pkg_resources import parse_version

import mate


@dataclass
class Column:  # pylint: disable=missing-class-docstring,missing-function-docstring
    name: str
    width: int
    fmt: Callable[[Union[int, float]], str] = '{}'.format
    color: str = 'white'
    bold: bool = False
    highlight: bool = False
    justification: Callable[..., str] = str.rjust

    @property
    def formatter(self):
        return colorize(' {} ', color=self.color, bold=self.bold, highlight=self.highlight)

    def title(self, width=None):
        if width is None:
            width = self.width
        return self.formatter.format(self.justification(self.name, width))

    def separator(self, width=None):
        if width is None:
            width = self.width
        return self.formatter.format(self.justification(':', width, '-'))

    def format(self, value, width=None):
        if width is None:
            width = self.width
        return self.formatter.format(self.fmt(value).rjust(width))


COLUMNS = [
    Column(name='Step', fmt='{:d}'.format,
           width=6, color='red'),
    Column(name='Cargo', fmt='{:d}'.format,
           width=5, color='green'),
    Column(name='Reward', fmt='{:+.2f}'.format,
           width=8, color='yellow'),
    Column(name='Target Episode Reward', fmt='{:+.2f}'.format,
           width=21, color='blue', bold=True),
    Column(name='Step / Cargo', fmt='{:.1f}'.format,
           width=12, color='magenta'),
    Column(name='Mean Transport Rate', fmt=lambda x: f'{100.0 * x:.3f}%',
           width=19, color='cyan', bold=True),
    Column(name='Mean Coverage Rate', fmt=lambda x: f'{100.0 * x:.3f}%',
           width=18, color='red', bold=True),
    Column(name='Normalized Target Episode Reward', fmt='{:+.5f}'.format,
           width=32, color='green', bold=True),
    Column(name='FPS', fmt='{:.1f}'.format,
           width=5, color='yellow'),
]  # fmt: skip
COLUMNS = OrderedDict([(column.name, column) for column in COLUMNS])


def load_entry(entry_point):
    """Load a module attribute from given entry point."""

    mod_name, attr_name = entry_point.split(':')
    mod = importlib.import_module(mod_name)
    entry = getattr(mod, attr_name)
    return entry


def evaluate(
    env, target_agents, render=False, video_path=None
):  # pylint: disable=missing-function-docstring,too-many-locals,too-many-branches,too-many-statements
    status = {}
    if render and video_path is not None:
        # pylint: disable-next=import-outside-toplevel
        from gym.wrappers.monitoring.video_recorder import VideoRecorder

        video_path = os.path.realpath(video_path)
        print(f'Rollout video will be saved to "{video_path}".')
        print()
        recorder = VideoRecorder(env, path=video_path)
        recorder.__del__ = lambda r: r.close()
    else:
        recorder = None

    target_joint_observation = env.reset()
    mate.group_reset(target_agents, target_joint_observation)
    target_infos = None

    if render:
        if recorder is not None:
            recorder.capture_frame()
        else:
            env.render()
        time.sleep(1.0)

    headers = False
    num_cargoes = 0
    target_team_episode_reward = 0.0
    time_start = time.perf_counter()
    coverage_rates = []
    while env.episode_step < env.max_episode_steps:
        target_joint_action = mate.group_step(
            env, target_agents, target_joint_observation, target_infos
        )

        target_joint_observation, target_team_reward, done, target_infos = env.step(
            target_joint_action
        )
        coverage_rates.append(env.coverage_rate)

        num_cargoes = env.num_delivered_cargoes
        target_team_episode_reward += target_team_reward

        values = [
            env.episode_step,
            num_cargoes,
            target_team_reward,
            target_team_episode_reward,
            env.episode_step / num_cargoes if num_cargoes > 0 else np.nan,
            env.mean_transport_rate,
            np.mean(coverage_rates),
            target_team_episode_reward / env.max_target_team_episode_reward,
            env.episode_step / (time.perf_counter() - time_start),
        ]

        if num_cargoes > 0 or done:
            status = dict(zip(COLUMNS, values))

        if render:
            if not headers:
                print('|'.join(['', *map(Column.title, COLUMNS.values()), '']))
                print('|'.join(['', *map(Column.separator, COLUMNS.values()), '']))
                headers = True
            print('|'.join(['', *map(Column.format, COLUMNS.values(), values), '']))

        if render:
            if recorder is not None:
                recorder.capture_frame()
            else:
                env.render()

        if done:
            break

    if render:
        if recorder is not None:
            recorder.close()
        time.sleep(1.0)
        print()

    return status


def parse_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser(
        prog='python -m mate.evaluate',
        description='Evaluation script for the Multi-Agent Tracking Environment.',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--help',
        '-h',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.',
    )

    environment_parser = parser.add_argument_group('environment')
    environment_parser.add_argument(
        '--config',
        '--cfg',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to a JSON/YAML configuration file of MultiAgentTracking.',
    )
    environment_parser.add_argument(
        '--enhanced-observation',
        type=str,
        metavar='TEAM',
        default='none',
        const='both',
        nargs='?',
        choices=['both', 'camera', 'target', 'none'],
        help=(
            "Enhance the agent's observation in the given team.\n"
            'If the argument is omitted, set for both teams.'
        ),
    )
    environment_parser.add_argument(
        '--shared-field-of-view',
        type=str,
        metavar='TEAM',
        default='none',
        const='both',
        nargs='?',
        choices=['both', 'camera', 'target', 'none'],
        help=(
            'Share the field of view among agents in the given team.\n'
            'If the argument is omitted, set for both teams.'
        ),
    )
    environment_parser.add_argument(
        '--no-communication',
        type=str,
        metavar='TEAM',
        default='none',
        const='both',
        nargs='?',
        choices=['both', 'camera', 'target', 'none'],
        help=(
            'Disable all communications for the given team.\n'
            'If the argument is omitted, set for both teams.'
        ),
    )
    environment_parser.add_argument(
        '--seed',
        type=int,
        metavar='SEED',
        default=0,
        help='Random seed for RNGs, overwrites agent arguments. (default: %(default)d)',
    )
    environment_parser.add_argument(
        '--episodes',
        type=int,
        metavar='EPISODE',
        default=20,
        help='Number of episodes to evaluate. (default: %(default)d)',
    )

    agent_parser = parser.add_argument_group('agent')
    agent_parser.add_argument(
        '--camera-agent',
        type=load_entry,
        metavar='ENTRY',
        default='mate:GreedyCameraAgent',
        help='Entry point of camera agent class.\n(default: %(default)s)',
    )
    agent_parser.add_argument(
        '--target-agent',
        type=load_entry,
        metavar='ENTRY',
        default='mate:GreedyTargetAgent',
        help='Entry point of target agent class.\n(default: %(default)s)',
    )
    agent_parser.add_argument(
        '--camera-kwargs',
        type=json.loads,
        metavar='STRING',
        default='{}',
        help=(
            'Keyword arguments of camera agents in JSON string.\n'
            "(example: '{\"discrete_levels\": 5}', default: '{}')"
        ),
    )
    agent_parser.add_argument(
        '--target-kwargs',
        type=json.loads,
        metavar='STRING',
        default='{}',
        help=(
            'Keyword arguments of target agents in JSON string.\n'
            "(example: '{\"discrete_levels\": 5}', default: '{}')"
        ),
    )
    agent_parser.add_argument(
        '--camera-discrete-levels',
        type=int,
        metavar='LEVEL',
        default=None,
        help=(
            'Levels of discrete action space for camera agents,\n'
            'continuous action space will be used if not present.'
        ),
    )
    agent_parser.add_argument(
        '--target-discrete-levels',
        type=int,
        metavar='LEVEL',
        default=None,
        help=(
            'Levels of discrete action space for camera agents,\n'
            'continuous action space will be used if not present.'
        ),
    )

    rendering_parser = parser.add_argument_group('rendering')
    rendering_parser.add_argument(
        '--no-render',
        action='store_true',
        help=(
            'Do not render the environment.\n'
            'Suppress options `--render-communication` and `--save-video`.'
        ),
    )
    rendering_parser.add_argument(
        '--render-communication',
        type=int,
        metavar='DURATION',
        default=None,
        const=20,
        nargs='?',
        help=(
            'Draw arrows for communication edges in the rendering results.\n'
            '(default duration: %(const)d)'
        ),
    )
    rendering_parser.add_argument(
        '--save-video',
        type=str,
        metavar='PATH',
        nargs='?',
        default=argparse.SUPPRESS,
        help='Save the render video (default: "video.mp4")',
    )

    args = parser.parse_args()

    assert issubclass(args.camera_agent, mate.CameraAgentBase), (
        f'You should provide a subclass of `mate.CameraAgentBase`. '
        f'Got camera_agent = {args.camera_agent}.'
    )
    assert issubclass(args.target_agent, mate.TargetAgentBase), (
        f'You should provide a subclass of `mate.TargetAgentBase`. '
        f'Got target_agent = {args.target_agent}.'
    )
    assert (
        args.episodes > 0
    ), f'The argument `episodes` should be a positive number. Got episodes = {args.episodes}.'

    if not hasattr(args, 'save_video'):
        args.save_video = None
    elif args.save_video is None:
        args.save_video = 'video.mp4'
    if args.no_render:
        args.save_video = None
    if args.save_video is not None and parse_version(gym.__version__) < parse_version('0.18.3'):
        gym.logger.warn(
            'Video recording requires gym 0.18.3 or higher (current version: %s).', gym.__version__
        )

    if args.no_render:
        args.render_communication = False

    args.camera_kwargs = OrderedDict(sorted(dict(args.camera_kwargs, seed=args.seed).items()))
    args.target_kwargs = OrderedDict(sorted(dict(args.target_kwargs, seed=args.seed).items()))
    args.camera_kwargs.move_to_end('seed')
    args.target_kwargs.move_to_end('seed')
    camera_kwargs_joined = ', '.join(f'{k}={v!r}' for k, v in args.camera_kwargs.items())
    target_kwargs_joined = ', '.join(f'{k}={v!r}' for k, v in args.target_kwargs.items())
    args.camera_name = '{cls.__module__}.{cls.__name__}({kwargs})'.format(
        cls=args.camera_agent, kwargs=camera_kwargs_joined
    )
    args.target_name = '{cls.__module__}.{cls.__name__}({kwargs})'.format(
        cls=args.target_agent, kwargs=target_kwargs_joined
    )

    return args


def main():  # pylint: disable=missing-function-docstring,too-many-branches,too-many-statements
    args = parse_arguments()

    mate.seed_everything(args.seed)

    camera_agent = args.camera_agent(**args.camera_kwargs)
    target_agent = args.target_agent(**args.target_kwargs)

    wrappers = []
    if args.enhanced_observation != 'none':
        wrappers.append(mate.WrapperSpec(mate.EnhancedObservation, team=args.enhanced_observation))
    if args.shared_field_of_view != 'none':
        wrappers.append(mate.WrapperSpec(mate.SharedFieldOfView, team=args.shared_field_of_view))
    if args.no_communication != 'none':
        wrappers.append(mate.WrapperSpec(mate.NoCommunication, team=args.no_communication))
    if args.render_communication is not None and args.render_communication:
        wrappers.append(
            mate.WrapperSpec(mate.RenderCommunication, duration=args.render_communication)
        )
    if args.camera_discrete_levels is not None:
        wrappers.append(mate.WrapperSpec(mate.DiscreteCamera, levels=args.camera_discrete_levels))
    if args.target_discrete_levels is not None:
        wrappers.append(mate.WrapperSpec(mate.DiscreteTarget, levels=args.target_discrete_levels))
    wrappers.append(mate.WrapperSpec(mate.MultiTarget, camera_agent=camera_agent))

    env = mate.make('MultiAgentTracking-v0', config=args.config, wrappers=wrappers)
    env.seed(args.seed)

    print(f'Environment:  {env}')
    print(f'Camera agent: {args.camera_name}')
    print(f'Target agent: {args.target_name}')

    target_agents = target_agent.spawn(env.num_targets)

    keys = [
        'Step / Cargo',
        'Target Episode Reward',
        'Mean Transport Rate',
        'Mean Coverage Rate',
        'Normalized Target Episode Reward',
    ]
    statuses = OrderedDict([(key, []) for key in keys])
    initial = 0
    postfix = None

    if not args.no_render:
        print()
        try:
            status = evaluate(env, target_agents, render=True, video_path=args.save_video)
        except KeyboardInterrupt:
            pass
        else:
            for key in keys:
                statuses[key].append(status[key])
            initial = 1
            postfix = OrderedDict([
                ('MeanCoverageRate', f'{100.0 * np.mean(statuses["Mean Coverage Rate"]):.1f}%'),
                ('MeanTransportRate', f'{100.0 * np.mean(statuses["Mean Transport Rate"]):.1f}%'),
                ('NormalizedTargetEpisodeReward', f'{np.mean(statuses["Normalized Target Episode Reward"]):+.5f}'),
                ('FPS', status['FPS'])
            ])  # fmt: skip
        finally:
            if env.viewer is not None:
                env.viewer.close()
                env.viewer = None

    try:
        with tqdm.trange(
            initial,
            args.episodes,
            desc='Evaluating',
            unit='episode',
            total=args.episodes,
            initial=initial,
            postfix=postfix,
        ) as pbar:
            for _ in pbar:
                status = evaluate(env, target_agents, render=False)
                for key in keys:
                    statuses[key].append(status[key])
                pbar.set_postfix(OrderedDict([
                    ('MeanCoverageRate', f'{100.0 * np.mean(statuses["Mean Coverage Rate"]):.1f}%'),
                    ('MeanTransportRate', f'{100.0 * np.mean(statuses["Mean Transport Rate"]):.1f}%'),
                    ('NormalizedTargetEpisodeReward', f'{np.mean(statuses["Normalized Target Episode Reward"]):+.5f}'),
                    ('FPS', status['FPS'])
                ]))  # fmt: skip
    except KeyboardInterrupt:
        pass

    if len(statuses[keys[-1]]) > 0:
        # pylint: disable=consider-using-f-string
        print('| {:>32} | {:>12} |'.format('Metric', 'Mean'))
        print('| {:->32} | {:->12} |'.format(':', ':'))
        for key, values in statuses.items():
            print(
                '|{}|{}|'.format(
                    COLUMNS[key].title(width=32), COLUMNS[key].format(np.mean(values), width=12)
                )
            )
        # pylint: disable-enable=consider-using-f-string


if __name__ == '__main__':
    main()
