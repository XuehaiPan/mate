#!/usr/bin/env python3

r"""
Example of IPPO agents for the Multi-Agent Tracking Environment.

.. code:: bash

    python3 -m examples.ippo.target

    python3 -m mate.evaluate --episodes 1 --render-communication \
        --target-agent examples.ippo:IPPOTargetAgent \
        --target-kwargs '{ "checkpoint_path": "examples/ippo/target/ray_results/IPPO/latest-checkpoint" }'
"""

import argparse
import os
import sys

import mate
from examples.ippo.target.agent import IPPOTargetAgent
from examples.ippo.target.train import experiment


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')

MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path',
        '--checkpoint',
        '--ckpt',
        type=str,
        metavar='PATH',
        default=CHECKPOINT_PATH,
        help='path to the checkpoint file',
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        metavar='STEP',
        default=MAX_EPISODE_STEPS,
        help='maximum episode steps (default: %(default)d)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', default=0, help='the global seed (default: %(default)d)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(
            (
                f'Model checkpoint ("{args.checkpoint_path}") does not exist. Please run the following command to train a model first:\n'
                f'  python -m examples.ippo.target.train'
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # Make agents ##############################################################
    camera_agent = mate.GreedyCameraAgent()
    target_agent = IPPOTargetAgent(checkpoint_path=args.checkpoint_path)

    # Make the environment #####################################################
    env_config = target_agent.config.get('env_config', {})
    enhanced_observation_team = str(env_config.get('enhanced_observation', None)).lower()

    base_env = mate.make(
        'MultiAgentTracking-v0',
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )
    base_env = mate.RenderCommunication(base_env)
    if enhanced_observation_team is not None:
        base_env = mate.EnhancedObservation(base_env, team=enhanced_observation_team)
    env = mate.MultiTarget(base_env, camera_agent=camera_agent)
    print(env)

    # Rollout ##################################################################
    target_agents = target_agent.spawn(env.num_targets)

    target_joint_observation = env.reset()
    env.render()

    mate.group_reset(target_agents, target_joint_observation)
    target_infos = None

    for i in range(MAX_EPISODE_STEPS):
        target_joint_action = mate.group_step(
            env, target_agents, target_joint_observation, target_infos
        )

        results = env.step(target_joint_action)
        target_joint_observation, target_team_reward, done, target_infos = results

        env.render()
        if done:
            break


if __name__ == '__main__':
    main()
