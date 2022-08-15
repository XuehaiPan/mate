#!/usr/bin/env python3

# Run: python3 -m examples.random

"""Example of random agents for the Multi-Agent Tracking Environment."""

import mate
from mate.agents import RandomCameraAgent, RandomTargetAgent


MAX_EPISODE_STEPS = 4000


def main():
    env = mate.make('MultiAgentTracking-v0')
    print(env)

    camera_agents = RandomCameraAgent().spawn(env.num_cameras)
    target_agents = RandomTargetAgent().spawn(env.num_targets)

    camera_joint_observation, target_joint_observation = env.reset()
    env.render()

    mate.group_reset(camera_agents, camera_joint_observation)
    mate.group_reset(target_agents, target_joint_observation)
    camera_infos = None
    target_infos = None
    for i in range(MAX_EPISODE_STEPS):
        camera_joint_action = mate.group_step(
            env, camera_agents, camera_joint_observation, camera_infos
        )
        target_joint_action = mate.group_step(
            env, target_agents, target_joint_observation, target_infos
        )

        (
            (camera_joint_observation, target_joint_observation),
            (camera_team_reward, target_team_reward),
            done,
            (camera_infos, target_infos),
        ) = env.step((camera_joint_action, target_joint_action))

        env.render()
        if done:
            break


if __name__ == '__main__':
    main()
