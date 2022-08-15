#!/usr/bin/env python3

# Run: python3 -m examples.naive

"""Example of naive agents for the Multi-Agent Tracking Environment."""

import mate
from mate.agents import NaiveCameraAgent, NaiveTargetAgent


MAX_EPISODE_STEPS = 4000


def main():
    base_env = mate.make('MultiAgentTracking-v0')
    base_env = mate.RenderCommunication(base_env)
    env = mate.MultiTarget(base_env, camera_agent=NaiveCameraAgent())
    print(env)

    target_agents = NaiveTargetAgent().spawn(env.num_targets)

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
