#!/usr/bin/env python3

# Run: python3 -m examples.heuristic

"""Example of built-in heuristic agents for the Multi-Agent Tracking Environment."""

import mate
from mate.agents import HeuristicCameraAgent, HeuristicTargetAgent


MAX_EPISODE_STEPS = 4000


def main():
    base_env = mate.make('MultiAgentTracking-v0')
    base_env = mate.RenderCommunication(base_env)
    env = mate.MultiCamera(base_env, target_agent=HeuristicTargetAgent())
    print(env)

    camera_agents = HeuristicCameraAgent().spawn(env.num_cameras)

    camera_joint_observation = env.reset()
    env.render()

    mate.group_reset(camera_agents, camera_joint_observation)
    camera_infos = None

    for i in range(MAX_EPISODE_STEPS):
        camera_joint_action = mate.group_step(
            env, camera_agents, camera_joint_observation, camera_infos
        )

        results = env.step(camera_joint_action)
        camera_joint_observation, camera_team_reward, done, camera_infos = results

        env.render()
        if done:
            break


if __name__ == '__main__':
    main()
