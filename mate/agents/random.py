"""Random agents."""

from mate.agents.base import CameraAgentBase, TargetAgentBase


__all__ = ['RandomCameraAgent', 'RandomTargetAgent']


class RandomCameraAgent(CameraAgentBase):
    """Random Camera Agent

    Random action.
    """

    def __init__(self, seed=None, frame_skip=20):
        """Initialize the agent.
        This function will be called only once on initialization.

        Note:
            Agents can obtain the number of teammates and opponents on reset,
            but not here. You are responsible for writing scalable policies and
            code to handle this.
        """

        super().__init__(seed=seed)

        self.frame_skip = frame_skip
        self.prev_action = None

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().

        Note:
            observation is a 1D array, not a 2D array with an additional
            dimension for agent indices.
        """

        super().reset(observation)

        self.prev_action = None

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Random action.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        if self.prev_action is None or self.episode_step % self.frame_skip == 0:
            action = self.prev_action = self.action_space.sample()
        else:
            action = self.prev_action
        return action


class RandomTargetAgent(TargetAgentBase):
    """Random Target Agent

    Random action.
    """

    def __init__(self, seed=None, frame_skip=20):
        """Initialize the agent.
        This function will be called only once on initialization.

        Note:
            Agents can obtain the number of teammates and opponents on reset,
            but not here. You are responsible for writing scalable policies and
            code to handle this.
        """

        super().__init__(seed=seed)

        self.frame_skip = frame_skip
        self.prev_action = None

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().

        Note:
            observation is a 1D array, not a 2D array with an additional
            dimension for agent indices.
        """

        super().reset(observation)

        self.prev_action = None

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Random action.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        if self.prev_action is None or self.episode_step % self.frame_skip == 0:
            action = self.prev_action = self.action_space.sample()
        else:
            action = self.prev_action
        return action
