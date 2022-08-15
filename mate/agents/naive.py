"""Built-in naive rule-based agents."""

import numpy as np

from mate.agents.base import CameraAgentBase, TargetAgentBase
from mate.constants import NUM_WAREHOUSES, WAREHOUSE_RADIUS, WAREHOUSES


__all__ = ['NaiveCameraAgent', 'NaiveTargetAgent']


class NaiveCameraAgent(CameraAgentBase):
    """Naive Camera Agent

    Rotates anti-clockwise with the maximum viewing angle.
    """

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Rotate anti-clockwise with the maximum viewing angle.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        action = self.np_random.uniform(0.0, 0.4) * self.action_space.high

        return action


class NaiveTargetAgent(TargetAgentBase):
    """Naive Target Agent

    Visits all warehouses in turn.
    """

    def __init__(self, seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """

        super().__init__(seed=seed)

        self.goal = 0
        self.prev_state = None
        self.prev_noise = None
        self.inc = +1

    @property
    def goal_location(self):
        """Location of the current warehouse."""

        return WAREHOUSES[self.goal]

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().
        """

        super().reset(observation)

        self.prev_state = self.state
        self.prev_noise = 0.5 * self.action_space.sample()

        self.goal = self.np_random.choice(NUM_WAREHOUSES)

        self.inc = self.np_random.choice([+1, -1])

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Visit all warehouses in turn.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        if np.linalg.norm(self.state.location - self.goal_location) <= 0.9 * WAREHOUSE_RADIUS:
            if self.state.goal_bits.any() or self.state.empty_bits.all():
                self.goal = (self.goal + self.inc) % NUM_WAREHOUSES
            else:
                while True:
                    self.goal = (self.goal + self.inc) % NUM_WAREHOUSES
                    if not self.state.empty_bits[self.goal]:
                        break

        prev_actual_action = self.state.location - self.prev_state.location

        action = self.goal_location - self.state.location
        step_size = np.linalg.norm(action)
        if step_size > self.state.step_size:
            action *= self.state.step_size / step_size

        prob = 0.05 if np.linalg.norm(prev_actual_action) > 0.2 * self.state.step_size else 0.75
        if self.np_random.binomial(1, prob) != 0:
            noise = 0.5 * self.action_space.sample()
        else:
            noise = self.prev_noise

        action = (action + noise).clip(min=self.action_space.low, max=self.action_space.high)

        self.prev_state = self.state
        self.prev_noise = noise
        return action
