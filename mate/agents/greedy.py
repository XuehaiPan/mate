"""Built-in greedy rule-based agents."""

import numpy as np

from mate.agents.base import CameraAgentBase, TargetAgentBase
from mate.constants import MAX_CAMERA_VIEWING_ANGLE, NUM_WAREHOUSES, WAREHOUSES
from mate.utils import normalize_angle, sin_deg


__all__ = ['GreedyCameraAgent', 'GreedyTargetAgent']


class GreedyCameraAgent(CameraAgentBase):  # pylint: disable=too-many-instance-attributes
    """Greedy Camera Agent

    Arbitrarily tracks the nearest target.
    If no target found, use previous action or generate a new random action.
    """

    def __init__(
        self, seed=None, memory_period=25, filterout_unloaded=False, filterout_beyond_range=True
    ):
        """Initialize the agent.
        This function will be called only once on initialization.
        """

        super().__init__(seed=seed)

        self.filterout_unloaded = filterout_unloaded
        self.filterout_beyond_range = filterout_beyond_range
        self.range_factor = 1.1  # 110%

        self.memory = None
        self.time2forget = None
        self.never_loaded = None
        self.memory_period = memory_period
        self.prev_action = self.DEFAULT_ACTION

        self.neighboring_teammate_states = {}
        self.message2send = {}
        self.communication_delay = None

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().
        """

        super().reset(observation)

        target_states, tracked_bits = self.get_all_opponent_states(observation)
        self.memory = list(target_states)
        self.time2forget = self.memory_period * np.asarray(tracked_bits, dtype=np.int64)
        self.never_loaded = np.ones(self.num_targets, dtype=np.bool8)

        self.prev_action = self.DEFAULT_ACTION

        self.neighboring_teammate_states.clear()
        self.message2send.clear()
        self.communication_delay = np.zeros(self.num_teammates, dtype=np.int64)
        self.message2send['state'] = self.state.copy()

    def observe(self, observation, info=None):
        """The agent observe the environment before sending messages.
        This function will be called before send_responses().
        """

        self.state, observation, info, messages = self.check_inputs(observation, info)

        self.process_messages(observation, messages)

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Arbitrarily track the nearest target.
        If no target found, use previous action or generate a new random action.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        tracked_targets = [self.memory[t] for t in np.flatnonzero(self.time2forget)]
        if self.filterout_beyond_range:
            threshold = self.range_factor * self.state.max_sight_range
            tracked_targets = [ts for ts in tracked_targets if (ts - self.state).norm < threshold]
        if self.filterout_unloaded:
            tracked_targets = [
                ts for ts in tracked_targets if ts.is_loaded or self.never_loaded[ts.index]
            ]

        if len(tracked_targets) > 0:
            action = self.act_from_target_states(tracked_targets)
        else:
            if self.np_random.binomial(1, 0.1) != 0:
                action = self.action_space.sample()
            else:
                action = self.prev_action

        self.prev_action = action
        return action

    def process_messages(self, observation, messages):  # pylint: disable=unused-argument
        """Process observation and prepare messages to teammates."""

        self.time2forget = np.maximum(self.time2forget - 1, 0, dtype=np.int64)

        target_states, tracked_bits = self.get_all_opponent_states(observation)
        for t in np.flatnonzero(tracked_bits):
            self.time2forget[t] = self.memory_period
            self.memory[t] = target_states[t]
            if target_states[t].is_loaded:
                self.never_loaded[t] = False
            self.message2send.setdefault('target_states', [])
            self.message2send['target_states'].append(target_states[t])

    def act_from_target_states(self, target_states):
        """Place the selected target at the center of the field of view."""

        assert (
            len(target_states) > 0
        ), 'You should provide at least one target to compute the action.'

        def select_target():
            """Select the nearest target."""

            return min(target_states, key=lambda ts: (ts - self.state).norm)

        def best_orientation():
            return (target_state - self.state).angle

        def best_viewing_angle():
            distance = (target_state - self.state).norm

            if (
                distance * (1.0 + sin_deg(self.state.min_viewing_angle / 2.0))
                >= self.state.max_sight_range
            ):
                return self.state.min_viewing_angle

            area_product = self.state.viewing_angle * np.square(self.state.sight_range)
            if distance <= np.sqrt(area_product / 180.0) / 2.0:
                return min(180.0, MAX_CAMERA_VIEWING_ANGLE)

            best = min(180.0, MAX_CAMERA_VIEWING_ANGLE)
            for _ in range(20):
                sight_range = distance * (1.0 + sin_deg(min(best / 2.0, 90.0)))
                best = area_product / np.square(sight_range)
            return np.clip(best, a_min=self.state.min_viewing_angle, a_max=MAX_CAMERA_VIEWING_ANGLE)

        target_state = select_target()

        return np.asarray(
            [
                normalize_angle(best_orientation() - self.state.orientation),
                best_viewing_angle() - self.state.viewing_angle,
            ]
        ).clip(min=self.action_space.low, max=self.action_space.high)

    def send_responses(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called before receive_responses().

        Send the newest target states to teammates if necessary.
        """

        messages = []

        self.communication_delay = np.maximum(self.communication_delay - 1, 0, dtype=np.int64)

        if len(self.message2send) > 0:
            for c in range(self.num_cameras):
                if c == self.index or self.communication_delay[c] > 0:
                    continue
                content = self.message2send.copy()
                if 'target_states' in content:
                    if c in self.neighboring_teammate_states and self.filterout_beyond_range:
                        teammate_state = self.neighboring_teammate_states[c]
                        threshold = self.range_factor * teammate_state.max_sight_range
                        content['target_states'] = [
                            ts
                            for ts in content['target_states']
                            if (ts - teammate_state).norm < threshold
                        ]
                        if len(content['target_states']) == 0:
                            del content['target_states']
                    else:
                        del content['target_states']
                if len(content) > 0:
                    messages.append(self.pack_message(recipient=c, content=content))
                    delay = self.np_random.randint(self.memory_period // 4, 2 * self.memory_period)
                    self.communication_delay[c] = delay

            self.message2send.clear()

        return messages

    def receive_responses(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_responses() but before act().

        Receive and process messages from teammates.
        """

        self.last_responses = tuple(messages)

        for message in self.last_responses:
            if 'state' in message.content:
                teammate_state = message.content['state']
                is_neighboring = True
                if self.filterout_beyond_range:
                    distance = (teammate_state - self.state).norm
                    threshold = (
                        self.state.max_sight_range
                        + self.range_factor * teammate_state.max_sight_range
                    )
                    is_neighboring = distance < threshold
                if is_neighboring:
                    self.neighboring_teammate_states[message.sender] = teammate_state
                elif message.sender in self.neighboring_teammate_states:
                    del self.neighboring_teammate_states[message.sender]
                self.neighboring_teammate_states[message.sender] = teammate_state

            for target_state in message.content.get('target_states', []):
                self.memory[target_state.index] = target_state
                self.time2forget[target_state.index] = self.memory_period
                if target_state.is_loaded:
                    self.never_loaded[target_state.index] = False


class GreedyTargetAgent(TargetAgentBase):  # pylint: disable=too-many-instance-attributes
    """Greedy Target Agent

    Arbitrarily runs towards the destination (desired warehouse) with some noise.
    """

    def __init__(self, seed=None, noise_scale=0.5):
        """Initialize the agent.
        This function will be called only once on initialization.
        """

        super().__init__(seed=seed)

        self.noise_scale = float(noise_scale)
        self.goal_bits = None
        self.prev_state = None
        self.prev_noise = None
        self.non_empty_warehouses = set(range(NUM_WAREHOUSES))
        self.need_communication = False

    @property
    def goal(self):
        """Index of the current warehouse."""

        if self.goal_bits is not None and self.goal_bits.any():
            return np.flatnonzero(self.goal_bits)[0]
        return None

    @property
    def goal_location(self):
        """Location of the current warehouse."""

        goal = self.goal
        if goal is not None:
            return WAREHOUSES[goal]
        return None

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().
        """

        super().reset(observation)

        self.prev_state = self.state
        self.prev_noise = 0.5 * self.action_space.sample()

        self.goal_bits = self.state.goal_bits.copy()

        self.non_empty_warehouses = set(range(NUM_WAREHOUSES))
        self.need_communication = False

    def observe(self, observation, info=None):
        """The agent observe the environment before sending messages.
        This function will be called before send_responses().
        """

        self.state, observation, info, messages = self.check_inputs(observation, info)

        self.process_messages(observation, messages)

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Arbitrarily run towards the warehouse with some noise.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        if self.state.goal_bits.any():
            self.goal_bits = self.state.goal_bits
        if self.goal is None or (
            not self.state.goal_bits.any() and self.goal not in self.non_empty_warehouses
        ):
            self.goal_bits = np.zeros_like(self.state.goal_bits)
            if len(self.non_empty_warehouses) > 0:
                new_goal = self.np_random.choice(list(self.non_empty_warehouses))
                self.goal_bits[new_goal] = 1

        prev_actual_action = self.state.location - self.prev_state.location

        if self.goal is not None:
            action = self.goal_location - self.state.location
        else:
            action = np.zeros_like(self.state.location)
        step_size = np.linalg.norm(action)
        if step_size > self.state.step_size:
            action *= self.state.step_size / step_size

        prob = 0.05 if np.linalg.norm(prev_actual_action) > 0.2 * self.state.step_size else 0.75
        if self.np_random.binomial(1, prob) != 0:
            noise = self.noise_scale * self.action_space.sample()
        else:
            noise = self.prev_noise

        action = (action + noise).clip(min=self.action_space.low, max=self.action_space.high)

        self.prev_state = self.state
        self.prev_noise = noise
        return action

    def process_messages(self, observation, messages):  # pylint: disable=unused-argument
        """Process observation and prepare messages to teammates."""

        seen_empty_warehouses = set(np.flatnonzero(self.state.empty_bits))
        if len(seen_empty_warehouses.intersection(self.non_empty_warehouses)) > 0:
            self.non_empty_warehouses.difference_update(seen_empty_warehouses)
            self.need_communication = True

    def send_responses(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called before receive_responses().

        Send indices of non-empty warehouses to teammate if necessary.
        """

        messages = []

        if self.need_communication:
            content = {'non_empty_warehouses': self.non_empty_warehouses.copy()}
            messages.append(self.pack_message(content=content))  # broadcasting
            self.need_communication = False

        return messages

    def receive_responses(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_responses() but before act().

        Receive and process messages from teammates.
        """

        self.last_responses = tuple(messages)

        for message in self.last_responses:
            self.non_empty_warehouses.intersection_update(message.content['non_empty_warehouses'])
