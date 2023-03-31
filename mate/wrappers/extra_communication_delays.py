# pylint: disable=missing-module-docstring

import heapq
from typing import Callable, Iterable, Tuple, Union

import gym
import numpy as np

from mate.utils import Message
from mate.wrappers.typing import MateEnvironmentType, WrapperMeta, assert_mate_environment


class ExtraCommunicationDelays(gym.Wrapper, metaclass=WrapperMeta):
    """Add extra message delays to communication channels. (Not used in the evaluation script.)

    Users can use this wrapper to implement a communication channel with random delays.
    """

    def __init__(
        self,
        env: MateEnvironmentType,
        delay: Union[int, Callable[[MateEnvironmentType, Message], int]] = 3,
    ) -> None:
        assert_mate_environment(env)
        assert callable(delay) or (isinstance(delay, int) and delay > 0), (
            f'The argument `delay` should be a callable function or a constant positive integer. '
            f'Got delay = {delay}.'
        )

        super().__init__(env)

        # A function with signature: (env, message) -> int
        # or a constant positive integer.
        self.delay = delay

        self.heap = []

    def reset(self, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        self.heap = []

        return self.env.reset(**kwargs)

    def send_messages(self, messages: Union[Message, Iterable[Message]]) -> None:
        """Buffer the messages from an agent to others in the same team.

        The environment will send the messages to recipients' through method
        receive_messages(), and also info field of step() results.
        """

        if isinstance(messages, Message):
            messages = (messages,)

        messages = list(messages)
        assert (
            len({m.team for m in messages}) <= 1
        ), f'All messages must be from the same team. Got messages = {messages}.'

        for message in messages:
            if callable(self.delay):
                delay = self.delay(self.unwrapped, message)
            else:
                delay = self.delay

            heapq.heappush(self.heap, (self.episode_step + delay, message))

        messages = []
        while len(self.heap) > 0 and self.heap[0][0] <= self.episode_step:
            _, message = heapq.heappop(self.heap)
            messages.append(message)

        self.env.send_messages(messages)
