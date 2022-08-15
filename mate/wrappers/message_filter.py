# pylint: disable=missing-module-docstring

from functools import partial
from typing import Callable, Iterable, Union

import gym

from mate.utils import Message
from mate.wrappers.typing import MateEnvironmentType, WrapperMeta, assert_mate_environment


class MessageFilter(gym.Wrapper, metaclass=WrapperMeta):
    """Filter messages from agents of intra-team communications. (Not used in the evaluation script.)

    Users can use this wrapper to implement a communication channel with limited
    bandwidth, limited communication range, or random dropout. This wrapper can
    be applied multiple times with different filter functions.

    Note:
        The filter function can also modify the message content. Users can use
        this to add channel signal noises etc.
    """

    def __init__(
        self,
        env: MateEnvironmentType,
        filter: Callable[[MateEnvironmentType, Message], bool],  # pylint: disable=redefined-builtin
    ) -> None:
        assert_mate_environment(env)
        assert callable(
            filter
        ), f'The argument `filter` should be a callable function. Got filter = {filter!r}.'

        super().__init__(env)

        # A function with signature: (env, message) -> bool
        self._filter = partial(filter, self.unwrapped)

    def send_messages(self, messages: Union[Message, Iterable[Message]]) -> None:
        """Buffer the messages from an agent to others in the same team.

        The environment will send the messages to recipients' through method
        receive_messages(), and also info field of step() results.
        """

        if isinstance(messages, Message):
            messages = (messages,)

        messages = list(filter(self._filter, self.route_messages(messages)))
        self.env.send_messages(messages)
