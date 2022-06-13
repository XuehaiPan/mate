# pylint: disable=missing-module-docstring

import functools

from mate.utils import Message
from mate.wrappers.message_filter import MessageFilter
from mate.wrappers.typing import MateEnvironmentType


class RestrictedCommunicationRange(MessageFilter):
    """Add a restricted communication range to channels. (Not used in the evaluation script.)"""

    def __init__(self, env: MateEnvironmentType, range_limit: float) -> None:
        self.range_limit = range_limit

        super().__init__(env, filter=functools.partial(self.filter, range_limit=self.range_limit))

    @staticmethod
    def filter(env: MateEnvironmentType, message: Message, range_limit: float) -> bool:
        """Filter out messages beyond range limit."""

        entities = [env.cameras, env.targets][message.team.value]
        sender, recipient = entities[message.sender], entities[message.recipient]

        return recipient.distance(sender) <= range_limit
