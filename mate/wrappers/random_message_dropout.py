# pylint: disable=missing-module-docstring

import functools

from mate.utils import Message
from mate.wrappers.message_filter import MessageFilter
from mate.wrappers.typing import MateEnvironmentType


class RandomMessageDropout(MessageFilter):
    """Randomly drop messages in communication channels. (Not used in the evaluation script.)"""

    def __init__(self, env: MateEnvironmentType, dropout_rate: float) -> None:
        assert 0.0 <= dropout_rate <= 1.0, (
            f'Dropout rate must be a float number between 0 and 1. '
            f'Got dropout_rate = {dropout_rate}.'
        )

        self.dropout_rate = dropout_rate

        super().__init__(env, filter=functools.partial(self.filter, dropout_rate=self.dropout_rate))

    @staticmethod
    # pylint: disable-next=unused-argument
    def filter(env: MateEnvironmentType, message: Message, dropout_rate: float) -> bool:
        """Randomly drop messages."""

        return not env.np_random.binomial(1, dropout_rate)
