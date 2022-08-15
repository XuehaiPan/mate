# pylint: disable=missing-module-docstring

from mate.wrappers.message_filter import MessageFilter
from mate.wrappers.typing import MateEnvironmentType


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class NoCommunication(MessageFilter):
    """Disable intra-team communications, i.e., filter out all messages."""

    def __init__(
        self, env: MateEnvironmentType, team: Literal['both', 'camera', 'target', 'none'] = 'both'
    ) -> None:
        assert team in (
            'both',
            'camera',
            'target',
            'none',
        ), f'Invalid argument team {team!r}. Expect one of {("both", "camera", "target", "none")}.'

        self.team = team

        if self.team == 'both':
            super().__init__(env, filter=lambda unwrapped, message: False)  # filter out all
        elif self.team == 'none':
            super().__init__(env, filter=lambda unwrapped, message: True)  # do nothing
        else:
            super().__init__(
                env, filter=lambda unwrapped, message: message.team.name.lower() != self.team
            )

    def __str__(self) -> str:
        return f'<{type(self).__name__}(team={self.team}){self.env}>'
