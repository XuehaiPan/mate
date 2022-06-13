# pylint: disable=missing-module-docstring

from typing import TypeVar, Union, TYPE_CHECKING

from mate.agents.base import CameraAgentBase, TargetAgentBase
from mate.environment import MultiAgentTracking, EnvMeta as WrapperMeta


__all__ = ['WrapperMeta', 'WrapperSpec',
           'AgentType', 'BaseEnvironmentType', 'assert_base_environment',
           'MultiAgentEnvironmentType', 'assert_multi_agent_environment',
           'MateEnvironmentType', 'assert_mate_environment']


if TYPE_CHECKING:
    from mate.wrappers import (EnhancedObservation, MoreTrainingInformation,
                               RescaledObservation, RelativeCoordinates,
                               DiscreteCamera, DiscreteTarget,
                               MultiCamera, MultiTarget,
                               NoCommunication, RenderCommunication,
                               RepeatedRewardIndividualDone)

AgentType = TypeVar('AgentType', CameraAgentBase, TargetAgentBase)
BaseEnvironmentType = TypeVar('BaseEnvironmentType',
                              MultiAgentTracking,
                              'EnhancedObservation', 'MoreTrainingInformation',
                              'RescaledObservation', 'RelativeCoordinates',
                              'DiscreteCamera', 'DiscreteTarget',
                              'NoCommunication', 'RenderCommunication',
                              'RepeatedRewardIndividualDone')
MultiAgentEnvironmentType = Union[BaseEnvironmentType, 'MultiCamera', 'MultiTarget']
MateEnvironmentType = MultiAgentTracking


def assert_mate_environment(env):  # pylint: disable=missing-function-docstring
    assert isinstance(env.unwrapped, MultiAgentTracking), (
        f'The unwrapped environment should be the Multi-Agent Tracking Environment. '
        f'Got env.unwrapped = {env.unwrapped}.'
    )
    assert isinstance(env, MultiAgentTracking), (
        f"You should wrap mate's built-in wrappers before yours. "
        f'Got env = {env}.'
    )


def assert_multi_agent_environment(env):  # pylint: disable=missing-function-docstring
    from mate.wrappers.single_team import SingleTeamSingleAgent  # pylint: disable=import-outside-toplevel

    assert_mate_environment(env)
    assert not isinstance(env, SingleTeamSingleAgent), (
        f'You should provide a multi-agent environment. '
        f'Got env = {env}.'
    )


def assert_base_environment(env):  # pylint: disable=missing-function-docstring
    from mate.wrappers.single_team import SingleTeamHelper  # pylint: disable=import-outside-toplevel

    assert_multi_agent_environment(env)
    assert not isinstance(env, SingleTeamHelper), (
        f'You should provide a instance of the basic setting of the Multi-Agent Tracking Environment, '
        f'i.e., two teams and multiple agents. '
        f'Got env = {env}.'
    )


class WrapperSpec:  # pylint: disable=too-few-public-methods
    """Helper class for creating environments with wrappers."""

    def __init__(self, wrapper, *args, **kwargs):
        assert callable(wrapper), (
            f'The argument `wrapper` should be a callable object. '
            f'Got wrapper = {wrapper:!r}.'
        )

        self.wrapper = wrapper
        self.args = args
        self.kwargs = kwargs

    def __call__(self, env):
        return self.wrapper(env, *self.args, **self.kwargs)
