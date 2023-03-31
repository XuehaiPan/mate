"""MATE: The Multi-Agent Tracking Environment."""

import os

import gym

from mate import agents, constants, environment, utils, wrappers
from mate.agents import *
from mate.constants import *
from mate.environment import *
from mate.utils import *
from mate.version import __version__
from mate.wrappers import *


__all__ = ['make']
__all__.extend(constants.__all__)
__all__.extend(environment.__all__)
__all__.extend(wrappers.__all__)
__all__.extend(agents.__all__)
__all__.extend(utils.__all__)


make = gym.make


def make_environment(config=None, wrappers=(), **kwargs):  # pylint: disable=redefined-outer-name
    """Helper function for creating a wrapped environment."""

    env = MultiAgentTracking(config, **kwargs)

    for wrapper in wrappers:
        assert (
            isinstance(wrapper, WrapperSpec)
            or callable(wrapper)
            or issubclass(wrapper, gym.Wrapper)
        ), (
            f'You should provide a wrapper class or an instance of `mate.WrapperSpec`. '
            f'Got wrapper = {wrapper!r}.'
        )
        env = wrapper(env)

    return env


gym.register(id='MultiAgentTracking-v0', entry_point=make_environment)
gym.register(id='MATE-v0', entry_point=make_environment)

gym.register(
    id='MATE-4v2-9-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-4v2-9.yaml')},
)

gym.register(
    id='MATE-4v2-0-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-4v2-0.yaml')},
)

gym.register(
    id='MATE-4v4-9-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-4v4-9.yaml')},
)

gym.register(
    id='MATE-4v4-0-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-4v4-0.yaml')},
)

gym.register(
    id='MATE-4v8-9-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-4v8-9.yaml')},
)

gym.register(
    id='MATE-4v8-0-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-4v8-0.yaml')},
)

gym.register(
    id='MATE-8v8-9-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-8v8-9.yaml')},
)

gym.register(
    id='MATE-8v8-0-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-8v8-0.yaml')},
)

gym.register(
    id='MATE-Navigation-v0',
    entry_point=make_environment,
    kwargs={'config': (ASSETS_DIR / 'MATE-Navigation.yaml')},
)


del os, gym
