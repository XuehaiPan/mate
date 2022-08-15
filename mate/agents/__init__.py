"""Built-in classes for agents."""

from mate.agents import greedy, heuristic, mixture, naive, random, utils
from mate.agents.base import CameraAgentBase, TargetAgentBase
from mate.agents.greedy import GreedyCameraAgent, GreedyTargetAgent
from mate.agents.heuristic import HeuristicCameraAgent, HeuristicTargetAgent
from mate.agents.mixture import MixtureCameraAgent, MixtureTargetAgent
from mate.agents.naive import NaiveCameraAgent, NaiveTargetAgent
from mate.agents.random import RandomCameraAgent, RandomTargetAgent
from mate.agents.utils import *


__all__ = [
    'CameraAgentBase',
    'TargetAgentBase',
    'RandomCameraAgent',
    'RandomTargetAgent',
    'NaiveCameraAgent',
    'NaiveTargetAgent',
    'GreedyCameraAgent',
    'GreedyTargetAgent',
    'HeuristicCameraAgent',
    'HeuristicTargetAgent',
    'MixtureCameraAgent',
    'MixtureTargetAgent',
]

__all__.extend(utils.__all__)
