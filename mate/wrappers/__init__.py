"""Wrapper classes for the Multi-Agent Tracking Environment."""

from mate.wrappers import typing
from mate.wrappers.auxiliary_camera_rewards import AuxiliaryCameraRewards
from mate.wrappers.auxiliary_target_rewards import AuxiliaryTargetRewards
from mate.wrappers.discrete_action_spaces import DiscreteCamera, DiscreteTarget
from mate.wrappers.enhanced_observation import EnhancedObservation
from mate.wrappers.extra_communication_delays import ExtraCommunicationDelays
from mate.wrappers.message_filter import MessageFilter
from mate.wrappers.more_training_information import MoreTrainingInformation
from mate.wrappers.no_communication import NoCommunication
from mate.wrappers.random_message_dropout import RandomMessageDropout
from mate.wrappers.relative_coordinates import RelativeCoordinates
from mate.wrappers.render_communication import RenderCommunication
from mate.wrappers.repeated_reward_individual_done import RepeatedRewardIndividualDone
from mate.wrappers.rescaled_observation import RescaledObservation
from mate.wrappers.restricted_communication_range import RestrictedCommunicationRange
from mate.wrappers.shared_field_of_view import SharedFieldOfView
from mate.wrappers.single_team import (
    MultiCamera,
    MultiTarget,
    SingleCamera,
    SingleTarget,
    group_act,
    group_communicate,
    group_observe,
    group_reset,
    group_step,
)
from mate.wrappers.typing import WrapperMeta, WrapperSpec


__all__ = [
    # Observation
    'EnhancedObservation',
    'SharedFieldOfView',
    'RescaledObservation',
    'RelativeCoordinates',
    'MoreTrainingInformation',
    # Action
    'DiscreteCamera',
    'DiscreteTarget',
    # Reward
    'AuxiliaryCameraRewards',
    'AuxiliaryTargetRewards',
    # Single team
    'group_reset',
    'group_step',
    'group_observe',
    'group_communicate',
    'group_act',
    'MultiCamera',
    'SingleCamera',
    'MultiTarget',
    'SingleTarget',
    # Communication
    'MessageFilter',
    'RestrictedCommunicationRange',
    'RandomMessageDropout',
    'NoCommunication',
    'ExtraCommunicationDelays',
    'RenderCommunication',
    # Miscellaneous
    'RepeatedRewardIndividualDone',
    'WrapperMeta',
    'WrapperSpec',
]
