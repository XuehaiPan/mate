import functools

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import (
    TorchDeterministic,
    TorchMultiActionDistribution,
)
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()


class DeterministicMessage(TorchDeterministic):
    def logp(self, actions):
        return 0.0

    def kl(self, other):
        return 0.0

    def entropy(self):
        return 0.0

    def deterministic_sample(self):
        return self.inputs

    def sampled_action_logp(self):
        return self.inputs.new(self.inputs.size(0)).zero_()

    def sample(self):
        return self.deterministic_sample()

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)


class ActionDistributionWithMessage(TorchMultiActionDistribution):
    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        assert isinstance(action_space, spaces.Dict)
        assert tuple(action_space.keys())[-1] == 'message' and isinstance(
            action_space['message'], spaces.Box
        )

        message_space = action_space['message']

        child_distributions = [*child_distributions[:-1], DeterministicMessage]
        input_lens = [*input_lens[:-1], np.prod(message_space.shape)]

        super().__init__(
            inputs,
            model,
            child_distributions=child_distributions,
            input_lens=input_lens,
            action_space=action_space,
        )

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        assert isinstance(action_space, spaces.Dict)
        assert tuple(action_space.keys())[-1] == 'message' and isinstance(
            action_space['message'], spaces.Box
        )

        message_space = action_space['message']

        input_lens = [
            dist_dim
            for dist_class, dist_dim in (
                ModelCatalog.get_action_dist(space, config=model_config, framework='torch')
                for space in action_space.values()
            )
        ]

        input_lens[-1] = DeterministicMessage.required_model_output_shape(
            message_space, model_config
        )

        return int(sum(input_lens))


ModelCatalog.register_custom_action_dist(
    'action_distribution_with_message', ActionDistributionWithMessage
)


def _fix_custom_multi_action_distribution(func):
    @functools.wraps(func)
    def fixed(dist_class, action_space, config, framework):
        dist_class, dist_dim = func(
            dist_class, action_space=action_space, config=config, framework=framework
        )

        if (
            isinstance(dist_class, functools.partial)
            and dist_class.func is ActionDistributionWithMessage
        ):
            dist_dim = dist_class.func.required_model_output_shape(action_space, config)

        return dist_class, dist_dim

    return fixed


ModelCatalog._get_multi_action_distribution = _fix_custom_multi_action_distribution(
    ModelCatalog._get_multi_action_distribution
)


del _fix_custom_multi_action_distribution
