from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.agents.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleMLP, get_space_flat_size, orthogonal_initializer


torch, nn = try_import_torch()


# Use sigmoid to scale to [0, 1], but also double magnitude of input to emulate
# the behavior of tanh activation used in DDPG and TD3 papers.
# After sigmoid squashing, re-scale to env action space bounds.
# Need to set "normalize_actions" to False in policy configuration.
class SquashAction(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        low_action = torch.from_numpy(action_space.low).float()
        action_range = torch.from_numpy(action_space.high - action_space.low).float()

        self.register_buffer('low_action', low_action)
        self.register_buffer('action_range', action_range)

    def forward(self, x):
        sigmoid_out = torch.sigmoid(2.0 * x)
        squashed = self.action_range * sigmoid_out + self.low_action
        return squashed


class MADDPGModel(DDPGTorchModel, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        # Extra MADDPGModel arguments
        actor_hiddens=None,
        actor_hidden_activation='tanh',
        critic_hiddens=None,
        critic_hidden_activation='tanh',
        twin_q=False,
        add_layer_norm=False,
        **kwargs,
    ):
        if actor_hiddens is None:
            actor_hiddens = [256, 256]

        if critic_hiddens is None:
            critic_hiddens = [256, 256]

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        assert hasattr(obs_space, 'original_space') and isinstance(
            obs_space.original_space, spaces.Dict
        )
        assert isinstance(self.action_space, spaces.Box)

        original_space = obs_space.original_space
        self.local_obs_space = original_space['obs']
        self.global_state_space = original_space['state']

        self.flat_obs_dim = get_space_flat_size(self.obs_space)
        self.space_dims = OrderedDict(
            [(key, get_space_flat_size(subspace)) for key, subspace in original_space.items()]
        )
        indices = np.cumsum([0, *self.space_dims.values()])
        self.flat_obs_slices = OrderedDict(
            [
                (key, slice(indices[i], indices[i + 1]))
                for i, key in enumerate(self.space_dims.keys())
            ]
        )

        self.local_obs_dim = self.space_dims['obs']
        self.local_obs_slice = self.flat_obs_slices['obs']
        self.global_state_dim = self.space_dims['state']
        self.global_state_slice = self.flat_obs_slices['state']

        self.action_dim = get_space_flat_size(self.action_space)
        # The action time step is shifted with callback `ShiftAgentActionTimestep`
        self.others_joint_action_dim = self.space_dims['prev_others_joint_action']
        self.others_joint_action_slice = self.flat_obs_slices['prev_others_joint_action']
        self.global_state_action_dim = (
            self.global_state_dim + self.others_joint_action_dim + self.action_dim
        )

        self.actor_hiddens = actor_hiddens or []
        self.critic_hiddens = critic_hiddens or list(self.actor_hiddens)
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation

        policy_model = SimpleMLP(
            name='actor',
            input_dim=self.local_obs_dim,
            hidden_dims=self.actor_hiddens,
            output_dim=self.action_dim,
            layer_norm=add_layer_norm,
            activation=self.actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=0.01),
        )
        if self.action_space.is_bounded('both'):
            self.policy_model = nn.Sequential(policy_model, SquashAction(self.action_space))
        else:
            self.policy_model = policy_model

        q_model_kwargs = {
            'name': 'critic',
            'input_dim': self.global_state_action_dim,
            'hidden_dims': self.critic_hiddens,
            'output_dim': 1,
            'layer_norm': add_layer_norm,
            'activation': self.critic_hidden_activation,
            'output_activation': None,
            'hidden_weight_initializer': orthogonal_initializer(scale=1.0),
            'output_weight_initializer': orthogonal_initializer(scale=1.0),
        }
        self.q_model = SimpleMLP(**q_model_kwargs)
        if twin_q:
            q_model_kwargs.update(name='twin_critic')
            self.twin_q_model = SimpleMLP(**q_model_kwargs)
        else:
            self.twin_q_model = None

    def get_q_values(self, obs_flat, actions, *, q_model=None):
        assert obs_flat.size(-1) == self.flat_obs_dim

        if q_model is None:
            q_model = self.q_model

        global_state = obs_flat[..., self.global_state_slice]
        others_joint_action = obs_flat[..., self.others_joint_action_slice]
        return self.q_model(torch.cat((global_state, others_joint_action, actions), -1))

    def get_twin_q_values(self, obs_flat, actions):
        return self.get_q_values(obs_flat, actions, q_model=self.twin_q_model)

    def get_policy_output(self, obs_flat):
        assert obs_flat.size(-1) == self.flat_obs_dim

        local_obs = obs_flat[..., self.local_obs_slice]
        return self.policy_model(local_obs)

    def policy_variables(self, as_dict: bool = False):
        if as_dict:
            return self.policy_model.state_dict()
        return list(self.policy_model.parameters())

    def q_variables(self, as_dict=False):
        if as_dict:
            return {
                **self.q_model.state_dict(),
                **(
                    self.twin_q_model.state_dict(prefix='twin_q_model')
                    if self.twin_q_model is not None
                    else {}
                ),
            }
        return [
            *tuple(self.q_model.parameters()),
            *tuple(self.twin_q_model.parameters() if self.twin_q_model is not None else []),
        ]

    def forward(self, input_dict, state, seq_lens):
        """Do nothing. The actual forward operations are `get_policy_output` and `get_q_values`."""

        return input_dict['obs_flat'].float(), state


ModelCatalog.register_custom_model('MADDPGModel', MADDPGModel)
