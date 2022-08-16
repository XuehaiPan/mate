from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size, orthogonal_initializer


torch, nn = try_import_torch()


class MAPPOModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        # Extra MAPPOModel arguments
        actor_hiddens=None,
        actor_hidden_activation='tanh',
        critic_hiddens=None,
        critic_hidden_activation='tanh',
        lstm_cell_size=256,
        **kwargs,
    ):
        if actor_hiddens is None:
            actor_hiddens = [256, 256]

        if critic_hiddens is None:
            critic_hiddens = [256, 256]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        assert hasattr(obs_space, 'original_space') and isinstance(
            obs_space.original_space, spaces.Dict
        )
        original_space = obs_space.original_space
        self.local_obs_space = original_space['obs']
        self.global_state_space = original_space['state']
        if 'action_mask' in original_space.spaces:
            self.action_mask_space = original_space['action_mask']
            self.has_action_mask = True
        else:
            self.action_mask_space = None
            self.has_action_mask = False

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

        if self.has_action_mask:
            self.action_mask_slice = self.flat_obs_slices['action_mask']
            assert self.space_dims['action_mask'] == num_outputs
        else:
            self.action_mask_slice = None

        self.actor_hiddens = actor_hiddens or []
        self.critic_hiddens = critic_hiddens or list(self.actor_hiddens)
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation
        self.lstm_cell_size = lstm_cell_size

        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim,
            hidden_dims=self.actor_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=num_outputs,
            activation=self.actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=0.01),
        )

        self.critic = SimpleRNN(
            name='critic',
            input_dim=self.global_state_dim,
            hidden_dims=self.critic_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=1,
            activation=self.critic_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        assert inputs.size(-1) == self.flat_obs_dim

        local_obs = inputs[..., self.local_obs_slice]
        actor_state_in = state[:2]
        action_out, actor_state_out = self.actor(local_obs, actor_state_in)

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        global_state = inputs[..., self.global_state_slice]
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(global_state, critic_state_in, features_only=True)

        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'

        return self.critic.output(self.critic.last_features).reshape(-1)


ModelCatalog.register_custom_model('MAPPOModel', MAPPOModel)
