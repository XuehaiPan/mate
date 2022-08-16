from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size


torch, nn = try_import_torch()


class MessageAggregator(nn.Module):
    def __init__(self, key_dim, value_dim, hidden_dim):
        super().__init__()

        self.key_dim = self.query_dim = key_dim
        self.value_dim = value_dim
        self.message_dim = self.key_dim + self.value_dim

        self.hidden_dim = hidden_dim

        self.query_predictor = nn.Linear(
            in_features=self.hidden_dim, out_features=self.query_dim, bias=False
        )
        self.scale = 1.0 / np.sqrt(self.query_dim)

    def forward(self, messages, hidden_states):
        # fmt: off
        assert messages.ndim == 3       # (B, T, Na * Dm)
        assert hidden_states.ndim == 3  # (B, T, Dh)
        B, T, joint_message_dim = messages.shape

        messages = messages.view(B, T, -1, self.message_dim)                   # (B, T, Na, Dm)
        keys, values = messages.split([self.key_dim, self.value_dim], dim=-1)  # (B, T, Na, *)

        queries = self.query_predictor(hidden_states)  # (B, T, Dq)
        queries = queries.unsqueeze(dim=-2)            # (B, T, 1, Dq)

        attns = self.scale * torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, 1, Na)
        attns = attns.softmax(dim=-1)                                       # (B, T, 1, Na)
        outputs = torch.matmul(attns, values)                               # (B, T, 1, Dv)
        outputs = outputs.squeeze(dim=-2)                                   # (B, T, Dv)
        # fmt: on

        return outputs


class TarMACModel(TorchRNN, nn.Module):
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
        # Extra TarMACModel arguments
        message_key_dim=32,
        message_value_dim=32,
        critic_use_global_state=True,
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
        assert isinstance(action_space, spaces.Dict) and tuple(action_space.keys())[-1] == 'message'
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
        self.joint_message_dim = self.space_dims['messages']
        self.joint_message_slice = self.flat_obs_slices['messages']

        self.action_space_dims = OrderedDict(
            [(key, get_space_flat_size(subspace)) for key, subspace in self.action_space.items()]
        )
        self.action_dim = self.action_space_dims['action']
        self.message_dim = self.action_space_dims['message']
        self.message_key_dim = self.message_query_dim = message_key_dim
        self.message_value_dim = message_value_dim
        assert self.message_dim == self.message_key_dim + self.message_value_dim
        assert self.joint_message_dim % self.message_dim == 0
        self.num_agents = self.joint_message_dim // self.message_dim

        if self.has_action_mask:
            self.action_mask_slice = self.flat_obs_slices['action_mask']
            assert self.space_dims['action_mask'] == num_outputs - self.message_dim
        else:
            self.action_mask_slice = None

        self.actor_hiddens = actor_hiddens or []
        self.critic_hiddens = critic_hiddens or list(self.actor_hiddens)
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation
        self.lstm_cell_size = lstm_cell_size

        self.message_aggregator = MessageAggregator(
            key_dim=self.message_key_dim,
            value_dim=self.message_value_dim,
            hidden_dim=self.lstm_cell_size * 2,
        )

        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim + self.message_value_dim,
            hidden_dims=self.actor_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=num_outputs,
            activation=self.actor_hidden_activation,
            output_activation=None,
        )

        self.critic_use_global_state = critic_use_global_state
        critic_obs_input_dim = (
            self.global_state_dim if self.critic_use_global_state else self.local_obs_dim
        )
        self.critic = SimpleRNN(
            name='critic',
            input_dim=critic_obs_input_dim + self.joint_message_dim,
            hidden_dims=self.critic_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=1,
            activation=self.critic_hidden_activation,
            output_activation=None,
        )

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        # fmt: off
        assert inputs.ndim == 3  # (B, T, *)
        B, T, flat_obs_dim = inputs.shape
        assert flat_obs_dim == self.flat_obs_dim

        local_obs = inputs[..., self.local_obs_slice]     # (B, T, Do)
        messages = inputs[..., self.joint_message_slice]  # (B, T, Na * Dm)

        action_out_list = []
        message_out_list = []
        hidden_states = state[:2]
        for t in range(T):
            aggregated_message = self.message_aggregator(messages[:, t:t + 1],                       # (B, 1, Dm)
                                                         torch.cat(hidden_states, dim=-1).unsqueeze(dim=1))
            local_obs_with_message = torch.cat((local_obs[:, t:t + 1], aggregated_message), dim=-1)  # (B, 1, Do + Dm)
            actor_out, hidden_states = self.actor(local_obs_with_message, hidden_states)             # (B, 1, Da + Dm)
            action_out, message_out = actor_out.split([self.action_dim, self.message_dim], dim=-1)   # (B, T, *)
            message_out = message_out.tanh()  # squash messages to [-1., +1.]
            action_out_list.append(action_out)
            message_out_list.append(message_out)
        actor_state_out = hidden_states
        action_out = torch.cat(action_out_list, dim=1)    # (B, T, Da)
        message_out = torch.cat(message_out_list, dim=1)  # (B, T, Dm)

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        if self.critic_use_global_state:
            global_state = inputs[..., self.global_state_slice]
            critic_inputs = torch.cat((global_state, messages), dim=-1)
        else:
            critic_inputs = torch.cat((local_obs, messages), dim=-1)
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(critic_inputs, critic_state_in, features_only=True)

        action_out = torch.cat((action_out, message_out), dim=-1)  # (B, 1, Da + Dm)
        # fmt: on
        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'

        return self.critic.output(self.critic.last_features).reshape(-1)

    def metrics(self):
        return {
            'num_in_comm_edges': self.num_agents,
        }


ModelCatalog.register_custom_model('TarMACModel', TarMACModel)
