from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import one_hot, sequence_mask

from examples.utils import (
    SimpleMLP,
    SimpleRNN,
    get_preprocessor,
    get_space_flat_size,
    orthogonal_initializer,
)


torch, nn = try_import_torch()


class MessageAggregator(nn.Module):
    def __init__(self, key_dim, query_dim, value_dim, output_dim):
        super().__init__()

        self.key_dim = key_dim
        self.query_dim = self.embed_dim = query_dim
        self.value_dim = value_dim

        self.output_dim = output_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.value_dim,
            num_heads=1,
            bias=True,
            add_zero_attn=True,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=self.embed_dim, out_features=self.output_dim, bias=True)

    def forward(self, queries, keys, values, attn_mask=None):
        # fmt: off
        assert queries.ndim == 4 and keys.ndim == 4 and values.ndim == 4
        B, T, Nam1 = keys.shape[:3]
        queries = queries.view(B * T, 1, -1)      # (B * T, 1, Dq)
        keys = keys.view(B * T, Nam1, -1)         # (B * T, Na - 1, Dk)
        values = values.view(B * T, Nam1, -1)     # (B * T, Na - 1, Dv)
        attn_mask = attn_mask.view(B * T, 1, -1)  # (B * T, 1, Na - 1)

        attn_output, attn_output_weights = self.attn(queries, keys, values, attn_mask=attn_mask)  # (B * T, 1, De)
        attn_output = attn_output.view(B, T, -1)  # (B, T, De)

        output = self.linear(attn_output)  # (B, T, Do)
        # fmt: on
        return output


class I2CModel(TorchRNN, nn.Module):
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
        # Extra I2CModel arguments
        message_dim=64,
        policy_corr_reg_coeff=0.01,
        temperature=0.1,
        prior_buffer_size=100000,
        prior_percentile=50,
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
        assert isinstance(self.action_space, spaces.Discrete)

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
        self.others_joint_obs_dim = self.space_dims['others_joint_observation']
        self.others_joint_obs_slice = self.flat_obs_slices['others_joint_observation']
        assert self.others_joint_obs_dim % self.local_obs_dim == 0
        self.num_agents = self.others_joint_obs_dim // self.local_obs_dim + 1

        self.action_preprocessor = get_preprocessor(self.action_space)
        self.action_dim = self.action_preprocessor.size
        # The action time step is shifted with callback `ShiftAgentActionTimestep`
        self.others_joint_action_dim = self.space_dims['prev_others_joint_action']
        self.others_joint_action_slice = self.flat_obs_slices['prev_others_joint_action']

        all_possible_actions = np.asarray(
            list(map(self.action_preprocessor.transform, range(self.action_space.n)))
        )
        self.num_all_possible_actions = len(all_possible_actions)
        all_possible_actions = torch.from_numpy(all_possible_actions)
        self.register_buffer('all_possible_actions', all_possible_actions)

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

        self.message_dim = message_dim
        self.message_aggregator = MessageAggregator(
            key_dim=self.local_obs_dim,
            query_dim=self.local_obs_dim,
            value_dim=self.local_obs_dim,
            output_dim=self.message_dim,
        )

        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim + self.message_dim,
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

        self.joint_q_network = SimpleMLP(
            name='joint_q_value',
            input_dim=self.global_state_dim + self.action_dim + self.others_joint_action_dim,
            hidden_dims=self.critic_hiddens,
            output_dim=1,
            activation=self.critic_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )

        self.policy_corr_reg_coeff = policy_corr_reg_coeff
        self.temperature = temperature

        self.prior_network = SimpleMLP(
            name='prior_network',
            input_dim=self.local_obs_dim + (self.num_agents - 1),
            hidden_dims=self.actor_hiddens,
            output_dim=1,
            activation=self.actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )
        self.prior_percentile = prior_percentile
        self.register_buffer(
            'agent_ids', torch.from_numpy(np.eye(self.num_agents - 1, dtype=np.float32))
        )
        self.prior_buffer = deque(maxlen=prior_buffer_size)
        self.register_buffer('prior_threshold', torch.tensor(0.0, dtype=torch.float32))

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    @torch.no_grad()
    def get_communication_mask(self, observation):
        # fmt: off
        observation = torch.from_numpy(observation).float().to(self.prior_threshold.device)  # (Do,)
        repeated_observation = observation.expand(self.num_agents - 1, -1)                   # (Na - 1, Do)
        prior_input = torch.cat((repeated_observation, self.agent_ids), dim=-1)              # (Na - 1, Do + Na - 1)
        comm_mask_logits = self.prior_network(prior_input)                                   # (Na - 1, Do + Na - 1)
        comm_mask = (comm_mask_logits >= 0.0).squeeze(dim=-1).bool()                         # (Na - 1,)
        comm_mask = comm_mask.cpu().numpy()                                                  # (Na - 1,)
        # fmt: on
        return comm_mask

    def forward_rnn(self, inputs, state, seq_lens):
        # fmt: off
        assert inputs.ndim == 3  # (B, T, *)
        B, T, flat_obs_dim = inputs.shape
        assert flat_obs_dim == self.flat_obs_dim

        Na = self.num_agents

        local_obs = inputs[..., self.local_obs_slice]                             # (B, T, Do)
        repeated_local_obs = local_obs.unsqueeze(dim=-2)                          # (B, T, 1, Do)
        repeated_local_obs = repeated_local_obs.expand(B, T, Na - 1, -1)          # (B, T, Na - 1, Do)
        batched_agent_ids = self.agent_ids.expand(B, T, -1, -1)                   # (B, T, Na - 1, Na - 1)
        prior_input = torch.cat((repeated_local_obs, batched_agent_ids), dim=-1)  # (B, T, Na - 1, Do + Na - 1)
        comm_mask_logits = self.prior_network(prior_input).squeeze(dim=-1)        # (B, T, Na - 1)
        comm_mask = (comm_mask_logits >= 0.0).unsqueeze(dim=-2)                   # (B, T, 1, Na - 1)

        others_joint_obs = inputs[..., self.others_joint_obs_slice]               # (B, T, (Na - 1) * Do)
        keys = values = others_joint_obs.view(B, T, Na - 1, -1)                   # (B, T, Na - 1, Do)
        query = local_obs.unsqueeze(dim=-2)                                       # (B, T, 1, Do)
        aggregated_message = self.message_aggregator(query, keys, values,         # (B, T, Dm)
                                                     attn_mask=comm_mask.logical_not())  # True for invalid

        actor_input = torch.cat((local_obs, aggregated_message), dim=-1)

        actor_state_in = state[:2]
        action_out, actor_state_out = self.actor(actor_input, actor_state_in)

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        global_state = inputs[..., self.global_state_slice]
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(global_state, critic_state_in, features_only=True)
        # fmt: on
        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'

        return self.critic.output(self.critic.last_features).reshape(-1)

    def custom_loss(self, policy_loss, loss_inputs):
        # fmt: off
        B, T = self.actor.last_features.shape[:2]
        Na = self.num_agents
        Nac = self.num_all_possible_actions
        Da = self.action_dim

        mask = sequence_mask(
            loss_inputs[SampleBatch.SEQ_LENS], maxlen=T, time_major=self.is_time_major()
        )
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t, additional_mask=None):
            if additional_mask is None:
                return torch.sum(t[mask]) / num_valid

            m1 = mask
            m2 = additional_mask
            while m1.ndim < t.ndim:
                m1 = m1.unsqueeze(dim=-1)
            while m2.ndim < t.ndim:
                m2 = m2.unsqueeze(dim=-1)

            m1 = torch.broadcast_to(m1, t.size())
            m2 = torch.broadcast_to(m2, t.size())
            additional_mask = torch.logical_and(m1, m2)
            if additional_mask.any().item():
                return torch.sum(t[additional_mask]) / torch.sum(additional_mask)
            return t.new(1).zero_()[0]

        global_obs = loss_inputs[SampleBatch.OBS]                                   # (B * T, *)
        action = one_hot(loss_inputs[SampleBatch.ACTIONS],                          # (B, T, Da)
                         self.action_space).float().view(B, T, -1)

        global_state = global_obs[..., self.global_state_slice].view(B, T, -1)      # (B, T, Ds)
        others_joint_action = global_obs[..., self.others_joint_action_slice]       # (B * T, (Na - 1) * Da)
        others_joint_action = others_joint_action.view(B, T, -1)                    # (B, T, (Na - 1) * Da)

        joint_state_action = torch.cat((global_state, others_joint_action, action), dim=-1)
        joint_q_value = self.joint_q_network(joint_state_action)                    # (B, T, 1)
        value_target = loss_inputs[Postprocessing.VALUE_TARGETS].view(B, T, -1)     # (B, T, 1)
        q_losses = torch.square(joint_q_value - value_target)

        all_possible_actions = self.all_possible_actions.expand(B, T, -1, -1)       # (B, T, Nac, Da)

        repeated_others_joint_action = others_joint_action.unsqueeze(dim=-2)        # (B, T, 1, (Na - 1) * Da)
        repeated_others_joint_action = repeated_others_joint_action.expand(-1, -1, Nac, -1)

        repeated_global_state = global_state.unsqueeze(dim=-2)                      # (B, T, 1, Ds)
        repeated_global_state = repeated_global_state.expand(-1, -1, Nac, -1)       # (B, T, Nac, Ds)

        q_values_i = self.joint_q_network(torch.cat((repeated_global_state,         # (B, T, Nac, 1)
                                                     repeated_others_joint_action,
                                                     all_possible_actions),
                                                    dim=-1))
        log_probs_i = (q_values_i.squeeze(dim=-1) / self.temperature).log_softmax(dim=-1)    # (B, T, Nac)

        self_all_possible_actions = all_possible_actions.unsqueeze(dim=-2)                   # (B, T, Naci, 1, Da)
        self_all_possible_actions = self_all_possible_actions.expand(B, T, -1, Nac, -1)      # (B, T, Naci, Nacj, Da)
        others_all_possible_actions = all_possible_actions.unsqueeze(dim=-3)                 # (B, T, 1, Nacj, Da)
        others_all_possible_actions = others_all_possible_actions.expand(B, T, Nac, -1, -1)  # (B, T, Naci, Nacj, Da)
        repeated_global_state = global_state.unsqueeze(dim=-2).unsqueeze(dim=-2)             # (B, T, 1, 1, Ds)
        repeated_global_state = repeated_global_state.expand(-1, -1, Nac, Nac, -1)           # (B, T, Naci, Nacj, Ds)
        repeated_others_joint_action = repeated_others_joint_action.unsqueeze(dim=-2)        # (B, T, Nac, 1, Doja)
        repeated_others_joint_action = repeated_others_joint_action.expand(-1, -1, Nac, Nac, -1)

        KL_values = []
        with torch.no_grad():
            for j in range(1, Na):
                pre_joint_action = repeated_others_joint_action[..., :(j - 1) * Da]  # (B, T, Naci, Nacj, Da * (j - 1))
                post_joint_action = repeated_others_joint_action[..., j * Da:]       # (B, T, Naci, Nacj, Da * (Na - j))
                enum_joint_action = torch.cat((pre_joint_action,                     # (B, T, Naci, Nacj, Da * (Na - 1))
                                               others_all_possible_actions,
                                               post_joint_action),
                                              dim=-1)
                q_values_ij = self.joint_q_network(torch.cat((repeated_global_state,  # (B, T, Naci, Nacj, 1)
                                                              enum_joint_action,
                                                              self_all_possible_actions),
                                                             dim=-1))
                q_values_ij = q_values_ij.squeeze(dim=-1)  # (B, T, Naci, Nacj)
                logexp_soft_q_i_sumup_j = (q_values_ij / self.temperature).logsumexp(dim=-1)  # (B, T, Naci)
                log_probs_i_sumup_j = logexp_soft_q_i_sumup_j.log_softmax(dim=-1)             # (B, T, Naci)

                KL = torch.kl_div(log_probs_i_sumup_j, target=log_probs_i, log_target=True).sum(dim=-1)  # (B, T)
                KL_values.append(KL)

            KL_values = torch.stack(KL_values, dim=-1)  # (B, T, Na - 1)
            KL_values_valid = KL_values[mask].detach().cpu().numpy().ravel()
            self.prior_buffer.extend(KL_values_valid)
            self.prior_threshold[...] = np.percentile(self.prior_buffer, self.prior_percentile)

        prior_logits = self.prior_network.last_output.squeeze(dim=-1)                # (B, T, Na - 1)
        prior_labels = (KL_values >= self.prior_threshold).float()                   # (B, T, Na - 1)
        prior_losses = nn.functional.binary_cross_entropy_with_logits(prior_logits,  # (B, T, Na - 1)
                                                                      target=prior_labels,
                                                                      reduction='none')

        policy_corr_KLs = torch.kl_div(log_probs_i, target=self.actor.last_output.log_softmax(dim=1),  # (B, T)
                                       log_target=True).sum(dim=-1)

        num_in_comm_edges = reduce_mean_valid(prior_logits >= 0.0)
        q_loss = reduce_mean_valid(q_losses)
        prior_loss = reduce_mean_valid(prior_losses)
        policy_corr_reg_loss = reduce_mean_valid(policy_corr_KLs)
        additional_loss = q_loss + prior_loss + self.policy_corr_reg_coeff * policy_corr_reg_loss

        self.num_in_comm_edges_metric = num_in_comm_edges.item()
        self.KL_values_min_metric = np.min(self.prior_buffer)
        self.KL_values_mean_metric = np.mean(self.prior_buffer)
        self.KL_values_max_metric = np.max(self.prior_buffer)
        self.prior_threshold_metric = self.prior_threshold.item()

        self.q_loss_metric = q_loss.item()
        self.prior_loss_metric = prior_loss.item()
        self.policy_corr_reg_loss_metric = policy_corr_reg_loss.item()
        self.additional_loss_metric = additional_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])
        # fmt: on
        return [loss + additional_loss for loss in policy_loss]

    def metrics(self):
        return {
            'num_in_comm_edges': self.num_in_comm_edges_metric,
            'KL_values_min': self.KL_values_min_metric,
            'KL_values_mean': self.KL_values_mean_metric,
            'KL_values_max': self.KL_values_max_metric,
            'prior_threshold': self.prior_threshold_metric,
            'q_loss': self.q_loss_metric,
            'prior_loss': self.prior_loss_metric,
            'policy_corr_reg_loss': self.policy_corr_reg_loss_metric,
            'additional_loss': self.additional_loss_metric,
            'policy_loss': self.policy_loss_metric,
        }


ModelCatalog.register_custom_model('I2CModel', I2CModel)
