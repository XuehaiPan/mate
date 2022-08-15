from collections import OrderedDict

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch


__all__ = ['SimpleMLP', 'SimpleRNN']


torch, nn = try_import_torch()


class SimpleMLP(nn.Module):
    def __init__(
        self,
        name,
        input_dim,
        hidden_dims,
        output_dim,
        layer_norm=False,
        activation='relu',
        output_activation=None,
    ):
        super().__init__()

        self.name = name
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation

        hidden_layers = []
        hidden_dims = [input_dim, *hidden_dims]
        for i, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append(
                (
                    f'{name}_hidden_{i}',
                    SlimFC(
                        in_size=in_size,
                        out_size=out_size,
                        initializer=normc_initializer(1.0),
                        activation_fn=self.activation,
                    ),
                )
            )
            if layer_norm:
                hidden_layers.append((f'{name}_layernorm_{i}', nn.LayerNorm(out_size)))
        self.hidden = nn.Sequential(OrderedDict(hidden_layers))
        self.output = SlimFC(
            in_size=hidden_dims[-1],
            out_size=self.output_dim,
            initializer=normc_initializer(0.01),
            activation_fn=self.output_activation,
        )

        self._last_features = None
        self._last_output = None

    def get_initial_state(self):
        return []

    def forward(self, x, features_only=False):
        self._last_features = self.hidden(x)

        if not features_only:
            out = self._last_output = self.output(self._last_features)
        else:
            out = self._last_features
            self._last_output = None

        return out

    @property
    def last_features(self):
        return self._last_features

    @property
    def last_output(self):
        return self._last_output


class SimpleRNN(nn.Module):
    def __init__(
        self,
        name,
        input_dim,
        hidden_dims,
        cell_size,
        output_dim,
        activation='relu',
        output_activation=None,
    ):
        super().__init__()

        self.name = name
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.cell_size = cell_size
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation

        hidden_layers = []
        hidden_dims = [input_dim, *hidden_dims]
        for i, (in_size, out_size) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append(
                (
                    f'{name}_hidden_{i}',
                    SlimFC(
                        in_size=in_size,
                        out_size=out_size,
                        initializer=normc_initializer(1.0),
                        activation_fn=self.activation,
                    ),
                )
            )
        self.hidden = nn.Sequential(OrderedDict(hidden_layers))
        self.lstm = nn.LSTM(hidden_dims[-1], self.cell_size, batch_first=True)
        self.output = SlimFC(
            in_size=hidden_dims[-1] + self.cell_size,
            out_size=self.output_dim,
            initializer=normc_initializer(0.01),
            activation_fn=self.output_activation,
        )

        self._last_features = None
        self._last_output = None

    def get_initial_state(self):
        lstm_param = next(self.lstm.parameters())

        return [
            lstm_param.new(self.cell_size).zero_(),
            lstm_param.new(self.cell_size).zero_(),
        ]

    def forward(self, x, hx=None, features_only=False):
        if hx is None or True:
            hx = [h.expand(x.size(0), -1) for h in self.get_initial_state()]
        h, c = hx

        x_in = self.hidden(x)
        h, c = h.unsqueeze(dim=0).contiguous(), c.unsqueeze(dim=0).contiguous()
        x_out, (h, c) = self.lstm(x_in, (h, c))
        self._last_features = torch.cat((x_in, x_out), dim=-1)

        if not features_only:
            out = self._last_output = self.output(self._last_features)
        else:
            out = self._last_features
            self._last_output = None

        return out, (h.squeeze(dim=0), c.squeeze(dim=0))

    @property
    def last_features(self):
        return self._last_features

    @property
    def last_output(self):
        return self._last_output
