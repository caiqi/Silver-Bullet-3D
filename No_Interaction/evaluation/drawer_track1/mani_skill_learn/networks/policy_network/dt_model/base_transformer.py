import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        d_model_old = d_model
        d_model = int(math.ceil(d_model * 0.5) * 2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :, :d_model_old]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


from torch.nn.modules import ModuleList
import copy


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


#
# class TransformerEncoder(Module):
#     r"""TransformerEncoder is a stack of N encoder layers
#
#     Args:
#         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#         num_layers: the number of sub-encoder-layers in the encoder (required).
#         norm: the layer normalization component (optional).
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = transformer_encoder(src)
#     """
#     __constants__ = ['norm']
#
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#
#     def forward(self, src: Tensor, mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layers in turn.
#
#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#         output = src
#
#         for mod in self.layers:
#             output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             output[output != output] = 0
#             # output = torch.nan_to_num(output)
#
#         if self.norm is not None:
#             output = self.norm(output)
#
#         return output

class TransformerModel(nn.Module):
    def __init__(self,
                 state_dim,
                 act_dim,
                 max_length,
                 disable_action,
                 nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, pass_through=False, original_weight=0.0, shuffle_noise=0.0):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_hid, dropout)
        encoder_layers = TransformerEncoderLayer(d_hid, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.pass_through = pass_through
        self.original_weight = original_weight
        self.shuffle_noise = shuffle_noise
        self.embed_state = torch.nn.Linear(state_dim, d_hid)
        self.embed_action = torch.nn.Linear(act_dim, d_hid)
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.d_model = d_hid
        self.decoder = nn.Linear(d_hid, act_dim)
        self.max_length = max_length
        self.disable_action = disable_action
        src_mask = generate_square_subsequent_mask(self.max_length * 2)
        self.register_buffer('src_mask', src_mask)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embed_state.weight.data.uniform_(-initrange, initrange)
        self.embed_action.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, states, actions, attention_mask=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # states: B, T, D
        # actions: B, T, D
        # attention_mask: B, T
        if self.pass_through:
            return states
        states_original = states
        states = self.embed_state(states)
        actions = self.embed_action(actions)
        batch_size, seq_length, hidden_size = states.shape
        if self.disable_action:
            stacked_inputs = states.permute(1, 0, 2)
            stacked_attention_mask = attention_mask
            stacked_inputs = stacked_inputs[-self.max_length:, :, :]
            stacked_attention_mask = stacked_attention_mask[:, -self.max_length:]
        else:
            stacked_inputs = torch.stack(
                (states, actions), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, hidden_size)
            stacked_inputs = stacked_inputs.permute(1, 0, 2)
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, -1)
            stacked_inputs = stacked_inputs[-self.max_length * 2:, :, :]
            stacked_attention_mask = stacked_attention_mask[:, -self.max_length * 2:]

        stacked_attention_mask = 1 - stacked_attention_mask
        stacked_attention_mask = stacked_attention_mask.type(torch.bool)
        # T, B, D
        src = self.pos_encoder(stacked_inputs)
        if self.training:
            T, B, D = src.shape
            src = src.reshape(T * B, D)
            random_mask = torch.rand(T * B) < self.shuffle_noise
            src[random_mask] = 0
            src = src.reshape(T, B, D)

        src_mask = self.src_mask[:stacked_inputs.shape[0], :stacked_inputs.shape[0]]
        # random_mask = torch.rand_like(src_mask) < self.shuffle_noise
        # random_mask = random_mask.type(torch.float)
        # random_mask[torch.nonzero(random_mask, as_tuple=True)] = float("-inf")
        # src_mask = src_mask + random_mask

        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=stacked_attention_mask)
        output = self.decoder(output)
        if self.disable_action:
            output = output.view(-1, 1, batch_size, self.act_dim)
            actions_pred = output[:, 0, :, :].permute(1, 0, 2)
            if self.original_weight > 0:
                weighted_actions = actions_pred * (
                        1 - self.original_weight) + states_original.detach() * self.original_weight
                return weighted_actions
            return actions_pred
        else:
            output = output.view(-1, 2, batch_size, self.act_dim)
            actions_pred = output[:, 0, :, :].permute(1, 0, 2)
            state_pred = output[:, 1, :, :].permute(1, 0, 2)
            if self.original_weight > 0:
                weighted_actions = actions_pred * (
                        1 - self.original_weight) + states_original.detach() * self.original_weight
                return weighted_actions
            return actions_pred

    def get_action(self, states, actions):
        # we don't care about the past rewards in this model
        state_dim = states.shape[-1]
        states = states.reshape(1, -1, state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        attention_mask = torch.cat([torch.ones(states.shape[1]), torch.zeros(self.max_length - states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        states = torch.cat(
            [states, torch.zeros((states.shape[0], self.max_length - states.shape[1], state_dim),
                                 device=states.device)],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [actions, torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                                  device=actions.device)],
            dim=1).to(dtype=torch.float32)
        action_preds = self.forward(
            states, actions, attention_mask=attention_mask)
        return action_preds[0][attention_mask.flatten().type(torch.bool)][-1]


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
