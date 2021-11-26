from math import log

import torch
from torch import nn

from einops import rearrange, repeat

class Attention(nn.Module):
    def __init__(
        self,
        input_size,
        context_size = None,
        layer_size = 64,
        heads = 1,
        dropout = 0.5
    ):
        super(Attention, self).__init__()
        self.heads = heads
        self.Q = nn.Linear(
            input_size,
            layer_size,
            bias = False
        )
        if context_size is None:
            self.K = nn.Linear(
                input_size,
                layer_size,
                bias = False
            )
            self.V = nn.Linear(
                input_size,
                layer_size,
                bias = False
            )
        else:
            self.K = nn.Linear(
                context_size,
                layer_size,
                bias = False
            )
            self.V = nn.Linear(
                context_size,
                layer_size,
                bias = False
            )
        self.softmax = nn.Softmax(dim = -1)
        self.output = nn.Sequential(
            nn.Linear(
                layer_size,
                input_size,
                bias = False
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        #h:heads, b:batches, y:y-axis x:x-axis
        q = rearrange(self.Q(x), 'b y (h x) -> (b h) y x', h = self.heads) #Query
        if context is None:
            k, v = map(
                lambda x:rearrange(x, 'b y (h x) -> (b h) y x', h = self.heads),
                (self.K(x), self.V(x))
            )
        else:
            k, v = map(
                lambda x:rearrange(x, 'b y (h x) -> (b h) y x', h = self.heads),
                (self.K(context), self.V(context))
            )
        #b:batches, y:y-axis, q:q x-axis, k: k x-axis
        #print('q',q.size())
        #print('k',k.size())
        z = torch.einsum('b q y, b k y -> b k q', q, k) / (x.size(-1)**(0.5)) #Scaled dot-product [QK.T/sqrt(dk)]

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(z)
            z = z.masked_fill(mask_expanded, -1e18)

        z = self.softmax(z)
        #b:batches, c:common dim, z:z x-axis, v:v x-axis
        z = torch.einsum('b y z, b y v -> b z v', z, v) #Dot product [ZV]
        #h:heads, b:batches, y:y-axis, x:x-axis
        z = rearrange(z, '(b h) y x -> b y (h x)', h = self.heads) #Concat data
        z = self.output(z)
        return z

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        layer_size = 64,
        heads = 1,
        dropout = 0.5
    ):
        super(DecoderOnlyTransformer, self).__init__()
        self.self_attention = Attention(
            input_size,
            layer_size = layer_size,
            heads = heads,
            dropout = dropout
        )
        self.linear = nn.Sequential(
            nn.Linear(
                input_size,
                input_size,
                bias = False
            ),
            nn.Dropout(dropout)
        )
        self.GELU = torch.nn.GELU()

    def forward(self, x):
        z = self.self_attention(x)
        z = nn.functional.normalize(z, dim=-1)
        z = self.linear(z)
        z = nn.functional.normalize(z, dim=-1)
        z = self.GELU(z)
        return z

class Perceiver(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        recursions = 1,
        transformer_blocks = 1,
        layer_size = 64,
        cross_heads = 1,
        self_heads = 1,
        cross_dropout = 0.5,
        self_dropout = 0.5
    ):
        super(Perceiver, self).__init__()
        self.recursions = recursions
        self.transformer_blocks = transformer_blocks
        self.cross_attention = Attention(
            latent_size,
            layer_size = layer_size,
            context_size = input_size,
            heads = cross_heads,
            dropout = cross_dropout
        )
        self.latent_transformer = DecoderOnlyTransformer(
            latent_size,
            layer_size,
            self_heads,
            self_dropout
        )

    def forward(self, x, latent):
        z = self.cross_attention(latent, context = x)
        for _ in range(self.recursions):
            for _ in range(self.transformer_blocks):
                z = self.latent_transformer(z)
            z = self.cross_attention(z, context = x)
        z = self.latent_transformer(z)
        return z

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.1,
        max_len = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0,
            max_len,
            dtype = torch.float
        ).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Representation(nn.Module):
    def __init__(
        self,
        latent_size,
        ntoken = 30,
        embedding_size = 64,
        padding_idx = 29,
        encoder_dropout = 0.5,
        h_inner = 64,
        h_heads = 1,
        h_dropout = 0.5
    ):
        super(Representation, self).__init__()
        self.latent = nn.Parameter(torch.randn(latent_size))
        self.Embedding = nn.Embedding(
            ntoken,
            embedding_size,
            padding_idx = padding_idx
        )
        self.PosEncoder = PositionalEncoding(
            embedding_size,
            encoder_dropout
        )
        self.HiddenNetwork = Attention(
            self.latent.size(-1),
            layer_size = h_inner,
            context_size = embedding_size,
            heads = h_heads,
            dropout = h_dropout
        )
        self.GELU = torch.nn.GELU()

    def forward(self, s):
        s_emb = self.Embedding(s)
        s_emb = self.PosEncoder(s_emb)
        latent = repeat(self.latent, 'y x -> b y x', b = s.size(0))
        h = self.HiddenNetwork(latent, s_emb)
        h = self.GELU(h)
        return h

class Backbone(nn.Module):
    def __init__(
        self,
        latent_size,
        action_space = 4096,
        embedding_size = 64,
        perceiver_inner = 64,
        recursions = 1,
        transformer_blocks = 1,
        cross_heads = 1,
        self_heads = 1,
        cross_dropout = 0.5,
        self_dropout = 0.5,
    ):
        super(Backbone, self).__init__()
        self.ActionSpace = nn.Embedding(
            action_space + 1,
            embedding_size
        )
        self.Perceiver = Perceiver(
            embedding_size,
            latent_size[-1],
            recursions = recursions,
            transformer_blocks = transformer_blocks,
            layer_size = perceiver_inner,
            cross_heads = cross_heads,
            self_heads = self_heads,
            cross_dropout = cross_dropout,
            self_dropout = self_dropout
        )
        self.GELU = torch.nn.GELU()

    def forward(self, s, a):
        a_emb = self.ActionSpace(a)
        enc = self.Perceiver(a_emb, s)
        enc = self.GELU(enc)
        return enc

class Value(nn.Module):
    def __init__(
        self,
        value_size,
        latent_size,
        value_inner = 64,
        value_heads = 1,
        value_dropout = 0.5,
    ):
        super(Value, self).__init__()
        self.value = nn.Parameter(torch.randn(value_size))
        self.ValueNetwork = Attention(
            self.value.size(-1),
            layer_size = value_inner,
            context_size = latent_size[-1],
            heads = value_heads,
            dropout = value_dropout
        )
        #self.tanh = nn.Tanh()

    def forward(self, enc):
        value = repeat(self.value, 'x -> b y x', b = enc.size(0), y = 1)
        v = self.ValueNetwork(value, enc)
        #v = self.tanh(v)
        return v

class Policy(nn.Module):
    def __init__(
        self,
        policy_size,
        latent_size,
        policy_inner = 64,
        policy_heads = 1,
        policy_dropout = 0.5
    ):
        super(Policy, self).__init__()
        self.action_space = policy_size
        self.policy = nn.Parameter(torch.randn(policy_size))
        self.PolicyNetwork = Attention(
            self.policy.size(-1),
            layer_size = policy_inner,
            context_size = latent_size[-1],
            heads = policy_heads,
            dropout = policy_dropout
        )
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, enc):
        policy = repeat(self.policy, 'x -> b y x', b = enc.size(0), y = 1)
        p = self.PolicyNetwork(policy, enc)
        p = self.softmax(p)
        return p

class Reward(nn.Module):
    def __init__(
        self,
        reward_size,
        latent_size,
        reward_inner = 64,
        reward_heads = 1,
        reward_dropout = 0.5,
    ):
        super(Reward, self).__init__()
        self.reward = nn.Parameter(torch.randn(reward_size))
        self.RewardNetwork = Attention(
            self.reward.size(-1),
            layer_size = reward_inner,
            context_size = latent_size[-1],
            heads = reward_heads,
            dropout = reward_dropout
        )

    def forward(self, enc):
        reward = repeat(self.reward, 'x -> b y x', b = enc.size(0), y = 1)
        r = self.RewardNetwork(reward, enc)
        return r

class NextState(nn.Module):
    def __init__(
        self,
        state_size,
        latent_size,
        state_k_inner = 64,
        state_k_heads = 1,
        state_k_dropout = 0.5
    ):
        super(NextState, self).__init__()
        self.state = nn.Parameter(torch.randn(state_size))
        self.StateNetwork = Attention(
            self.state.size(-1),
            layer_size = state_k_inner,
            context_size = latent_size[-1],
            heads = state_k_heads,
            dropout = state_k_dropout
        )
        self.GELU = torch.nn.GELU()

    def forward(self, enc):
        state = repeat(self.state, 'y x -> b y x', b = enc.size(0))
        s_k = self.StateNetwork(state, enc)
        s_k = self.GELU(s_k)
        return s_k
