from math import log

import torch
from torch import nn

from einops import rearrange, repeat

class Attention(nn.Module):
    """
    Multi-head attention model
    """
    def __init__(
        self,
        input_size,
        context_size = None,
        layer_size = 64,
        heads = 1,
        dropout = 0.5
    ):
        """
        Input: input_size - integer representing the size of the input data
               context_size - integer representing the size of the context data (default = None) [OPTIONAL]
               layer_size - integer representing the size of the layers (default = 64) [OPTIONAL]
               heads - integer representing the amount of heads to use (default = 1) [OPTIONAL]
               dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
        Description: Initailize attention model class creating the appropiate layers
        Output: None
        """
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
        """
        Input: x - tensor containing the input data
               context - tensor containing contextual data used in cross attention (default = None) [OPTIONAL]
               mask - tensor containing masked values (default = None) [OPTIONAL]
        Description: Forward pass of multi-head attention layer
        Output: tensor containing model output
        """
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
        z = torch.einsum('b q y, b k y -> b k q', q, k) / (x.size(-1) ** (0.5)) #Scaled dot-product [QK.T/sqrt(dk)]

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
    """
    Decoder only transformer model
    """
    def __init__(
        self,
        input_size,
        layer_size = 64,
        heads = 1,
        dropout = 0.5
    ):
        """
        Input: input_size - integer representing the size of the input data
               layer_size - integer representing the size of the layers (default = 64) [OPTIONAL]
               heads - integer representing the amount of heads to use (default = 1) [OPTIONAL]
               dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
        Description: Initailize decoder only transformer class creating the appropiate layers
        Output: None
        """
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
        """
        Input: x - tensor containing input data for decoder only transformer
        Description: Forward pass of decoder only transformer
        Output: None
        """
        z = self.self_attention(x)
        z = nn.functional.normalize(z, dim = -1)
        z = self.linear(z)
        z = nn.functional.normalize(z, dim = -1)
        z = self.GELU(z)
        return z

class Perceiver(nn.Module):
    """
    Perceiver model
    """
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
        """
        Input: input_size - integer representing the size of the input data
               latent_size - integer representing the size of the latent layer
               recursions - integer representing the amount of recursions to run (default = 1) [OPTIONAL]
               transformer_blocks - integer representing the amount of transformer blocks to use in our recursions (default = 1) [OPTIONAL]
               layer_size - integer representing the size of the layers (default = 64) [OPTIONAL]
               cross_heads - integer representing the amount of heads to use in the cross attention blocks (default = 1) [OPTIONAL]
               self_heads - integer representing the amount of heads in the self attention blocks (default = 1) [OPTIONAL]
               self_dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
        Description: Initailize perceiver class creating the appropiate layers
        Output: None
        """
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
        """
        Input: x - tensor containing input data
               latent - tensor containing latent input data
        Description: Forward pass of perceiver model
        Output: tensor containing output of model
        """
        z = self.cross_attention(latent, context = x)
        for _ in range(self.recursions):
            for _ in range(self.transformer_blocks):
                z = self.latent_transformer(z)
            z = self.cross_attention(z, context = x)
        z = self.latent_transformer(z)
        return z

class PositionalEncoding(nn.Module):
    """
    Encode input vectors with posistional data
    """
    def __init__(
        self,
        d_model,
        dropout = 0.1,
        max_len = 5000
    ):
        """
        Input: d_model - integer containing the size of the data model input
               dropout - integer representing the dropout percentage you want to use (Default=0.1) [OPTIONAL]
               max_len - integer representing the max amount of tokens in a input (Default=5000) [OPTIONAL]
        Description: Initailize positional encoding layer
        Output: None
        """
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
        """
        Input: x - pytorch tensor containing the input data for the model
        Description: forward pass of the positional encoding layer
        Output: pytorch tensor containing positional encoded data (floats)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Representation(nn.Module):
    """
    Representation model
    """
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
        """
        Input: latent_size - integer representing the size of the latent layer
               ntoken - integer representing the amount of tokens (default = 30) [OPTIONAL]3
               embedding_size - integer representing the size of the embedding layers (default = 64) [OPTIONAL]
               padding_idx - integer representing the index of the padding token (default = 29) [OPTIONAL]
               encoder_dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
               h_inner - integer representing the size of our hidden layer (default = 64) [OPTIONAL]
               h_heads - integer representing the amount of heads in the hidden layer (default = 1) [OPTIONAL]
               h_dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
        Description: Initailize representation class creating the appropiate layers
        Output: None
        """
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
        """
        Input: s - tensor representing the encoded state of the task
        Description: Forward pass of the representation model
        Output: tensor representing the output of the model
        """
        s_emb = self.Embedding(s)
        s_emb = self.PosEncoder(s_emb)
        latent = repeat(self.latent, 'y x -> b y x', b = s.size(0))
        h = self.HiddenNetwork(latent, s_emb)
        h = self.GELU(h)
        return h

class Backbone(nn.Module):
    """
    Backbone layer of multi task model
    """
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
        """
        Input: latent_size - integer representing the size of the latent layer
               action_space - integer representing the amount of possible actions (default = 4096) [OPTIONAL]
               embedding_size - integer representing the size of the embedding layers (default = 64) [OPTIONAL]
               perceiver_inner - integer representing the size of the perceiver model (default = 64) [OPTIONAL]
               recursions - integer representing the amount of recursions to run (default = 1) [OPTIONAL]
               transformer_blocks - integer representing the amount of transformer blocks to use in our recursions (default = 1) [OPTIONAL]
               cross_heads - integer representing the amount of heads to use in the cross attention blocks (default = 1) [OPTIONAL]
               self_heads - integer representing the amount of heads in the self attention blocks (default = 1) [OPTIONAL]
               cross_dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
               self_dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
        Description: Initailize representation class creating the appropiate layers
        Output: None
        """
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
        """
        Input: s - torch representing the encoded hidden state representation
               a - torch representing the action being taken
        Description: Forward pass of backbone layer
        Output: None
        """
        a_emb = self.ActionSpace(a)
        enc = self.Perceiver(a_emb, s)
        enc = self.GELU(enc)
        return enc

class Head(nn.Module):
    """
    Head layer for multi task model
    """
    def __init__(
        self,
        input_size,
        latent_size,
        inner = 64,
        heads = 1,
        dropout = 0.5,
        activation = None
    ):
        """
        Input: state_size - integer representing the size of the state layer
               latent_size - integer representing the size of the latent layer
               state_inner - integer representing the size of the state model (default = 64) [OPTIONAL]
               state_heads - integer representing the amount of heads in the attention block (default = 1) [OPTIONAL]
               state_dropout - float representing the amount of dropout to use (default = 0.5) [OPTIONAL]
        Description: Initailize next state class creating the appropiate layers
        Output: None
        """
        super(Head, self).__init__()
        self.latent = nn.Parameter(torch.randn(input_size))
        self.cross_attention = Attention(
            self.latent.size(-1),
            layer_size = inner,
            context_size = latent_size[-1],
            heads = heads,
            dropout = dropout
        )
        self.activation = activation

    def forward(self, enc):
        """
        Input: enc - tensor representing the auto encoded action
        Description: Forward pass of next state head
        Output: None
        """
        if len(self.latent.size()) == 1:
            latent = repeat(self.latent, 'x -> b y x', b = enc.size(0), y = 1)
        else:
            latent = repeat(self.latent, 'y x -> b y x', b = enc.size(0))
        s_k = self.cross_attention(latent, enc)
        if self.activation is not None:
            s_k = self.activation(s_k)
        return s_k
