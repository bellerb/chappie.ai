import os
import json
import torch
import pandas as pd

from tools.toolbox import ToolBox
from ai.model import ChunkedCrossAttention, DecoderOnlyTransformer, Representation, Backbone

param_name = 'skills/chess/data/models/test_V2/parameters.json'

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(param_name):
    with open(param_name) as f:
        m_param = json.load(f)
else:
    raise Exception('ERROR - Supplied model parameter file does not exist.')

p_model = m_param['model']
representation = Representation(
    p_model['latent_size'],
    ntoken = p_model['ntoken'],
    embedding_size = p_model['embedding_size'],
    padding_idx = p_model['padding_idx'],
    encoder_dropout = p_model['encoder_dropout'],
    h_inner = p_model['h_inner'],
    h_heads = p_model['h_heads'],
    h_dropout = p_model['h_dropout']
).to(Device)

backbone = Backbone(
    p_model['latent_size'],
    action_space = p_model['action_space'],
    embedding_size = p_model['embedding_size'],
    perceiver_inner = p_model['perceiver_inner'],
    recursions = p_model['g_recursions'],
    transformer_blocks = p_model['transformer_blocks'],
    cross_heads = p_model['cross_heads'],
    self_heads = p_model['self_heads'],
    cross_dropout = p_model['cross_dropout'],
    self_dropout = p_model['self_dropout']
).to(Device)

e_db = ToolBox.build_embedding_db(representation, backbone, f_name = 'skills/chess/data/models/test_V2/logs/game_log.csv')

x = torch.tensor(e_db.iloc[5][0])
#print(x)
print(x.shape)

n = x.size(0) #Sequence length
m = 4 #Chunk length
k = 2 #Amount of neighbours
d = x.size(-1) #Embedding size
l = n // m #Number of chunks

chunks = x.reshape(l, m, d)
#print(chunks)
print(chunks.shape)

neighbours = ToolBox.get_kNN(chunks, e_db)
#print(neighbours)
print(neighbours.shape)

Cca = ChunkedCrossAttention(
    x.size(-1),
    layer_size = 64,
    heads = 1,
    dropout = 0.5,
    n = n, #Sequence length
    m = m, #Chunk length
    k = k, #Amount of neighbours
    r = neighbours.size(2), #Retrieval length
    d = d #Embedding size
)

z = Cca(x, neighbours)
#print(z)
print(z.shape)
