import os
import torch
import pandas as pd

from tools.toolbox import ToolBox
from ai.model import ChunkedCrossAttention, DecoderOnlyTransformer

n = 12 #Sequence length
m = 4 #Chunk length
k = 2 #Amount of neighbours
r = 5 #Retrieval length
d = 2 #Embedding size
l = n // m #Number of chunks
t = 50 #Amount of tokens in db

#x = torch.rand(n, d)
#print(x)

def build_embedding_db(f_name = None, s_header = 'state'):
    if f_name is not None and os.path.isfile(f_name):
        t_db = pd.read_csv(f_name)
        t_db = t_db[[h for h in t_db if s_header in h]].drop_duplicates()
    else:
        t_db = None
    print(t_db)
    if t_db is not None:
        e_db = torch.tensor(t_db.values)
    else:
        e_db = None
    print(e_db)
    return e_db

e_db = build_embedding_db(f_name = 'skills/chess/data/models/test_V2/logs/game_log.csv')
quit()
'''
e_db = []
for emb in torch.rand(t, r, d):
    e_db.append([emb.tolist()])
e_db = pd.DataFrame(e_db)
#print(e_db)
'''

chunks = torch.rand(l - 1, r, d)

neighbours = ToolBox.get_kNN(chunks, e_db)
#print(neighbours)

Transformer = DecoderOnlyTransformer(
    neighbours.size(-1),
    layer_size = 64,
    heads = 1,
    dropout = 0.5
)

x = Transformer(x.resize(1, n, d)).resize(n, d)
print(x)

for u in range(len(neighbours)):
    neighbours[u] = Transformer(neighbours[u])

Cca = ChunkedCrossAttention(
    x.size(-1),
    layer_size = 64,
    heads = 1,
    dropout = 0.5,
    n = x.size(0), #Sequence length
    m = m, #Chunk length
    k = k, #Amount of neighbours
    r = r, #Retrieval length
    d = d #Embedding size
)

z = Cca(x, neighbours)
print(z)
