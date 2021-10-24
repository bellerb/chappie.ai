from games.chess.chess import Chess

from model import ChappieZero

import torch


game = Chess()

game.display()

x = torch.tensor([[2,3,5,4,5,2,5,6]])

l = torch.full((1,2,3), 1).float()
r = torch.full((1,1,3), 1).float()
p = torch.full((1,64,64), 1).float()
print(x.size())
print(l.size())
print(r.size())
print(p.size())

chappie = ChappieZero(
    x.size(-1),
    l.size(-1),
    r.size(-1),
    p.size(-1),
    ntoken = 30,
    embedding_size=64,
    padding_idx=29,
    encoder_dropout=0.5,
    latent_inner=10,
    latent_heads=1,
    latent_dropout=0.5,
    perceiver_inner=64,
    recursions=1,
    transformer_blocks=1,
    cross_heads=1,
    self_heads=1,
    cross_dropout=0.5,
    self_dropout=0.5,
    reward_inner=64,
    reward_heads=1,
    reward_dropout=0.5,
    policy_inner=64,
    policy_heads=1,
    policy_dropout=0.5
)

v,p = chappie(x,l,r,p)
print('-----')
print(v.size())
print(p.size())
