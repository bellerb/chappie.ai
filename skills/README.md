# Skills
### Description
Folder where all interaction with exteral tasks occur. For each task a small wrapper is created that the AI interacts with. New models are trained for every task individually with custom paramaters specific to the task at hand.

# Parameter Example
```
{
  "data": {
    "token_bank":"skills/chess/data/token_bank.csv",
    "active-models":{
      "representation":"h.pth.tar",
      "backbone":"g.pth.tar",
      "value":"v.pth.tar",
      "policy":"p.pth.tar",
      "state":"s.pth.tar",
      "reward":"r.pth.tar",
      "cca":"cca.pth.tar"
    }
  },
  "search":{
    "T": [
      [500000, 1],
      [750000, 0.5],
      [1000000, 0.25]
    ],
    "sim_amt": 25,
    "workers":2,
    "c1":1.25,
    "c2":19652,
    "d_a":0.3,
    "e_f":0.25,
    "g_d":1.0,
    "max_depth":20,
    "single_player": false
  },
  "model": {
    "latent_size":[20,20],
    "h_size":[250, 200],
    "value_size":[1],
    "reward_size":[1],
    "ntoken":34,
    "g_recursions":5,
    "action_space":4096,
    "embedding_size":400,
    "padding_idx":33,
    "transformer_blocks":3,
    "perceiver_inner":800,
    "value_inner":400,
    "h_inner":400,
    "policy_inner":400,
    "reward_inner":400,
    "state_inner":400,
    "chunked_inner":64,
    "cross_heads":4,
    "self_heads":4,
    "h_heads":4,
    "value_heads":4,
    "policy_heads":4,
    "reward_heads":4,
    "state_heads":4,
    "chunked_heads":1,
    "encoder_dropout":0.5,
    "cross_dropout":0.5,
    "self_dropout":0.5,
    "h_dropout":0.5,
    "value_dropout":0.5,
    "policy_dropout":0.5,
    "reward_dropout":0.5,
    "state_dropout":0.5,
    "chunked_dropout":0.5,
    "chunked_length":4,
    "neighbour_amt":2,
    "moe_k":1,
    "experts":1,
    "moe":false,
    "retro":false
  },
  "training": {
    "bsz": 10,
    "lr": 0.0001,
    "epoch": 10,
    "h_step":1,
    "b_step":1,
    "cca_step":1,
    "v_step":1,
    "p_step":1,
    "s_step":1,
    "r_step":1,
    "h_gamma":0.1,
    "b_gamma":0.1,
    "cca_gamma":0.1,
    "v_gamma":0.1,
    "p_gamma":0.1,
    "s_gamma":0.1,
    "r_gamma":0.1,
    "h_max_norm":0.5,
    "b_max_norm":0.5,
    "cca_max_norm":0.5,
    "v_max_norm":0.5,
    "p_max_norm":0.5,
    "s_max_norm":0.5,
    "r_max_norm":0.5
  }
}
```
