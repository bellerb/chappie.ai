# Skills
### Description
Folder where all interaction with exteral tasks occur. For each task a small wrapper is created that the AI interacts with. New models are trained for ever task individually with custom paramaters specific to the task at hand.

# Parameter Example
```
{
  "data": {
    "token_bank":"skills/chess/data/token_bank.csv",
    "active-models":{
      "representation":"skills/chess/data/models/h_new.pth.tar",
      "backbone":"skills/chess/data/models/g_new.pth.tar",
      "value":"skills/chess/data/models/v_new.pth.tar",
      "policy":"skills/chess/data/models/p_new.pth.tar",
      "state":"skills/chess/data/models/s_new.pth.tar",
      "reward":"skills/chess/data/models/r_new.pth.tar"
    }
  },
  "search":{
    "T": [
      [500000, 1],
      [750000, 0.5],
      [1000000, 0.25]
    ],
    "sim_amt": 25,
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
    "state_k_inner":400,
    "cross_heads":4,
    "self_heads":4,
    "h_heads":4,
    "value_heads":4,
    "policy_heads":4,
    "reward_heads":4,
    "state_k_heads":4,
    "encoder_dropout":0.5,
    "cross_dropout":0.5,
    "self_dropout":0.5,
    "h_dropout":0.5,
    "value_dropout":0.5,
    "policy_dropout":0.5,
    "reward_dropout":0.5,
    "state_k_dropout":0.5
  },
  "training": {
    "bsz": 10,
    "lr": 0.0001,
    "epoch": 10
  }
}
```
