import os
import json
import math
from ai.search import MCTS
from ai.model import ChappieZero

import torch

class Agent():
    def __init__(self,folder='data/',model_name='model_active.pth.tar',param_name='model_param.json'):
        #Model parameters
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        param_path = os.path.join(folder,param_name)
        if os.path.exists(param_path):
            with open(param_path) as f:
                m_param = json.load(f)
        else:
            raise Exception('ERROR - Supplied model parameter file does not exist.')
        #Build model
        self.Model = ChappieZero(
            m_param['input-size'],
            m_param['latent-size'],
            m_param['reward-size'],
            m_param['policy-size'],
            ntoken = m_param['ntoken'],
            embedding_size=m_param['embedding-size'],
            padding_idx=m_param['padding-idx'],
            encoder_dropout=m_param['encoder-dropout'],
            latent_inner=m_param['latent-inner'],
            latent_heads=m_param['latent-heads'],
            latent_dropout=m_param['latent-dropout'],
            perceiver_inner=m_param['perceiver-inner'],
            recursions=m_param['recursions'],
            transformer_blocks=m_param['transformer-blocks'],
            cross_heads=m_param['cross-heads'],
            self_heads=m_param['self-heads'],
            cross_dropout=m_param['cross-dropout'],
            self_dropout=m_param['self-dropout'],
            reward_inner=m_param['reward-inner'],
            reward_heads=m_param['reward-heads'],
            reward_dropout=m_param['reward-dropout'],
            policy_inner=m_param['policy-inner'],
            policy_heads=m_param['policy-heads'],
            policy_dropout=m_param['policy-dropout']
        ).to(self.Device)
        model_path = os.path.join(folder,model_name)
        if os.path.exists(model_path):
            with open(model_path) as f:
                checkpoint = torch.load(f,map_location=self.Device)
                self.Model.load_state_dict(checkpoint['state_dict'])
        #Inialize search
        self.MCTS = MCTS(self.Model)

    def choose_action(self,game):
        self.MCTS.search(self,game,parent_hash)
        u_bank = {}
        for s,a in game.actions():
            u_bank[(s,a)] = self.MCTS.tree[hash].Q + (self.MCTS.Cpuct * self.MCTS.tree[hash].P * (math.sqrt(math.log(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N))))
        m_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
        if len(m_bank) > 0:
            action = random.choice(m_bank)
        else:
            action = ''
        return action
