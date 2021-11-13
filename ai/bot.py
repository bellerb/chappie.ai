import os
import json
import math
from ai.search import MCTS
from ai.model import ChappieZero

import torch

import random
import numpy as np

class _h():
    def predict(self, s):
        """
        TEMPORARY HIDDEN STATE MODEL FOR TESTING
        """
        return np.random.rand(4)

class _f():
    def __init__(self, action_space = 4):
        self.action_space = action_space

    def predict(self, s):
        """
        TEMPORARY PREDICTION MODEL FOR TESTING
        """
        return random.choice([-1, 0, 1]), np.random.rand(self.action_space)

class _g():
    def predict(self, s, a):
        """
        TEMPORARY DYNAMICS MODEL FOR TESTING
        """
        return 1, s * 0.1

class Agent():
    def __init__(self, param_name='model_param.json', train = False):
        #Model parameters
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(param_name):
            with open(param_name) as f:
                m_param = json.load(f)
        else:
            raise Exception('ERROR - Supplied model parameter file does not exist.')
        '''
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
        model_path = os.path.join(folder, model_name)
        if os.path.exists(model_path):
            with open(model_path) as f:
                checkpoint = torch.load(f, map_location=self.Device)
                self.Model.load_state_dict(checkpoint['state_dict'])
        '''
        self.action_space = m_param['prediction']['policy_size'][0]
        predictions = _f(action_space=self.action_space)
        dynamics = _g()
        self.representation = _h()
        #Inialize search
        self.MCTS = MCTS(
            predictions,
            dynamics,
            c1 = m_param['search']['c1'],
            c2 = m_param['search']['c2'],
            d_a = m_param['search']['d_a'],
            e_f = m_param['search']['e_f'],
            g_d = m_param['search']['g_d'],
            Q_max = m_param['search']['Q_max'],
            Q_min = m_param['search']['Q_min'],
            single_player = m_param['search']['single_player'],
            max_depth = float('inf') if m_param['search']['max_depth'] == None else m_param['search']['max_depth']
        )
        self.train = train
        if self.train == True:
            #self.T = m_param['search']['T'] #Tempature
            self.T = 1
        else:
            self.T = 1
        self.sim_amt = m_param['search']['sim_amt']

    def choose_action(self, state):
        h_s = self.representation.predict(state)
        for x in range(self.sim_amt):
            self.MCTS.depth = 0
            self.MCTS.search(h_s, train = self.train)

        s_hash = self.MCTS.state_hash(h_s)
        counts = {a: self.MCTS.tree[(s_hash,a)].N for a in range(self.action_space)}

        if self.T == 0:
            a_bank = [k for k,v in counts.items() if v == max(counts.values())]
            a = random.choice(a_bank)
            probs = [0] * len(counts)
            probs[a] = 1
        else:
            c_s = sum(counts.values())
            probs = [(x ** (1./self.T)) / c_s for x in counts.values()]
        return probs
