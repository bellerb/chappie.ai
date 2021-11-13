import os
import json
import torch
import random
import numpy as np

from tqdm import tqdm

from ai.search import MCTS
from ai.model import Representation, Predictions, Dynamics

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

class Agent:
    def __init__(self, param_name='model_param.json', train = False):
        #Model parameters
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(param_name):
            with open(param_name) as f:
                m_param = json.load(f)
        else:
            raise Exception('ERROR - Supplied model parameter file does not exist.')
        self.action_space = m_param['model']['action_space']
        p_model = m_param['model']
        self.representation = Representation(
            p_model['latent_size'],
            p_model['h_size'],
            ntoken = p_model['ntoken'],
            embedding_size = p_model['embedding_size'],
            padding_idx = p_model['padding_idx'],
            encoder_dropout = p_model['encoder_dropout'],
            perceiver_inner = p_model['perceiver_inner'],
            recursions = p_model['h_recursions'],
            transformer_blocks = p_model['transformer_blocks'],
            cross_heads = p_model['cross_heads'],
            self_heads = p_model['self_heads'],
            cross_dropout = p_model['cross_dropout'],
            self_dropout = p_model['self_dropout'],
            h_inner = p_model['h_inner'],
            h_heads = p_model['h_heads'],
            h_dropout = p_model['h_dropout']
        ).to(self.Device)
        self.representation.eval()
        #self.representation = _h()
        predictions = Predictions(
            p_model['h_size'],
            p_model['latent_size'],
            p_model['value_size'],
            p_model['action_space'],
            perceiver_inner = p_model['perceiver_inner'],
            recursions = p_model['f_recursions'],
            transformer_blocks = p_model['transformer_blocks'],
            cross_heads = p_model['cross_heads'],
            self_heads = p_model['self_heads'],
            cross_dropout = p_model['cross_dropout'],
            self_dropout = p_model['self_dropout'],
            value_inner = p_model['value_inner'],
            value_heads = p_model['value_heads'],
            value_dropout = p_model['value_dropout'],
            policy_inner = p_model['policy_inner'],
            policy_heads = p_model['policy_heads'],
            policy_dropout = p_model['policy_dropout']
        ).to(self.Device)
        predictions.eval()
        #predictions = _f(action_space=self.action_space)
        dynamics = Dynamics(
            p_model['h_size'],
            p_model['reward_size'],
            p_model['action_space'],
            ntoken = p_model['ntoken'],
            action_space = p_model['action_space'],
            embedding_size = p_model['embedding_size'],
            padding_idx = p_model['padding_idx'],
            encoder_dropout = p_model['encoder_dropout'],
            perceiver_inner = p_model['perceiver_inner'],
            recursions = p_model['g_recursions'],
            transformer_blocks = p_model['transformer_blocks'],
            cross_heads = p_model['cross_heads'],
            self_heads = p_model['self_heads'],
            cross_dropout = p_model['cross_dropout'],
            self_dropout = p_model['self_dropout'],
            reward_inner = p_model['reward_inner'],
            reward_heads = p_model['reward_heads'],
            reward_dropout = p_model['reward_dropout'],
            state_k_inner = p_model['state_k_inner'],
            state_k_heads = p_model['state_k_heads'],
            state_k_dropout = p_model['state_k_dropout']
        ).to(self.Device)
        dynamics.eval()
        #dynamics = _g()
        '''
        model_path = os.path.join(folder, model_name)
        if os.path.exists(model_path):
            with open(model_path) as f:
                checkpoint = torch.load(f, map_location=self.Device)
                self.Model.load_state_dict(checkpoint['state_dict'])
        '''
        #Inialize search
        if m_param['search']['max_depth'] is None:
            m_d = float('inf')
        else:
            m_d = m_param['search']['max_depth']
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
            max_depth = m_d
        )
        self.train = train
        if self.train is True:
            #self.T = m_param['search']['T'] #Tempature
            self.T = 1
        else:
            self.T = 1
        self.sim_amt = m_param['search']['sim_amt']

    def choose_action(self, state):
        #print(state)
        h_s = self.representation(state)
        #print(h_s)
        for _ in tqdm(range(self.sim_amt),desc='MCTS'):

            self.MCTS.depth = 0
            self.MCTS.search(h_s, train = self.train)

        s_hash = self.MCTS.state_hash(h_s)
        #print(s_hash)
        #print('--------------')
        counts = {a: self.MCTS.tree[(s_hash,a)].N for a in range(self.action_space)}
        #print(counts)

        if self.T == 0:
            a_bank = [k for k,v in counts.items() if v == max(counts.values())]
            a = random.choice(a_bank)
            probs = [0] * len(counts)
            probs[a] = 1
        else:
            c_s = sum(counts.values())
            #print(c_s)
            probs = [(x ** (1./self.T)) / c_s for x in counts.values()]
        return probs
