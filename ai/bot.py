import os
import time
import json
import torch
import random
import numpy as np

from tqdm import tqdm

from einops import rearrange

from ai.search import MCTS
from ai.model import Representation, Predictions, Dynamics

class Agent:
    def __init__(self, param_name='model_param.json', train = False):
        #Initalize models
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(param_name):
            with open(param_name) as f:
                m_param = json.load(f)
        else:
            raise Exception('ERROR - Supplied model parameter file does not exist.')
        self.action_space = m_param['model']['action_space']
        p_model = m_param['model']
        representation = Representation(
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
        representation.eval()
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
        self.m_weights = {
            'representation':{
                'model':representation,
                'param':m_param['data']['active-models']['representation']
            },
            'predictions':{
                'model':predictions,
                'param':m_param['data']['active-models']['predictions']
            },
            'dynamics':{
                'model':dynamics,
                'param':m_param['data']['active-models']['dynamics']
            }
        }
        for m in self.m_weights:
            if os.path.exists(self.m_weights[m]['param']):
                checkpoint = torch.load(
                    self.m_weights[m]['param'],
                    map_location=self.Device
                )
                self.m_weights[m]['model'].load_state_dict(checkpoint['state_dict'])
        #Inialize search
        if m_param['search']['max_depth'] is None:
            m_d = float('inf')
        else:
            m_d = m_param['search']['max_depth']
        self.MCTS = MCTS(
            self.m_weights['predictions']['model'],
            self.m_weights['dynamics']['model'],
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
        self.train_control = train
        if self.train_control is True:
            #self.T = m_param['search']['T'] #Tempature
            self.T = 1
        else:
            self.T = 1
        self.sim_amt = m_param['search']['sim_amt']
        self.bsz = m_param['training']['bsz'] #Batch size
        self.lr = m_param['training']['lr'] #Learning rate

    def choose_action(self, state):
        with torch.no_grad():
            h_s = self.m_weights['representation']['model'](state)
        for _ in tqdm(range(self.sim_amt),desc='MCTS'):
            self.MCTS.depth = 0
            self.MCTS.search(h_s, train = self.train_control)

        s_hash = self.MCTS.state_hash(h_s)
        value = self.MCTS.tree[(s_hash, None)].Q
        counts = {a: self.MCTS.tree[(s_hash,a)].N for a in range(self.action_space)}

        if self.T == 0:
            a_bank = [k for k,v in counts.items() if v == max(counts.values())]
            a = random.choice(a_bank)
            probs = [0] * len(counts)
            probs[a] = 1
        else:
            c_s = sum(c ** (1./self.T) for c in counts.values())
            probs = [(x ** (1./self.T)) / c_s for x in counts.values()]
        self.MCTS.tree = {}
        return probs, value

    def train(self, data):
        #Initailize training
        v_criterion = torch.nn.MSELoss() #Mean squared error loss
        p_criterion = torch.nn.BCELoss() #Binary cross entropy loss
        h_optimizer = torch.optim.SGD(
            self.m_weights['representation']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        f_optimizer = torch.optim.SGD(
            self.m_weights['predictions']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        g_optimizer = torch.optim.SGD(
            self.m_weights['dynamics']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        self.m_weights['representation']['model'].train() #Turn on the train mode
        self.m_weights['predictions']['model'].train() #Turn on the train mode
        self.m_weights['dynamics']['model'].train() #Turn on the train mode
        #train_data = train_data.sample(frac=1).reset_index(drop=True) #Shuffle training data
        train_data = torch.tensor(data.values) #Set training data to a tensor
        #Start training model
        t_steps = 0
        total_loss = 0.0
        start_time = time.time() #Get time of starting process
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bsz)):
            state, v_targets, p_targets = self.get_batch(train_data, i, self.bsz) #Get batch data with the selected targets being masked
            h = self.m_weights['representation']['model'](state)
            v, p = self.m_weights['predictions']['model'](h)
            v = rearrange(v, 'b y x -> b (y x)')
            p = rearrange(p, 'b y x -> b (y x)')
            v_loss = v_criterion(v, v_targets) #Apply loss function to results
            p_loss = p_criterion(p, p_targets) #Apply loss function to results
            loss = v_loss + p_loss
            loss.backward() #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['representation']['model'].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.m_weights['predictions']['model'].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.m_weights['dynamics']['model'].parameters(), 0.5)
            h_optimizer.step()
            f_optimizer.step()
            g_optimizer.step()
            total_loss += loss.item() #Increment total loss
            t_steps += 1
        #Updated new model
        for m in self.m_weights:
            torch.save({
                'state_dict': self.m_weights[m]['model'].state_dict(),
            }, self.m_weights[m]['param'])
        print(f'{time.time() - start_time} ms | {train_data.size(0)} samples | {total_loss / t_steps} loss\n')

    def get_batch(self, source, x, y):
        """
        Input: source - pytorch tensor containing data you wish to get batches from
               x - integer representing the index of the data you wish to gather
               y - integer representing the amount of rows you want to grab
        Description: Generate input and target data for training model
        Output: list of pytorch tensors containing input and target data [x,y]
        """
        data = torch.tensor([])
        v_target = torch.tensor([])
        p_target = torch.tensor([])
        for i in range(y):
            #Training data
            if len(source) > 0 and x+i < len(source):
                d_seq = source[x+i][:len(source[x+i])-(self.action_space+1)]
                data = torch.cat((data, d_seq))
                #Target data
                v_seq = source[x+i][-1].reshape(1)
                v_target = torch.cat((v_target, v_seq))
                p_seq = source[x+i][-(self.action_space+1):-1]
                p_target = torch.cat((p_target, p_seq))
        return (
            data.reshape(min(y, len(source[x:])),len(source[0])-(self.action_space+1)).to(torch.int64),
            v_target.reshape(min(y, len(source[x:])), 1).to(torch.float),
            p_target.reshape(min(y, len(source[x:])), self.action_space).to(torch.float)
        )
