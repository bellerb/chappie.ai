import os
import time
import json
import torch
import random
import numpy as np

from tqdm import tqdm

from einops import rearrange
from datetime import datetime

from ai.search import MCTS
from ai.model import Representation, Dynamics, Value, Policy, NextState, Reward

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
            ntoken = p_model['ntoken'],
            embedding_size = p_model['embedding_size'],
            padding_idx = p_model['padding_idx'],
            encoder_dropout = p_model['encoder_dropout'],
            h_inner = p_model['h_inner'],
            h_heads = p_model['h_heads'],
            h_dropout = p_model['h_dropout']
        ).to(self.Device)
        representation.eval()
        dynamics = Dynamics(
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
        ).to(self.Device)
        dynamics.eval()
        policy = Policy(
            p_model['action_space'],
            p_model['latent_size'],
            policy_inner = p_model['policy_inner'],
            policy_heads = p_model['policy_heads'],
            policy_dropout = p_model['policy_dropout']
        ).to(self.Device)
        policy.eval()
        value = Value(
            p_model['value_size'],
            p_model['latent_size'],
            value_inner = p_model['value_inner'],
            value_heads = p_model['value_heads'],
            value_dropout = p_model['value_dropout']
        ).to(self.Device)
        value.eval()
        state = NextState(
            p_model['latent_size'],
            p_model['latent_size'],
            state_k_inner = p_model['state_k_inner'],
            state_k_heads = p_model['state_k_heads'],
            state_k_dropout = p_model['state_k_dropout']
        ).to(self.Device)
        state.eval()
        reward = Reward(
            p_model['reward_size'],
            p_model['latent_size'],
            reward_inner = p_model['reward_inner'],
            reward_heads = p_model['reward_heads'],
            reward_dropout = p_model['reward_dropout']
        ).to(self.Device)
        reward.eval()
        self.m_weights = {
            'representation':{
                'model':representation,
                'param':m_param['data']['active-models']['representation']
            },
            'dynamics':{
                'model':dynamics,
                'param':m_param['data']['active-models']['dynamics']
            },
            'value':{
                'model':value,
                'param':m_param['data']['active-models']['value']
            },
            'policy':{
                'model':policy,
                'param':m_param['data']['active-models']['policy']
            },
            'state':{
                'model':state,
                'param':m_param['data']['active-models']['state']
            },
            'reward':{
                'model':reward,
                'param':m_param['data']['active-models']['reward']
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
            self.m_weights['dynamics']['model'],
            self.m_weights['value']['model'],
            self.m_weights['policy']['model'],
            self.m_weights['state']['model'],
            self.m_weights['reward']['model'],
            c2 = m_param['search']['c2'],
            d_a = m_param['search']['d_a'],
            e_f = m_param['search']['e_f'],
            g_d = m_param['search']['g_d'],
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
            d = self.m_weights['dynamics']['model'](h_s, torch.tensor([[0]])) #dynamics function
        for _ in tqdm(range(self.sim_amt),desc='MCTS'):
            self.MCTS.depth = 0
            self.MCTS.search(d, train = self.train_control)
        s_hash = self.MCTS.state_hash(d)
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
        mse = torch.nn.MSELoss() #Mean squared error loss
        bce = torch.nn.BCELoss() #Binary cross entropy loss
        h_optimizer = torch.optim.SGD(
            self.m_weights['representation']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        g_optimizer = torch.optim.SGD(
            self.m_weights['dynamics']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        v_optimizer = torch.optim.SGD(
            self.m_weights['value']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        p_optimizer = torch.optim.SGD(
            self.m_weights['policy']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        s_optimizer = torch.optim.SGD(
            self.m_weights['state']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        r_optimizer = torch.optim.SGD(
            self.m_weights['reward']['model'].parameters(),
            lr=self.lr
        ) #Optimization algorithm using stochastic gradient descent
        self.m_weights['representation']['model'].train() #Turn on the train mode
        self.m_weights['dynamics']['model'].train() #Turn on the train mode
        self.m_weights['value']['model'].train() #Turn on the train mode
        self.m_weights['policy']['model'].train() #Turn on the train mode
        self.m_weights['state']['model'].train() #Turn on the train mode
        self.m_weights['reward']['model'].train() #Turn on the train mode
        train_data = torch.tensor(data.values) #Set training data to a tensor
        #Start training model
        t_steps = 0
        total_loss = {
            'hidden loss':0.,
            'dynamics loss':0.,
            'value loss':0.,
            'policy loss':0.,
            'state loss':0,
            'reward loss':0.
        }
        start_time = time.time() #Get time of starting process
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bsz)):
            state, s_targets, p_targets, v_targets, r_targets = self.get_batch(train_data, i, self.bsz) #Get batch data with the selected targets being masked
            h = self.m_weights['representation']['model'](state)

            a = torch.argmax(p_targets, dim = -1)
            for j in range(a.size(-1)-1):
                a[len(a) - j - 1] = a[len(a) - j - 2]
            a[0] = 0
            a = rearrange(a, '(y x) -> y x ', x = 1)

            d = self.m_weights['dynamics']['model'](h, a)
            v = self.m_weights['value']['model'](d)
            p = self.m_weights['policy']['model'](d)
            s = self.m_weights['state']['model'](d)
            r = self.m_weights['reward']['model'](d)
            s_h = self.m_weights['representation']['model'](s_targets)

            v = rearrange(v, 'b y x -> b (y x)')
            p = rearrange(p, 'b y x -> b (y x)')
            r = rearrange(r, 'b y x -> b (y x)')

            v_loss = mse(v, v_targets) #Apply loss function to results
            p_loss = bce(p, p_targets) #Apply loss function to results
            r_loss = mse(r, r_targets) #Apply loss function to results
            s_loss = mse(s, s_h) #Apply loss function to results
            h_loss = v_loss.clone() + p_loss.clone() + r_loss.clone()
            d_loss = v_loss.clone() + p_loss.clone() + r_loss.clone() + s_loss.clone()

            h_optimizer.zero_grad()
            total_loss['hidden loss'] += h_loss.item()
            h_loss.backward(
                retain_graph = True,
                inputs = list(self.m_weights['representation']['model'].parameters())
            ) #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['representation']['model'].parameters(), 0.5)
            h_optimizer.step()

            total_loss['dynamics loss'] += d_loss.item()
            g_optimizer.zero_grad()
            d_loss.backward(
                retain_graph = True,
                inputs = list(self.m_weights['dynamics']['model'].parameters())
            ) #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['dynamics']['model'].parameters(), 0.5)
            g_optimizer.step()

            total_loss['value loss'] += v_loss.item()
            v_optimizer.zero_grad()
            v_loss.backward(
                inputs = list(self.m_weights['value']['model'].parameters())
            ) #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['value']['model'].parameters(), 0.5)
            v_optimizer.step()

            total_loss['policy loss'] += p_loss.item()
            p_optimizer.zero_grad()
            p_loss.backward(
                inputs = list(self.m_weights['policy']['model'].parameters())
            ) #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['policy']['model'].parameters(), 0.5)
            p_optimizer.step()

            total_loss['state loss'] += s_loss.item()
            s_optimizer.zero_grad()
            s_loss.backward(
                inputs = list(self.m_weights['state']['model'].parameters())
            ) #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['state']['model'].parameters(), 0.5)
            s_optimizer.step()

            total_loss['reward loss'] += r_loss.item()
            r_optimizer.zero_grad()
            r_loss.backward(
                inputs = list(self.m_weights['reward']['model'].parameters())
            ) #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(self.m_weights['reward']['model'].parameters(), 0.5)
            r_optimizer.step()

            t_steps += 1
        #Updated new model
        for m in self.m_weights:
            torch.save({
                'state_dict': self.m_weights[m]['model'].state_dict(),
            }, self.m_weights[m]['param'])
        t_log = {**{'Date':datetime.now(),'Samples':train_data.size(0),'Time':time.time() - start_time},**{k:(v/t_steps) for k,v in total_loss.items()}}
        print(f'{time.time() - start_time} ms | {train_data.size(0)} samples | {"| ".join(f"{v/t_steps} {k}" for k, v in total_loss.items())}')
        return t_log

    def get_batch(self, source, x, y):
        """
        Input: source - pytorch tensor containing data you wish to get batches from
               x - integer representing the index of the data you wish to gather
               y - integer representing the amount of rows you want to grab
        Description: Generate input and target data for training model
        Output: tuple of pytorch tensors containing input and target data [x, p, v, r]
        """
        data = torch.tensor([])
        v_target = torch.tensor([])
        r_target = torch.tensor([])
        p_target = torch.tensor([])
        s_target = torch.tensor([])
        for i in range(y):
            if len(source) > 0 and x + i < len(source):
                d_seq = source[x + i][:len(source[x + i]) - (self.action_space + 2)]
                data = torch.cat((data, d_seq))
                if x + i < len(source) - 1:
                    s_seq = source[x + i + 1][:len(source[x + i + 1]) - (self.action_space + 2)]
                else:
                    s_seq = source[x + i][:len(source[x + i]) - (self.action_space + 2)]
                s_target = torch.cat((s_target, s_seq))
                v_seq = source[x + i][-2].reshape(1)
                v_target = torch.cat((v_target, v_seq))
                r_seq = source[x + i][-1].reshape(1)
                r_target = torch.cat((r_target, r_seq))
                p_seq = source[x + i][-(self.action_space + 2):-2]
                p_target = torch.cat((p_target, p_seq))
        return (
            data.reshape(min(y, len(source[x:])), len(source[0]) - (self.action_space + 2)).to(torch.int64),
            s_target.reshape(min(y, len(source[x:])), len(source[0]) - (self.action_space + 2)).to(torch.int64),
            p_target.reshape(min(y, len(source[x:])), self.action_space).to(torch.float),
            v_target.reshape(min(y, len(source[x:])), 1).to(torch.float),
            r_target.reshape(min(y, len(source[x:])), 1).to(torch.float)
        )

