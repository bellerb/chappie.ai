import os
import time
import json
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

from einops import rearrange
from datetime import datetime

from ai.search import MCTS
from tools.toolbox import ToolBox
from ai.model import Representation, Backbone, Head


class Agent:
    """
    Main agent interface
    """
    def __init__(self, param_name='model_param.json', train = False):
        """
        Input: param_name - string representing the file that contains the models parameters
               train - boolean control for if the AI is in training mode or not (default = False) [OPTIONAL]
        Description: Agent initail variables
        Output: None
        """
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
        ).to(self.Device)
        backbone.eval()
        Cca = ChunkedCrossAttention(
            p_model['latent_size'][-1],
            layer_size = p_model['chunked_inner'],
            heads = p_model['chunked_heads'],
            dropout = p_model['chunked_dropout'],
            n = p_model['latent_size'][0], #Sequence length
            m = p_model['chunked_length'], #Chunk length
            k = p_model['neighbour_amt'], #Amount of neighbours
            r = p_model['chunked_length'], #Retrieval length
            d = p_model['latent_size'][-1] #Embedding size
        ).to(self.Device)
        Cca.eval()
        self.m_weights = {
            'representation':{
                'model':representation,
                'param':m_param['data']['active-models']['representation']
            },
            'backbone':{
                'model':backbone,
                'param':m_param['data']['active-models']['backbone']
            },
            'cca':{
                'model':Cca,
                'param':m_param['data']['active-models']['cca']
            }
        }
        heads = {
            'policy':{
                'input':'action_space',
                'activation': torch.nn.Softmax(dim = -1)
            },
            'value':{
                'input':'value_size',
                'activation': None
            },
            'state':{
                'input':'latent_size',
                'activation': torch.nn.GELU()
            },
            'reward':{
                'input':'reward_size',
                'activation': None
            }
        }
        for h in heads:
            self.m_weights[h] = {
                'model': Head(
                    p_model[f'{heads[h]["input"]}'],
                    p_model['latent_size'],
                    inner = p_model[f'{h}_inner'],
                    heads = p_model[f'{h}_heads'],
                    dropout = p_model[f'{h}_dropout'],
                    activation = heads[h]['activation']
                ).to(self.Device),
                'param': m_param['data']['active-models'][h] if 'data' in m_param
                            and 'active-models' in m_param['data'] and h in m_param['data']['active-models']
                            else None
            }
            self.m_weights[h]['model'].eval()
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
            self.m_weights['backbone']['model'],
            self.m_weights['value']['model'],
            self.m_weights['policy']['model'],
            self.m_weights['state']['model'],
            self.m_weights['reward']['model'],
            Cca = self.m_weights['cca']['model'],
            action_space = m_param['model']['action_space'],
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
        self.epoch = m_param['training']['epoch'] #Training epochs
        self.workers = m_param['search']['workers'] #Amount of threads in search

    def choose_action(self, state, legal_moves = None, e_db = None):
        """
        Input: state - tensor containing the encoded state of a task
               legal_moves - numpy array containing the legal moves for the task (default = None) [OPTIONAL]
        Description: Choose the best action for the task
        Output: tuple containing a list of action probabilities and value of current state of game
        """
        #Expand first node of game tree
        with torch.no_grad():
            h_s = self.m_weights['representation']['model'](state)
            d = self.m_weights['backbone']['model'](h_s, torch.tensor([[0]])) #backbone function
            if e_db is not None:
                chunks = d.reshape(l, m, d)[:l - 1]
                neighbours = ToolBox.get_kNN(chunks, e_db)
                d = self.m_weights['cca']['model'](d, neighbours) #chunked cross-attention
        s_hash = self.MCTS.state_hash(d)
        self.MCTS.tree[(s_hash, None)] = self.MCTS.Node()
        self.MCTS.expand_tree(d, s_hash, None, mask = legal_moves, noise = True)
        self.MCTS.tree[(s_hash, None)].R = self.m_weights['reward']['model'](d).reshape(1).item()
        self.MCTS.tree[(s_hash, None)].N += 1
        #Run simulations
        for _ in tqdm(range(self.sim_amt),desc='MCTS'):
            self.MCTS.l = 0
            search = ToolBox.multi_thread(
                [{'name':f'search {x}', 'func':self.MCTS.search_V2, 'args':(d, self.train_control)} for x in range(self.workers)],
                workers = self.workers
            )
        #Find best move
        value = self.MCTS.tree[(s_hash, None)].Q
        reward = self.MCTS.tree[(s_hash, None)].R
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
        return (probs, value, reward)

    def train(self, data, folder = None):
        """
        Input: data - dataframe containing training data
        Description: Training of the models
        Output: dataframe containing the training log
        """
        #Initailize training
        mse = torch.nn.MSELoss() #Mean squared error loss
        bce = torch.nn.BCELoss() #Binary cross entropy loss
        h_optimizer = torch.optim.Adam(
            self.m_weights['representation']['model'].parameters(),
            lr=self.lr
        )
        g_optimizer = torch.optim.Adam(
            self.m_weights['backbone']['model'].parameters(),
            lr=self.lr
        )
        v_optimizer = torch.optim.Adam(
            self.m_weights['value']['model'].parameters(),
            lr=self.lr
        )
        p_optimizer = torch.optim.Adam(
            self.m_weights['policy']['model'].parameters(),
            lr=self.lr
        )
        s_optimizer = torch.optim.Adam(
            self.m_weights['state']['model'].parameters(),
            lr=self.lr
        )
        r_optimizer = torch.optim.Adam(
            self.m_weights['reward']['model'].parameters(),
            lr=self.lr
        )
        self.m_weights['representation']['model'].train() #Turn on the train mode
        self.m_weights['backbone']['model'].train() #Turn on the train mode
        self.m_weights['value']['model'].train() #Turn on the train mode
        self.m_weights['policy']['model'].train() #Turn on the train mode
        self.m_weights['state']['model'].train() #Turn on the train mode
        self.m_weights['reward']['model'].train() #Turn on the train mode
        t_log = []
        #Start training model
        start_time = time.time() #Get time of starting process
        for epoch in range(self.epoch):
            t_steps = 0
            total_loss = {
                'hidden loss':0.,
                'backbone loss':0.,
                'value loss':0.,
                'policy loss':0.,
                'state loss':0,
                'reward loss':0.
            }
            for batch, i in enumerate(range(0, len(data), self.bsz)):
                state, s_targets, p_targets, v_targets, r_targets, a_targets = self.get_batch(data, i, self.bsz) #Get batch data with the selected targets being masked

                h = self.m_weights['representation']['model'](state)
                d = self.m_weights['backbone']['model'](h, a_targets)
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

                total_loss['backbone loss'] += d_loss.item()
                g_optimizer.zero_grad()
                d_loss.backward(
                    retain_graph = True,
                    inputs = list(self.m_weights['backbone']['model'].parameters())
                ) #Backpropegate through model
                torch.nn.utils.clip_grad_norm_(self.m_weights['backbone']['model'].parameters(), 0.5)
                g_optimizer.step()

                h = self.m_weights['representation']['model'](state)
                d = self.m_weights['backbone']['model'](h, a_targets)
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
            print(f'EPOCH {epoch} | {time.time() - start_time} ms | {len(data)} samples | {"| ".join(f"{v/t_steps} {k}" for k, v in total_loss.items())}\n')
            t_log.append({
                **{
                    'Date':datetime.now(),
                    'Epoch':epoch,
                    'Samples':len(data),
                    'Time':time.time() - start_time
                },
                **{k:(v/t_steps) for k,v in total_loss.items()}
            })
        #Updated new model
        if folder is not None and os.path.exists(f'{folder}/weights') == False:
            os.makedirs(f'{folder}/weights') #Create folder
        for m in self.m_weights:
            torch.save({
                'state_dict': self.m_weights[m]['model'].state_dict(),
            }, f"{folder}/weights/{self.m_weights[m]['param']}" if folder is not None else self.m_weights[m]['param'])
        return t_log

    def get_batch(self, source, x, y, v_h = 'value', r_h = 'reward', s_h = 'state', p_h = 'prob', a_h = 'action'):
        """
        Input: source - dataframe containing training data for the model
               x - integer representing the index of the data you wish to gather
               y - integer representing the amount of rows you want to grab
               v_h - string representing the value header (default = 'value') [OPTIONAL]
               r_h - string representing the reward header (default = 'reward') [OPTIONAL]
               s_h - string representing the state header (default = 'state') [OPTIONAL]
               a_h - string representing the action header (default = 'action') [OPTIONAL]
        Description: Get batch of training data
        Output: tuple of pytorch tensors containing input and target data [x, x + 1, p, v, r, a]
        """
        v_headers = [v_h]
        r_headers = [r_h]
        a_headers = [a_h]
        s_headers = [h for h in source if s_h in h]
        p_headers = [h for h in source if p_h in h]

        s = source[s_headers].iloc[x:x+y]
        p = source[p_headers].iloc[x:x+y]
        v = source[v_headers].iloc[x:x+y]
        r = source[r_headers].iloc[x:x+y]
        a = source[a_headers].iloc[x:x+y] + 1

        a_0 = pd.DataFrame([{a_h:0} for _ in range(len(a))])

        s_1 = source[s_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in s_1.iloc[-1].isna().tolist():
            s_1.iloc[-1] = s.iloc[-1]
            s_1[f'{s_h}0'].iloc[-1] = 0 if s_1[f'{s_h}0'].iloc[-1] == 1 else 1

        p_1 = source[p_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in p_1.iloc[-1].isna().tolist():
            p_1.iloc[-1] = p.iloc[-1]

        v_1 = source[v_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in v_1.iloc[-1].isna().tolist():
            v_1.iloc[-1] = v.iloc[-1]

        r_1 = source[r_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in r_1.iloc[-1].isna().tolist():
            r_1.iloc[-1] = r.iloc[-1]

        state = s.append(s, ignore_index = True)
        state = torch.tensor(state.values)

        s_target = s.append(s_1, ignore_index = True)
        s_target = torch.tensor(s_target.values)

        p_target = p.append(p_1, ignore_index = True)
        p_target = torch.tensor(p_target.values)

        v_target = v.append(v_1, ignore_index = True)
        v_target = torch.tensor(v_target.values)

        r_target = r.append(r_1, ignore_index = True)
        r_target = torch.tensor(r_target.values)

        a_target = a_0.append(a, ignore_index = True)
        a_target = torch.tensor(a_target.values)
        return (
            state.to(torch.int64),
            s_target.to(torch.int64),
            p_target.to(torch.float),
            v_target.to(torch.float),
            r_target.to(torch.float),
            a_target.to(torch.int64)
        )
