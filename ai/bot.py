import os
import time
import json
import random
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from einops import rearrange

from ai.search import MCTS
from tools.toolbox import ToolBox
from ai.model import Representation, Backbone, Head, ChunkedCrossAttention


class Agent:
    """
    Main agent interface
    """
    def __init__(self, param_name='model_param.json', train = False, game_num = 0):
        """
        Input: param_name - string representing the file that contains the models parameters
               train - boolean control for if the AI is in training mode or not (default = False) [OPTIONAL]
               game_num - amount of games that have been trained (default = 0) [OPTIONAL]
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
        if p_model['retro'] is True:
            Cca = ChunkedCrossAttention(
                p_model['latent_size'][-1],
                layer_size = p_model['chunked_inner'],
                heads = p_model['chunked_heads'],
                dropout = p_model['chunked_dropout'],
                n = p_model['latent_size'][0], #Sequence length
                m = p_model['chunked_length'], #Chunk length
                k = p_model['neighbour_amt'], #Amount of neighbours
                r = p_model['latent_size'][0], #Retrieval length
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
            }

        }
        if p_model['retro'] is True:
            self.m_weights['cca'] = {
                'model':Cca,
                'param':m_param['data']['active-models']['cca']
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
            Cca = self.m_weights['cca']['model'] if p_model['retro'] is True else None,
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
            self.noise = True
        else:
            self.noise = False
        #Decay tempature with as more training games played to cause pUCT formula to become more exploitative
        self.T = 1 * ((1 - m_param['search']['T']) ** game_num) #Decay tempature using exponential decay formula
        self.move_count = m_param['search']['move_cache'] #Amount of moves to cache
        self.sim_amt = m_param['search']['sim_amt'] #Amount of simulations to run
        self.workers = m_param['search']['workers'] #Amount of threads in search
        self.training_settings =  m_param['training'] #Training settings
        #self.single_player = m_param['search']['single_player']
        self.tools = ToolBox()
        if p_model['retro'] is True:
            MAX_EDB = m_param['search']['max_embedding_data'] #Max amount of embeddings to use in CCA
            e_db_file_name = f"{'/'.join(s for s in param_name.split('/')[:-1])}/logs/encoded_data.csv".replace('(temp)','')
            if os.path.exists(e_db_file_name):
                self.E_DB = self.tools.read_n_from_bottom_csv(
                    e_db_file_name, 
                    MAX_EDB, 
                    headers = ['encoding', 'state']
                )
                self.E_DB['state'] = self.E_DB['state'].apply(eval).apply(np.array)
                self.E_DB['encoding'] = self.E_DB['encoding'].apply(eval).apply(np.array)
            else:
                self.E_DB = self.tools.build_embedding_db(
                    representation,
                    backbone,
                    f_name = f"{'/'.join(s for s in param_name.split('/')[:-1])}/logs/game_log.csv".replace('(temp)','')
                )
                self.E_DB.to_csv(f"{'/'.join(s for s in param_name.split('/')[:-1])}/logs/encoded_data.csv".replace('(temp)',''), index=False)
                self.E_DB = self.E_DB.iloc[-MAX_EDB:]
        else:
            self.E_DB = None
        self.total_loss = {}
        torch.autograd.set_detect_anomaly(True)

    def choose_action(self, state, legal_moves = None):
        """
        Input: state - tensor containing the encoded state of a task
               legal_moves - numpy array containing the legal moves for the task (default = None) [OPTIONAL]
        Description: Choose the best action for the task
        Output: tuple containing a list of action probabilities and value of current state of game
        """
        #Expand first node of game tree
        with torch.no_grad():
            h_s = self.m_weights['representation']['model'](state)
            d = self.m_weights['backbone']['model'](h_s, torch.zeros(1,1).to(torch.long)) #backbone function
            if self.E_DB is not None:
                #Perform chunked cross attention
                d = torch.squeeze(d)
                chunks = d.reshape(
                    self.m_weights['cca']['model'].l,
                    self.m_weights['cca']['model'].m,
                    self.m_weights['cca']['model'].d
                )[:self.m_weights['cca']['model'].l - 1]
                neighbours = self.tools.get_kNN(
                    chunks,
                    self.E_DB[~self.E_DB['state'].isin(state.float().tolist())]
                )
                d = self.m_weights['cca']['model'](d, neighbours) #chunked cross-attention
                d = d.reshape(1, d.size(0), d.size(1))
        s_hash = self.MCTS.state_hash(d)
        self.MCTS.tree[(s_hash, None)] = self.MCTS.Node()
        self.MCTS.expand_tree(d, s_hash, None, mask = legal_moves, noise = self.noise)
        self.MCTS.tree[(s_hash, None)].R = self.m_weights['reward']['model'](d).reshape(1).item()
        self.MCTS.tree[(s_hash, None)].N += 1
        #Run simulations
        for _ in tqdm(range(self.sim_amt),desc='MCTS'):
            self.MCTS.l = 0
            search = self.tools.multi_thread(
                [{'name':f'search {x}', 'func':self.MCTS.search, 'args':(d, self.E_DB)} for x in range(self.workers)],
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
            c_s = sum(c ** (1. / self.T) for c in counts.values())
            probs = [(x ** (1. / self.T)) / c_s for x in counts.values()]
        self.MCTS.tree = {}
        return (probs, value, reward)

    def init_model_4_training(self, optimizer, scheduler, model_name, step_header, gamma_header, m_header='model', lr_header='lr'):
        """
        Input: optimizer - string representing the name of the optimizer variable for the model
               scheduler - string representing the name of the scheduler variable for the model
               model_name - string representing the name of the model
               step_header - string representing the header for the step setting
               gamma_header - string representing the header of the gamma decay setting
               m_header - string representing the model header (Default = 'model') [OPTIONAL]
               lr_header - string representing the learning rate header (Default ='lr') [OPTIONAL]
        Description: initalizes a model for training purposes
        Output: None
        """
        self.__dict__[optimizer] = torch.optim.Adam(
            self.m_weights[model_name][m_header].parameters(),
            lr=self.training_settings[lr_header]
        )
        self.__dict__[scheduler] = torch.optim.lr_scheduler.StepLR(
            self.__dict__[optimizer],
            self.training_settings[step_header],
            gamma=self.training_settings[gamma_header]
        )
        self.m_weights[model_name][m_header].train()

    def train_layer(self, layer_name, param, data):
        """
        Input: layer_name - string representing the name of the layer to load
               param - dictionary containing training parameters
               data - dataframe containing training data
        Description: Module for training a layer of the model
        Output: list containing the training log
        """
        if 'bsz' not in param or 'epoch' not in param:
            raise Exception('Error - found missing parameters, "bsz" and "epoch" needed.')
        training_function = getattr(Agent, f'train_{layer_name}_layer')
        start_time, log = time.time(), []
        for epoch in range(param['epoch']):
            t_steps, loss = 0, 0.
            with tqdm(total=int(len(data) / param['bsz']) + 1, desc='Training Batch') as pbar:
                #print(len(data), param['bsz'])
                for batch, i in enumerate(range(0, len(data), param['bsz'])):
                    #print(data)
                    state, s_targets, p_targets, v_targets, r_targets, a_targets = self.get_batch(data, i, param['bsz'])
                    #print('atarge',a_targets.size(),s_targets.size())
                    #Run model_specific training function
                    training_variables = [self]
                    for k in training_function.__code__.co_varnames[:training_function.__code__.co_argcount]:
                        #print(k)
                        if k == 'self': continue
                        if k in locals():
                            training_variables.append(locals()[k])
                        else:
                            v, p, r, s, s_h = self.forward_pass(state, a_targets, s_targets)
                            if k in locals():
                                training_variables.append(locals()[k])
                        #print(f'k = {locals()[k].size()}')
                    training_function(*training_variables)
                    t_steps += 1
                    pbar.update(1)
                    #Garbage clean up 
                    if 'v' in locals(): del v
                    if 'p' in locals(): del p
                    if 'r' in locals(): del r
                    if 's' in locals(): del s
                    if 's_h' in locals(): del s_h
            print(f'EPOCH {epoch} | {time.time() - start_time} ms | {len(data)} samples | {self.total_loss[f"{layer_name} loss"] / t_steps} loss\n')
            log.append({
                'Date':datetime.now(),
                'Epoch':epoch,
                'Samples':len(data),
                'Time':time.time() - start_time,
                f'{layer_name} loss': self.total_loss[f"{layer_name} loss"] / t_steps
            })
        return log

    def train(self, data, folder = None, encoder = True, full_count = 1):
        """
        Input: data - dataframe containing training data
               folder -  string representing the folder to save the new weights in (Default = None) [OPTIONAL]
               encoder - boolean representing if you want to retrain the encoder models or not (Default = True) [OPTIONAL]
               full_count - integer representing the amoun  t of epochs to do full model training for (Default = 1) [OPTIONAL]
        Description: Training of the models
        Output: dataframe containing the training log
        """
        #Training loss functions
        self.mse = torch.nn.MSELoss() #Mean squared error loss
        self.bce = torch.nn.BCELoss() #Binary cross entropy loss
        #Load model weights
        self.init_model_4_training('h_optimizer', 'h_scheduler', 'representation', 'h_step', 'h_gamma') #Hidden layer settings
        self.init_model_4_training('b_optimizer', 'b_scheduler', 'backbone', 'b_step', 'b_gamma') #Backbone layer settings
        if self.E_DB is not None:
            self.init_model_4_training('c_optimizer', 'c_scheduler', 'cca', 'c_step', 'c_gamma') #Chunked cross-attention layer settings
        self.init_model_4_training('v_optimizer', 'v_scheduler', 'value', 'v_step', 'v_gamma') #Value head settings
        self.init_model_4_training('p_optimizer', 'p_scheduler', 'policy', 'p_step', 'p_gamma') #Policy head settings
        self.init_model_4_training('s_optimizer', 's_scheduler', 'state', 's_step', 's_gamma') #Next state representation head settings
        self.init_model_4_training('r_optimizer', 'r_scheduler', 'reward', 'r_step', 'r_gamma') #Reward settings
        #Start training model
        t_log = [] #Training log
        start_time = time.time() #Get time of starting process
        for epoch in range(self.training_settings['epoch']):
            t_steps = 0 #Current training step
            self.total_loss = {'value loss':0., 'policy loss':0., 'state loss':0, 'reward loss':0.}
            if encoder is True and epoch <= full_count - 1:
                self.total_loss['representation loss'] = 0.
                self.total_loss['backbone loss'] = 0.
            if self.E_DB is not None:
                self.total_loss['cca loss'] = 0.
            if encoder is True and epoch > full_count - 1:
                data = data[data['Game-ID']==data.iloc[-1]['Game-ID']]
            with tqdm(total=int(len(data)/self.training_settings['bsz']) + 1, desc='Training Batch') as pbar:
                for batch, i in enumerate(range(0, len(data), self.training_settings['bsz'])):
                    state, s_targets, p_targets, v_targets, r_targets, a_targets = self.get_batch(data, i, self.training_settings['bsz']) #Get batch data with the selected targets being masked
                    if epoch <= full_count - 1 and encoder is True:
                        '''
                        #Train trunk of model
                        if self.E_DB is not None:
                            self.tools.multi_thread(
                                [
                                    {'name':'representation', 'func':self.train_representation_layer, 'args':(state, a_targets, s_targets, v_targets, p_targets, r_targets)},
                                    {'name':'cca', 'func':self.train_cca_layer, 'args':(state, a_targets, s_targets, v_targets, p_targets, r_targets)},
                                    {'name':'backbone', 'func':self.train_backbone_layer, 'args':(state, a_targets, s_targets, v_targets, p_targets, r_targets)}
                                ],
                                workers = 2
                            )
                        else:
                            self.tools.multi_thread(
                                [
                                    {'name':'representation', 'func':self.train_representation_layer, 'args':(state, a_targets, s_targets, v_targets, p_targets, r_targets)},
                                    {'name':'backbone', 'func':self.train_backbone_layer, 'args':(state, a_targets, s_targets, v_targets, p_targets, r_targets)}
                                ],
                                workers = 2
                            )
                        '''
                        #Train trunk of model
                        self.train_representation_layer(state, a_targets, s_targets, v_targets, p_targets, r_targets)
                        #print('rep')
                        self.train_backbone_layer(state, a_targets, s_targets, v_targets, p_targets, r_targets)
                        #print('back')
                    if self.E_DB is not None:
                        self.train_cca_layer(state, a_targets, s_targets, v_targets, p_targets, r_targets)
                        #print('cca')

                    #Train heads of model
                    v, p, r, s, s_h = self.forward_pass(state, a_targets, s_targets)
                    '''
                    self.tools.multi_thread(
                        [
                            {'name':'value', 'func':self.train_value_layer, 'args':(v, v_targets)},
                            {'name':'policy', 'func':self.train_policy_layer, 'args':(p, p_targets)},
                            {'name':'state', 'func':self.train_next_state_layer, 'args':(s, s_h)},
                            {'name':'reward', 'func':self.train_reward_layer, 'args':(r, r_targets)}
                        ],
                        workers = 2
                    )
                    '''
                    self.train_value_layer(v, v_targets)
                    #print('val')
                    self.train_policy_layer(p, p_targets)
                    #print('pol')
                    self.train_state_layer(s, s_h)
                    #print('state')
                    self.train_reward_layer(r, r_targets)
                    #print('reward')
                    t_steps += 1
                    pbar.update(1)
            #Learning rate decay
            if epoch <= full_count - 1 and encoder is True:
                self.h_scheduler.step()
                self.b_scheduler.step()
            if self.E_DB is not None:
                self.c_scheduler.step()
            self.v_scheduler.step()
            self.p_scheduler.step()
            self.r_scheduler.step()
            self.s_scheduler.step()
            print(f'EPOCH {epoch} | {time.time() - start_time} ms | {len(data)} samples | {"| ".join(f"{v/t_steps} {k}" for k, v in self.total_loss.items())}\n')
            t_log.append({
                **{
                    'Date':datetime.now(),
                    'Epoch':epoch,
                    'Samples':len(data),
                    'Time':time.time() - start_time
                },
                **{k:(v / t_steps) for k, v in self.total_loss.items()}
            })
        #Updated new model
        if folder is not None and os.path.exists(f'{folder}/weights') is False:
            os.makedirs(f'{folder}/weights') #Create folder
        for m in self.m_weights:
            torch.save({
                'state_dict': self.m_weights[m]['model'].state_dict(),
            }, f"{folder}/weights/{self.m_weights[m]['param']}" if folder is not None else self.m_weights[m]['param'])
        return t_log

    def forward_pass(self, state, a_targets, s_targets):
        """
        Input: state - tensor representing the current state
               a_targets - tensor representing the target actions
               s_targets - tensor representing the state targets
        Description: forward pass of the model
        Output: tuple containing the output of the models heads
        """
        #Model trunk
        h = self.m_weights['representation']['model'](state)
        #print("AAAAA",h.size(),a_targets.size(),state.size())
        d = self.m_weights['backbone']['model'](h, a_targets)
        if self.E_DB is not None:
            d_hold = torch.tensor([])
            for i, row in enumerate(d):
                chunks = row.reshape(
                    self.m_weights['cca']['model'].l,
                    self.m_weights['cca']['model'].m,
                    self.m_weights['cca']['model'].d
                )[:self.m_weights['cca']['model'].l - 1]
                neighbours = self.tools.get_kNN(chunks, self.E_DB)
                c = self.m_weights['cca']['model'](row, neighbours) #chunked cross-attention
                c = c.reshape(1, c.size(0), c.size(1))
                d_hold = torch.cat([d_hold, c])
            d = d_hold
            del d_hold
        #Model heads
        v = self.m_weights['value']['model'](d)
        v = rearrange(v, 'b y x -> b (y x)')
        p = self.m_weights['policy']['model'](d)
        p = rearrange(p, 'b y x -> b (y x)')
        r = self.m_weights['reward']['model'](d)
        r = rearrange(r, 'b y x -> b (y x)')
        s = self.m_weights['state']['model'](d)
        s_h = self.m_weights['representation']['model'](s_targets)
        return v, p, r, s, s_h

    def train_representation_layer(self, state, a_targets, s_targets, v_targets, p_targets, r_targets):
        """
        Input: state - tensor representing the current state
               a_targets - tensor representing the choosen actions
               s_targets - tensor representing the next state head targets
               v_targets - tensor representing the value head targets
               p_targets - tensor representing the policy head targets
               r_targets - tensor representing the reward head targets
        Description: updating of the representation layers weights
        Output: None
        """
        #print(f'self = {self}')
        v, p, r, s, s_h = self.forward_pass(state, a_targets, s_targets)
        v_loss = self.mse(v, v_targets) #Apply loss function to results
        p_loss = self.bce(p, p_targets) #Apply loss function to results
        r_loss = self.mse(r, r_targets) #Apply loss function to results
        #s_loss = self.mse(s[:len(s_h)], s_h) #Apply loss function to results
        h_loss = v_loss.clone() + p_loss.clone() + r_loss.clone()
        #Update hidden layer weights
        self.h_optimizer.zero_grad()
        if 'representation loss' in self.total_loss:
            self.total_loss['representation loss'] += h_loss.item()
        else:
            self.total_loss['representation loss'] = h_loss.item()
        h_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['representation']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['representation']['model'].parameters(),
            self.training_settings['h_max_norm']
        )
        self.h_optimizer.step()

    def train_backbone_layer(self, state, a_targets, s_targets, v_targets, p_targets, r_targets):
        """
        Input: state - tensor representing the current state
               a_targets - tensor representing the choosen actions
               s_targets - tensor representing the next state head targets
               v_targets - tensor representing the value head targets
               p_targets - tensor representing the policy head targets
               r_targets - tensor representing the reward head targets
        Description: updating of the backbone layers weights
        Output: None
        """
        h_0 = self.m_weights['representation']['model'](state[:len(s_targets)])
        #d_0 = self.m_weights['backbone']['model'](h_0, a_targets[len(s_targets):])
        d_0 = self.m_weights['backbone']['model'](h_0, a_targets)

        h_1 = self.m_weights['representation']['model'](s_targets)
        #d_1 = self.m_weights['backbone']['model'](h_1, torch.tensor([[0]] * len(s_targets)))
        d_1 = self.m_weights['backbone']['model'](h_1, torch.tensor([[0]] * len(a_targets)))

        d_loss = self.mse(d_0, d_1) #Apply loss function to results
        if 'backbone loss' in self.total_loss:
            self.total_loss['backbone loss'] += d_loss.item()
        else:
            self.total_loss['backbone loss'] = d_loss.item()
        self.b_optimizer.zero_grad()
        d_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['backbone']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['backbone']['model'].parameters(),
            self.training_settings['b_max_norm']
        )
        self.b_optimizer.step()

    def train_cca_layer(self, state, a_targets, s_targets, v_targets, p_targets, r_targets):
        """
        Input: state - tensor representing the current state
               a_targets - tensor representing the choosen actions
               s_targets - tensor representing the next state head targets
               v_targets - tensor representing the value head targets
               p_targets - tensor representing the policy head targets
               r_targets - tensor representing the reward head targets
        Description: updating of the chunked cross-attention layers weights
        Output: None
        """
        v, p, r, s, s_h = self.forward_pass(state, a_targets, s_targets)
        v_loss = self.mse(v, v_targets) #Apply loss function to results
        p_loss = self.bce(p, p_targets) #Apply loss function to results
        r_loss = self.mse(r, r_targets) #Apply loss function to results
        s_loss = self.mse(s[:len(s_h)], s_h) #Apply loss function to results
        cca_loss = v_loss.clone() + p_loss.clone() + r_loss.clone() + s_loss.clone()
        #Update chunked cross-attention layer weights
        if 'cca loss' in self.total_loss:
            self.total_loss['cca loss'] += cca_loss.item()
        else:
            self.total_loss['cca loss'] = cca_loss.item()
        self.c_optimizer.zero_grad()
        cca_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['cca']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['cca']['model'].parameters(),
            self.training_settings['c_max_norm']
        )
        self.c_optimizer.step()

    def train_value_layer(self, v, v_targets):
        """
        Input: v - tensor representing the predicted value
               v_targets - tensor representing the value head targets
        Description: updating of the value layers weights
        Output: None
        """
        v_loss = self.mse(v, v_targets) #Apply loss function to results
        if 'value loss' in self.total_loss:
            self.total_loss['value loss'] += v_loss.item()
        else:
            self.total_loss['value loss'] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['value']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['value']['model'].parameters(),
            self.training_settings['v_max_norm']
        )
        self.v_optimizer.step()

    def train_policy_layer(self, p, p_targets):
        """
        Input: p - tensor representing the predicted policy
               p_targets - tensor representing the value head targets
        Description: updating of the policy layers weights
        Output: None
        """
        p_loss = self.bce(p, p_targets) #Apply loss function to results
        if 'policy loss' in self.total_loss:
            self.total_loss['policy loss'] += p_loss.item()
        else:
            self.total_loss['policy loss'] = p_loss.item()
        self.p_optimizer.zero_grad()
        p_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['policy']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['policy']['model'].parameters(),
            self.training_settings['p_max_norm']
        )
        self.p_optimizer.step()

    def train_reward_layer(self, r, r_targets):
        """
        Input: r - tensor representing the predicted reward
               r_targets - tensor representing the reward head targets
        Description: updating of the value layers weights
        Output: None
        """
        r_loss = self.mse(r, r_targets) #Apply loss function to results
        if 'reward loss' in self.total_loss:
            self.total_loss['reward loss'] += r_loss.item()
        else:
            self.total_loss['reward loss'] = r_loss.item()
        self.r_optimizer.zero_grad()
        r_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['reward']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['reward']['model'].parameters(),
            self.training_settings['r_max_norm']
        )
        self.r_optimizer.step()

    def train_state_layer(self, s, s_h):
        """
        Input: s - tensor representing the predicted next state representation
               s_h - tensor representing the prediction representation encodings
        Description: updating of the next state layers weights using self supervision
        Output: None
        """
        s_loss = self.mse(s[:len(s_h)], s_h) #Apply loss function to results
        if 'state loss' in self.total_loss:
            self.total_loss['state loss'] += s_loss.item()
        else:
            self.total_loss['state loss'] = s_loss.item()
        self.s_optimizer.zero_grad()
        s_loss.backward(
            retain_graph = True,
            inputs = list(self.m_weights['state']['model'].parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            self.m_weights['state']['model'].parameters(),
            self.training_settings['s_max_norm']
        )
        self.s_optimizer.step()

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

        #print("ACTION",x, y)

        a_0 = pd.DataFrame([{a_h:0} for _ in range(len(a))])

        s_1 = source[s_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        #s_1[f'{s_h}0'] = np.where(s_1[f'{s_h}0'] == 0., 1., 0.)
        if True in s_1.iloc[-1].isna().tolist():
            s_1.iloc[-1] = s.iloc[-1]
            #if self.single_player is False:
                #s_1[f'{s_h}0'].iloc[-1] = 0 if s_1[f'{s_h}0'].iloc[-1] == 1 else 1

        p_1 = source[p_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in p_1.iloc[-1].isna().tolist():
            p_1.iloc[-1] = p.iloc[-1]

        v_1 = source[v_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in v_1.iloc[-1].isna().tolist():
            v_1.iloc[-1] = v.iloc[-1]

        r_1 = source[r_headers].shift(periods = -1, axis = 0).iloc[x:x+y]
        if True in r_1.iloc[-1].isna().tolist():
            r_1.iloc[-1] = r.iloc[-1]

        state = pd.concat([s, s], ignore_index=True)
        state = torch.tensor(state.values)

        s_target = pd.concat([s_1, s_1], ignore_index=True)
        s_target = torch.tensor(s_target.values)

        p_target = pd.concat([p, p_1], ignore_index=True)
        p_target = torch.tensor(p_target.values)

        v_target = pd.concat([v, v_1], ignore_index=True)
        v_target = torch.tensor(v_target.values)

        r_target = pd.concat([r, r_1], ignore_index = True)
        r_target = torch.tensor(r_target.values)

        a_target = pd.concat([a_0, a], ignore_index = True)
        #a_target = a_0.append(a, ignore_index = True)
        a_target = torch.tensor(a_target.values)
        return (
            state.to(torch.int64),
            s_target.to(torch.int64),
            p_target.to(torch.float),
            v_target.to(torch.float),
            r_target.to(torch.float),
            a_target.to(torch.int64)
        )

    def model_parameters(self, layers, trainable = False):
        """
        Input: layers - dictionary containing full model layers
               trainable - boolean representing if you want to count only the trainable parameters (Default = False) [OPTIONAL]
        Description: Count total amount of parameters in a model
        Output: dictionary containing model parameters count
        """
        result = {}
        full_model_param = 0
        for m in layers:
            result[m] = sum(p.numel() for p in layers[m]['model'].parameters() if p.requires_grad) \
                if trainable is True else sum(p.numel() for p in layers[m]['model'].parameters())
        result['full'] = sum(v for v in result.values())
        return result
