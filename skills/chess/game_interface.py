'''
import os
import time
import math
import json
import torch

import pandas as pd

from shutil import copyfile
'''
import os
import json
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from ai.bot import Agent
from tasks.games.chess.chess import Chess
from skills.chess.game_plumbing import Plumbing

class chess:
    """
    Reinforcement training for AI
    """
    def __init__(self, train = False):
        """
        Input: train - boolean representing if the game is being played in training mode or not
        Description: initalize chess game
        Output: None
        """
        self.train = train

    def legal_moves(self, chess_game):
        """
        Input: chess_game - object containing the current chess game
        Description: returns all legal moves
        Output: numpy array containing all legal moves
        """
        legal = np.zeros((8,8,8,8))
        for cur,moves in chess_game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((cur[0].isupper() and chess_game.p_move == 1) or (cur[0].islower() and chess_game.p_move == -1)):
                cur_pos = chess_game.board_2_array(cur)
                for next in moves:
                    legal[cur_pos[1]][cur_pos[0]][next[1]][next[0]] = 1.
        return legal.flatten()

    def play_game(
        self,
        game_name,
        epoch,
        train = False,
        players = [
            'skills/chess/data/active_param.json',
            'human'
        ]
    ):
        """
        Input: game_name - string representing the game board name
               epoch - integer representing the current epoch
               white - string representing the file name for the white player (Default='model-active.pth.tar') [OPTIONAL]
               black - string representing the file name for the black player (Default='model-new.pth.tar') [OPTIONAL]
               search_amount - integer representing the amount of searches the ai's should perform (Default=50) [OPTIONAL]
               max_depth - integer representing the max depth each search can go (Default=5) [OPTIONAL]
               best_of - integer representing the amount of games played in a bracket (Default=5) [OPTIONAL]
        Description: Plays game for training
        Output: tuple containing game state, training data and which of the players won
        """
        log = []
        human_code = [
            'h',
            'hum',
            'human'
        ]
        end = False
        a_players = []
        plumbing = Plumbing()
        chess_game = deepcopy(Chess()) #'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
        while True:
            for i, p in enumerate(players):
                while True:
                    if str(p).lower() in human_code and len(log) < len(players):
                        a_players.append('human')
                    elif len(log) < len(players):
                        a_players.append(deepcopy(Agent(param_name = p, train = train)))
                    if 'human' in a_players:
                        if chess_game.p_move == 1:
                            print('\nWhites Turn [UPPER CASE]\n')
                        else:
                            print('\nBlacks Turn [LOWER CASE]\n')
                        chess_game.display()
                    if a_players[i] == 'human':
                        cur = input('What piece do you want to move?\n')
                        next = input('Where do you want to move the piece to?\n')
                    else:
                        enc_state = plumbing.encode_state(chess_game)
                        #legal = self.legal_moves(chess_game) #Filter legal moves for inital state
                        probs, v = a_players[i].choose_action(enc_state)
                        print(v)
                        probs = np.array(probs)
                        probs[probs == 0] = 1e-8
                        legal = self.legal_moves(chess_game) #Filter legal moves for inital state
                        #print(legal.sum(), len(legal))
                        probs = np.multiply(legal, probs)
                        probs = probs.tolist()
                        max_prob = max(probs)
                        a_bank = [j for j, v in enumerate(probs) if v == max_prob]
                        b_a = random.choice(a_bank)
                        a_map = np.zeros(4096)
                        a_map[b_a] = 1
                        a_map = a_map.reshape((8,8,8,8))
                        a_index = [(cy, cx, ny, nx) for cy, cx, ny, nx in zip(*np.where(a_map == 1))][0]
                        cur = f'{chess_game.x[a_index[1]]}{chess_game.y[a_index[0]]}'
                        next = f'{chess_game.x[a_index[3]]}{chess_game.y[a_index[2]]}'

                    valid = False
                    if chess_game.move(cur,next) == False:
                        print('Invalid move')
                    else:
                        valid = True
                        #Better game logging
                        cur_pos = chess_game.board_2_array(cur)
                        next_pos = chess_game.board_2_array(next)
                        log.append({
                            **{f'state{i}':float(s) for i, s in enumerate(enc_state[0])},
                            **{f'action{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)}
                        })
                        '''
                        log.append({
                            **{f'state{i}':float(s) for i,s in enumerate(plumbing.encode_state(chess_game)[0])},
                            **{f'prob{x}':p for x, p in enumerate(probs)}
                        })
                        '''
                        print(f'w {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n') if chess_game.p_move > 0 else print(f'b {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')

                        if a_players[i] != 'human':
                            state = chess_game.check_state(chess_game.EPD_hash())
                            if state == '50M' or state == '3F':
                                state = [0,1,0] #Auto tie
                            elif state == 'PP':
                                chess_game.pawn_promotion(n_part='Q') #Auto queen
                            if state != [0,1,0]:
                                state = chess_game.is_end()
                        else:
                            state = chess_game.is_end()
                            if state == [0,0,0]:
                                if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
                                    chess_game.pawn_promotion()
                        if sum(state) > 0:
                            print(f'FINISHED | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} STATE:{state}\n')
                            game_train_data = pd.DataFrame(log)
                            game_train_data = game_train_data.astype(float)
                            end = True
                        break
                if end == True:
                    break
                if valid == True:
                    chess_game.p_move = chess_game.p_move * (-1)
            if end == True:
                break
        del a_players
        return state, game_train_data

    def traing_session(
        self,
        games = 10,
        boards = 1,
        best_of = 5,
        players = [
            {
                'param':'skills/chess/data/active_param.json',
                'train':False
            },
            {
                'param':'skills/chess/data/new_param.json',
                'train':True
            }
        ]
    ):
        #Initalize variables
        GAMES = games #Games to play on each board
        BOARDS = boards #Amount of boards to play on at a time
        BEST_OF = best_of #Amount of games played when evaluating the models
        t_bank = []
        a_players = []
        game_results = {}
        for p in players:
            game_results[p['param']] = 0
            a_players.append(p['param'])
            if p['train'] == True:
                t_bank.append(p['param'])
        game_results['tie'] = 0
        train_data = pd.DataFrame()
        #Begin training games
        for epoch in range(GAMES):
            print(f'STARTING GAMES\n')
            state, train_data = self.play_game(
                'TEST',
                epoch,
                train = True,
                players = a_players
            )
            if state == [1,0,0]:
                print(f'{a_players[0]} WINS')
                game_results[a_players[0]] += 1
            elif state == [0,0,1]:
                print(f'{a_players[-1]} WINS')
                game_results[a_players[-1]] += 1
            else:
                print('TIE GAME')
                game_results['tie'] += 1
            print(game_results)
            '''
            w_m = max(game_results, key=game_results.get)
            if sum([v for v in game_results.values()]) >= BEST_OF and len(t_bank) == 1 and True in [True for p in t_bank if hash(p) == w_m]:
                l_m = [p for p in a_players if hash(p) != wm][0]
                copyfile(
                    os.path.join(folder,w_m),
                    os.path.join(folder,l_m)
                ) #Overwrite active model with new model
            '''
            if sum([v for v in game_results.values()]) >= BEST_OF:
                game_results = {p: 0 for p in a_players}
            if state == [0,0,0]:
                train_data['value'] = [0.] * len(train_data)
            elif state == [1,0,0]:
                train_data['value'] = np.where(train_data['state0'] == 0., 1., -1.)
            else:
                train_data['value'] = np.where(train_data['state0'] == 0., -1., 1.)
            for m in t_bank:
                Agent(param_name = m, train = False).train(train_data)
            train_data = pd.DataFrame()
            a_players.reverse()
