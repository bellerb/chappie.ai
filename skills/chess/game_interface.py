import os
import json
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from shutil import copyfile, rmtree, copytree
from datetime import datetime

from ai.bot import Agent
from tasks.games.chess.chess import Chess
from skills.chess.game_plumbing import Plumbing

class chess:
    """
    Main interface for the AI to play chess
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
        EPD = None,
        SILENT = True,
        players = [
            'skills/chess/data/active_param.json',
            'human'
        ]
    ):
        """
        Input: game_name - string representing the name of the match
               epoch - integer representing the current epoch
               train - boolean used as training control (default = False) [OPTIONAL]
               EPD - string representing the EPD hash to load the board into (default = None) [OPTIONAl]
               SILENT - boolean used for control of displaying stats or not (default = True) [OPTIONAL]
               players - list containing the player paramater files (default = ['skills/chess/data/active_param.json','human']
        Description: play a game of chess
        Output: tuple containing the game outcome and a dataframe containing the game log
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
        if EPD is None:
            chess_game = deepcopy(Chess()) #'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
        else:
            chess_game = deepcopy(Chess(EPD=EPD)) #'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
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
                    enc_state = plumbing.encode_state(chess_game)
                    if a_players[i] == 'human':
                        cur = input('What piece do you want to move?\n')
                        next = input('Where do you want to move the piece to?\n')
                    else:
                        legal = self.legal_moves(chess_game) #Filter legal moves for inital state
                        legal[legal == 0] = float('-inf')
                        probs, v = a_players[i].choose_action(enc_state, legal_moves = legal)
                        max_prob = max(probs)
                        if SILENT == False:
                            print('\n--------------- STATS ---------------')
                            print(f' Value            | {v}')
                            print(f' Move Probability | {max_prob}')
                            print('-------------------------------------\n')
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
                        if SILENT == False or str(p).lower() in human_code:
                            print('Invalid move')
                    else:
                        valid = True
                        cur_pos = chess_game.board_2_array(cur)
                        next_pos = chess_game.board_2_array(next)
                        if a_players[i] == 'human':
                            log.append({
                                **{f'state{i}':float(s) for i,s in enumerate(plumbing.encode_state(chess_game)[0])},
                                **{f'action{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)}
                            })
                        else:
                            log.append({
                                **{f'state{i}':float(s) for i,s in enumerate(plumbing.encode_state(chess_game)[0])},
                                **{f'prob{x}':p for x, p in enumerate(probs)}
                            })
                        if SILENT == False or a_players[i] == 'human':
                            if chess_game.p_move > 0:
                                print(f'w {cur.lower()}-->{next.lower()} | GAME:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')
                            else:
                                print(f'b {cur.lower()}-->{next.lower()} | GAME:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')
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
                            if SILENT == False or a_players[i] == 'human':
                                print(f'FINISHED | GAME:{epoch} BOARD:{game_name} MOVE:{len(log)} STATE:{state}\n')
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
        loops = 15,
        games = 10,
        boards = 1,
        best_of = 5,
        EPD = None,
        SILENT = True,
        player = 'skills/chess/data/new_param.json'
    ):
        """
        Input: games - integer representing the amount of games to train on (default = 10) [OPTIONAL]
               boards - integer representing the amount of boards to play at once (default = 1) [OPTIONAL]
               best_of = integer representing the amount of games to use in a round-robin (default = 5) [OPTIONAL]
               EPD - string representing the EPD hash to load the board into (default = None) [OPTIONAl]
               SILENT - boolean used for control of displaying stats or not (default = True) [OPTIONAL]
               players - list of player parameters (default = [{'param':'skills/chess/data/new_param.json','train':True}] [OPTIONAL]
        Description: train ai by playing multiple games of chess
        Output: None
        """
        #Initalize variables
        LOOPS = loops
        T_GAMES = games #Games of self play
        BOARDS = boards #Amount of boards to play on at a time
        BEST_OF = best_of #Amount of games played when evaluating the models
        #t_bank = []
        '''
        a_players = []
        for p in players:
            a_players.append(p['param'])
            if p['train'] == True:
                t_bank.append(p['param'])
        '''
        if os.path.exists('skills/chess/data/training_log.csv'):
            t_log = pd.read_csv('skills/chess/data/training_log.csv')
        else:
            t_log = pd.DataFrame()
        if os.path.exists(f"{'/'.join(player.split('/')[:-1])}/models/{player.split('/')[-1].split('.')[0]}") == False:
            os.makedirs(f"{'/'.join(player.split('/')[:-1])}/models/{player.split('/')[-1].split('.')[0]}") #Create folder
        n_player = '.'.join(s + '(temp)' if x == 0 else s for x, s in enumerate(player.split('.')))
        if os.path.exists(n_player) == False:
            print('CREAT NEW')
            copyfile(player, n_player) #Overwrite active model with new model
            if os.path.exists(n_player):
                with open(n_player) as f:
                    m_param = json.load(f)
                    for m in m_param['data']['active-models']:
                        s_hold = m_param['data']['active-models'][m].split('/')
                        m_param['data']['active-models'][m] = '/'.join(s for x, s in enumerate(s_hold[:-2]))
                        m_param['data']['active-models'][m] += f"/{n_player.split('/')[-1].split('.')[0]}/{s_hold[-1]}"
                with open(n_player, 'w', encoding='utf-8') as f:
                    json.dump(m_param, f, ensure_ascii=False, indent=4)
                    del m_param
                if len(os.listdir(f"{'/'.join(player.split('/')[:-1])}/models/{player.split('/')[-1].split('.')[0]}")) > 0:
                    copytree(
                        f"{'/'.join(player.split('/')[:-1])}/models/{player.split('/')[-1].split('.')[0]}",
                        f"{'/'.join(n_player.split('/')[:-1])}/models/{n_player.split('/')[-1].split('.')[0]}"
                    )
                else:
                    os.makedirs(f"{'/'.join(n_player.split('/')[:-1])}/models/{n_player.split('/')[-1].split('.')[0]}")
        game_results = {'white':0, 'black':0, 'tie':0}
        train_data = pd.DataFrame()
        #Begin training games
        for _ in range(LOOPS):
            for t, g_count in enumerate([T_GAMES, BEST_OF]):
                if t == 0:
                    a_players = [n_player] #Self-play
                    pass
                else:
                    a_players = [player, n_player] #Evaluate
                    pass
                for g in range(g_count):
                    #PLAY GAME
                    print(f'STARTING GAMES\n')
                    state, train_data = self.play_game(
                        'TEST',
                        g,
                        train = True if t == 0 else False,
                        EPD = EPD,
                        SILENT = SILENT,
                        players = a_players
                    )
                    if t == 0:
                        if state == [1,0,0]:
                            print(f'WHITE WINS')
                            game_results['white'] += 1
                        elif state == [0,0,1]:
                            print(f'BLACK WINS')
                            game_results['black'] += 1
                        else:
                            print('TIE GAME')
                            game_results['tie'] += 1
                    else:
                        print(game_results)
                        if state == [1,0,0]:
                            print(f'{a_players[0]} WINS')
                            game_results[a_players[0]] += 1
                        elif state == [0,0,1]:
                            print(f'{a_players[-1]} BLACK WINS')
                            game_results[a_players[-1]] += 1
                        else:
                            print('TIE GAME')
                            game_results['tie'] += 1
                    print(game_results)
                    print()
                    #UPDATE TRAINING STATE
                    if t == 0 and sum([v for v in game_results.values()]) >= g_count:
                        game_results = {player:0, n_player:0, 'tie':0}
                    elif t == 0:
                        pass
                    elif sum([v for v in game_results.values()]) >= g_count:
                        m_wins = max(game_results.values())
                        winners = [p for p in game_results if game_results[p] == m_wins]
                        if len(os.listdir(f"{'/'.join(player.split('/')[:-1])}/models/{player.split('/')[-1].split('.')[0]}")) == 0 or len(winners) == 1 and winners[0] == n_player:
                            Plumbing.overwrite_model(
                                n_player,
                                player
                            )
                        game_results = {'white':0, 'black':0, 'tie':0}
                    else:
                        a_players.reverse()
                    #LOG TRAINING DATA
                    if state == [0,0,0]:
                        train_data['value'] = [0.] * len(train_data)
                    elif state == [1,0,0]:
                        train_data['value'] = np.where(train_data['state0'] == 0., 1., -1.)
                    else:
                        train_data['value'] = np.where(train_data['state0'] == 0., -1., 1.)
                    train_data['reward'] = [0.] * len(train_data)
                    m_log = pd.DataFrame(Agent(param_name = n_player, train = False).train(train_data))
                    m_log['model'] = n_player
                    t_log = t_log.append(m_log,ignore_index=True)
                    del m_log
                    t_log.to_csv('skills/chess/data/training_log.csv', index=False)
                    if os.path.exists('skills/chess/data/game_log.csv'):
                        g_log = pd.read_csv('skills/chess/data/game_log.csv')
                    else:
                        g_log = pd.DataFrame()
                    train_data['Date'] = [datetime.now()] * len(train_data)
                    g_log = g_log.append(train_data, ignore_index=True)
                    g_log.to_csv('skills/chess/data/game_log.csv', index=False)
                    #GARBEGE CLEAN UP
                    del g_log
                    train_data = pd.DataFrame()
                    if len(os.listdir(f"{'/'.join(player.split('/')[:-1])}/models/{player.split('/')[-1].split('.')[0]}")) == 0:
                        Plumbing.overwrite_model(
                            n_player,
                            player
                        ) #Move model data over if none exists
        rmtree(f"{'/'.join(n_player.split('/')[:-1])}/models/{n_player.split('/')[-1].split('.')[0]}") #Remove temporary model folder
        os.remove(n_player) #Remove temporary param
