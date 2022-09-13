import os
import json
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from string import ascii_uppercase, digits
from shutil import copyfile, rmtree, copytree
from datetime import datetime

#from ai.bot import Agent
from ai.bot import Agent
from tasks.games.chess.chess import Chess
from skills.chess.game_plumbing import Plumbing
from tools.toolbox import ToolBox

class chess:
    """
    Main interface for the AI to play chess
    """
    def __init__(self):
        self.tools = ToolBox()

    def legal_moves(self, chess_game):
        """
        Input: chess_game - object containing the current chess game
        Description: returns all legal moves
        Output: numpy array containing all legal moves
        """
        legal = np.zeros((8, 8, 8, 8))
        for cur, moves in chess_game.possible_board_moves(capture=True).items():
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
        ],
        tie_min = 100,
        game_num = 0,
        game_max = float('inf')
    ):
        """
        Input: game_name - string representing the name of the match
               epoch - integer representing the current epoch
               train - boolean used as training control (Default = False) [OPTIONAL]
               EPD - string representing the EPD hash to load the board into (Default = None) [OPTIONAl]
               SILENT - boolean used for control of displaying stats or not (Default = True) [OPTIONAL]
               players - list containing the player paramater files (Default = ['skills/chess/data/active_param.json', 'human'] [OPTIONAL]
               tie_min - integer representing the minimum amount of moves for an auto tie game to be possible (Default = 100) [OPTIONAL]
               game_max - integer representing the maximum amount of moves playable before triggering an auto tie (Default = inf) [OPTIONAL]
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
                        a_players.append(deepcopy(Agent(param_name = p, train = train, game_num = game_num)))
                    if 'human' in a_players:
                        if chess_game.p_move == 1:
                            print('\nWhites Turn [UPPER CASE]\n')
                        else:
                            print('\nBlacks Turn [LOWER CASE]\n')
                        chess_game.display()
                    else:
                        plumbing.move_count = a_players[i].move_count
                    enc_state = plumbing.encode_state(chess_game)
                    if a_players[i] == 'human':
                        while True:
                            cur = input('What piece do you want to move?\n').strip()
                            next = input('Where do you want to move the piece to?\n').strip()
                            if len(cur) == 2 and len(next) == 2:
                                break
                            else:
                                print('\nInvalid input, cur and next must each be 2 char long (ex. cur = A2 & next = A4)\n')
                        a_map = np.zeros((8, 8, 8, 8))
                        a_map[chess_game.y.index(cur[1])][chess_game.x.index(cur[0])][chess_game.y.index(next[1])][chess_game.x.index(next[0])] = 1
                        a_map = a_map.flatten()
                        b_a = np.where(a_map == 1)[0][0]
                    else:
                        t1 = datetime.now()
                        legal = self.legal_moves(chess_game) #Filter legal moves for inital state
                        legal[legal == 0] = float('-inf')
                        probs, v, r = a_players[i].choose_action(enc_state, legal_moves = legal)
                        max_prob = max(probs)
                        if SILENT == False:
                            print('\n--------------- STATS ---------------')
                            print(f' Player           | {p}')
                            print(f' Colour           | {"white" if chess_game.p_move > 0 else "black"}')
                            print('- - - - - - - - - - - - - - - - - - -')
                            print(f' Value            | {v}')
                            print(f' Reward           | {r}')
                            print(f' Probability      | {float(max_prob)*100}%')
                            print('-------------------------------------\n')
                        a_bank = [j for j, v in enumerate(probs) if v == max_prob]
                        b_a = random.choice(a_bank)
                        a_map = np.zeros(4096)
                        a_map[b_a] = 1
                        a_map = a_map.reshape((8, 8, 8, 8))
                        a_index = [(cy, cx, ny, nx) for cy, cx, ny, nx in zip(*np.where(a_map == 1))][0]
                        cur = f'{chess_game.x[a_index[1]]}{chess_game.y[a_index[0]]}'
                        next = f'{chess_game.x[a_index[3]]}{chess_game.y[a_index[2]]}'
                    valid = False
                    if chess_game.move(cur, next) == False:
                        if SILENT == False or str(p).lower() in human_code:
                            print('Invalid move')
                    else:
                        valid = True
                        if plumbing.move_count > 1:
                            plumbing.move_que = enc_state.tolist()[0]
                        cur_pos = chess_game.board_2_array(cur)
                        next_pos = chess_game.board_2_array(next)
                        if a_players[i] == 'human':
                            log.append({
                                **{f'state{i}':float(s) for i, s in enumerate(enc_state[0])},
                                **{f'prob{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)},
                                **{'action':b_a},
                                'player':p
                            })
                        else:
                            log.append({
                                **{f'state{i}':float(s) for i, s in enumerate(enc_state[0])},
                                **{f'prob{x}':p for x, p in enumerate(probs)},
                                **{'action':b_a, 'time':(datetime.now() - t1).total_seconds()},
                                'player':p
                            })
                        if SILENT == False or a_players[i] == 'human':
                            if chess_game.p_move > 0:
                                print(f'w {cur.lower()}-->{next.lower()} | GAME:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')
                            else:
                                print(f'b {cur.lower()}-->{next.lower()} | GAME:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')
                        if a_players[i] != 'human':
                            state = chess_game.check_state(chess_game.EPD_hash())
                            t_code = True if state != False else False
                            if ((state == '50M' or state == '3F') and len(log) > tie_min) or len(log) >= game_max:
                                state = [0, 1, 0] #Auto tie
                            elif state == 'PP':
                                chess_game.pawn_promotion(n_part='Q') #Auto queen
                            if state != [0, 1, 0]:
                                state = chess_game.is_end() if len(log) > tie_min else chess_game.is_end(choice=False)
                        else:
                            state = chess_game.is_end()
                            if state == [0, 0, 0]:
                                if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
                                    chess_game.pawn_promotion()
                        if sum(state) > 0:
                            if SILENT == False or a_players[i] == 'human':
                                print(f'FINISHED | GAME:{epoch} BOARD:{game_name} MOVE:{len(log)} STATE:{state}\n')
                            game_train_data = pd.DataFrame(log)
                            float_headers = [h for h in game_train_data if h != 'player']
                            game_train_data[float_headers] = game_train_data[float_headers].astype(float)
                            end = True
                        break
                if end == True:
                    break
                if valid == True:
                    chess_game.p_move = chess_game.p_move * (-1)
            if end == True:
                break
        for i, p in enumerate(players):
            if a_players[i] != 'human' and a_players[i].E_DB is not None:
                new_tokens = a_players[i].tools.convert_token_2_embedding(
                    game_train_data[[h for h in game_train_data if 'state' in h]].drop_duplicates(),
                    a_players[i].m_weights['representation']['model'],
                    a_players[i].m_weights['backbone']['model']
                )
                #new_tokens['state'] = new_tokens['state'].apply(lambda x: x if isinstance(x, list) else x.tolist())
                #new_tokens['encoding'] = new_tokens['encoding'].apply(lambda x: x if isinstance(x, list) else x.tolist())
                a_players[i].E_DB = a_players[i].E_DB.append(new_tokens, ignore_index=True)
                a_players[i].E_DB['state'] = a_players[i].E_DB['state'].apply(lambda x: x if isinstance(x, list) else x.tolist())
                a_players[i].E_DB['encoding'] = a_players[i].E_DB['encoding'].apply(lambda x: x if isinstance(x, list) else x.tolist())
                a_players[i].E_DB.to_csv(f"{'/'.join(p.split('/')[:-1]).replace('(temp)', '')}/logs/encoded_data.csv", index=False)
        del a_players
        if t_code == True: print(f'Game Code Found = {state}\n')
        return state, game_train_data

    def traing_session(
        self,
        loops = 15,
        games = 10,
        boards = 1,
        best_of = 5,
        EPD = None,
        SILENT = True,
        player = 'skills/chess/data/models/test',
        tie_min = 100,
        game_max = float('inf'),
        full_model = False
    ):
        """
        Input: games - integer representing the amount of games to train on (Default = 10) [OPTIONAL]
               boards - integer representing the amount of boards to play at once (Default = 1) [OPTIONAL]
               best_of = integer representing the amount of games to use in a round-robin (Default = 5) [OPTIONAL]
               EPD - string representing the EPD hash to load the board into (Default = None) [OPTIONAl]
               SILENT - boolean used for control of displaying stats or not (Default = True) [OPTIONAL]
               players - list of player parameters (Default = [{'param':'skills/chess/data/new_param.json', 'train':True}] [OPTIONAL]
               tie_min - integer representing the minimum amount of moves for an auto tie game to be possible (Default = 100) [OPTIONAL]
               game_max - integer representing the maximum amount of moves playable before triggering an auto tie (Default = inf) [OPTIONAL]
               full_model - boolean representing if the full model is being trained every exploration game or not (Default = False) [OPTIONAL]
        Description: train ai by playing multiple games of chess
        Output: None
        """
        #Initalize variables
        LOOPS = loops
        T_GAMES = games #Games of self play
        BOARDS = boards #Amount of boards to play on at a time
        BEST_OF = best_of #Amount of games played when evaluating the models
        if os.path.exists(f'{player}/logs/training_log.csv'):
            t_log = pd.read_csv(f'{player}/logs/training_log.csv')
        else:
            t_log = pd.DataFrame()
        n_player = player + '(temp)'
        if os.path.exists(n_player) == False:
            print('CREATE NEW')
            os.makedirs(n_player) #Create folder
            copyfile(f'{player}/parameters.json', f'{n_player}/parameters.json') #Overwrite active model with new model
            if os.path.exists(f'{n_player}/parameters.json'):
                if os.path.exists(f'{player}/weights') and len(os.listdir(f"{player}/weights")) > 0:
                    copytree(
                        f'{player}/weights',
                        f'{n_player}/weights'
                    )
                else:
                    os.makedirs(f'{n_player}/weights')
        train_data = pd.DataFrame()
        if os.path.exists(f'{player}/logs/game_log.csv'):
            g_log = pd.read_csv(f'{player}/logs/game_log.csv').drop_duplicates(subset=['Game-ID'])
            g_num = len(g_log)
            del g_log
        else:
            g_num = 0
        #Begin training games
        for _ in range(LOOPS):
            for t, g_count in enumerate([T_GAMES, BEST_OF]):
                if t == 0:
                    a_players = [f'{n_player}/parameters.json'] #Self-play
                    game_results = {'white':0, 'black':0, 'tie':0}
                    pass
                else:
                    a_players = [f'{player}/parameters.json', f'{n_player}/parameters.json'] #Evaluate
                    game_results = {f'{player}/parameters.json':0, f'{n_player}/parameters.json':0, 'tie':0}
                    pass
                for g in range(g_count):
                    #PLAY GAME
                    print(f'\nSTARTING GAMES\n')
                    state, train_data = self.play_game(
                        'TEST',
                        g,
                        train = True if t == 0 else False,
                        EPD = EPD,
                        SILENT = SILENT,
                        players = a_players,
                        tie_min = tie_min,
                        game_max = game_max,
                        game_num = g_num
                    )
                    if t == 0:
                        if state == [1, 0, 0]:
                            print(f'WHITE WINS')
                            game_results['white'] += 1
                            train_data['value'] = np.where(train_data['state0'] == 0., 1., -1.)
                        elif state == [0, 0, 1]:
                            print(f'BLACK WINS')
                            game_results['black'] += 1
                            train_data['value'] = np.where(train_data['state0'] == 0., -1., 1.)
                        else:
                            print('TIE GAME')
                            game_results['tie'] += 1
                            train_data['value'] = [0.] * len(train_data)
                        b_elo = ''
                    else:
                        print(game_results)
                        if state == [1, 0, 0]:
                            print(f'{a_players[0]} WINS')
                            game_results[a_players[0]] += 1
                            train_data['value'] = np.where(train_data['state0'] == 0., 1., -1.)
                            b_elo = 0 if n_player == a_players[0] else 1
                        elif state == [0, 0, 1]:
                            print(f'{a_players[-1]} WINS')
                            game_results[a_players[-1]] += 1
                            train_data['value'] = np.where(train_data['state0'] == 0., -1., 1.)
                            b_elo = 0 if n_player == a_players[-1] else 1
                        else:
                            print('TIE GAME')
                            game_results['tie'] += 1
                            train_data['value'] = [0.] * len(train_data)
                            b_elo = 0
                    print(game_results)
                    print()
                    #UPDATE TRAINING STATE
                    if t == 0 and sum([v for v in game_results.values()]) >= g_count:
                        game_results = {f'{player}/parameters.json':0, f'{n_player}/parameters.json':0, 'tie':0}
                    elif t == 0:
                        pass
                    elif sum([v for v in game_results.values()]) >= g_count:
                        m_wins = max(game_results.values())
                        winners = [p for p in game_results if game_results[p] == m_wins]
                        print(winners)
                        if os.path.exists(f'{player}/weights') == False:
                            os.makedirs(f'{player}/weights') #Create folder
                        if len(os.listdir(f'{player}/weights')) == 0 or len(winners) == 1 and winners[0] == n_player:
                            print('OVERWRITE')
                            self.tools.overwrite_model(
                                n_player,
                                player
                            )
                        game_results = {'white':0, 'black':0, 'tie':0}
                    else:
                        a_players.reverse()
                    #LOG TRAINING DATA
                    train_data['reward'] = [0.] * len(train_data)

                    if os.path.exists(f'{player}/logs/game_log.csv'):
                        g_log = pd.read_csv(f'{player}/logs/game_log.csv')
                    else:
                        g_log = pd.DataFrame()
                    if t == 0:
                        train_data['ELO'] = [''] * len(train_data)
                    else:
                        cur_ELO = g_log['ELO'].dropna().iloc[-1] if 'ELO' in g_log and len(g_log['ELO'].dropna()) > 0 else 0
                        ELO = self.tools.update_ELO(
                            cur_ELO, #ELO_p1,
                            cur_ELO,  #ELO_p2
                            tie = True if state == [0, 0, 0] else False
                        )
                        train_data['ELO'] = [ELO[b_elo]] * len(train_data)
                        #print(cur_ELO, ELO, b_elo)
                    train_data['Game-ID'] = ''.join(random.choices(ascii_uppercase + digits, k=random.randint(15, 15)))
                    train_data['Date'] = [datetime.now()] * len(train_data)
                    g_log = g_log.append(train_data, ignore_index=True)
                    if os.path.exists(f'{player}/logs') == False:
                        os.makedirs(f'{player}/logs') #Create folder
                    g_log.to_csv(f'{player}/logs/game_log.csv', index=False)
                    if t == 0:
                        if full_model == True:
                            s_headers = [h for h in g_log if 'state' in h]
                            g_log = g_log.drop_duplicates(subset=s_headers, keep='last')
                            #m_log = pd.DataFrame(Agent(param_name = f'{n_player}/parameters.json', train = False).train(g_log[g_log['value']!=0.0].drop_duplicates(subset=s_headers, keep='last'), folder=n_player))
                            m_log = pd.DataFrame(Agent(param_name = f'{n_player}/parameters.json', train = False, game_num = g_num).train(g_log, folder=n_player))
                            del s_headers
                            del g_log
                        else:
                            if g == g_count - 1:
                                s_headers = [h for h in g_log if 'state' in h]
                                g_log = g_log.drop_duplicates(subset=s_headers, keep='last')
                                m_log = pd.DataFrame(Agent(param_name = f'{n_player}/parameters.json', train = False, game_num = g_num).train(g_log, folder=n_player))
                                del s_headers
                                del g_log
                            else:
                                del g_log
                                m_log = pd.DataFrame(Agent(param_name = f'{n_player}/parameters.json', train = False, game_num = g_num).train(train_data, folder=n_player, encoder=False))
                        if os.path.exists(f'{player}/logs/training_log.csv'):
                            t_log = pd.read_csv(f'{player}/logs/training_log.csv')
                        else:
                            t_log = pd.DataFrame()
                        m_log['model'] = player
                        t_log = t_log.append(m_log, ignore_index=True)
                        del m_log
                        t_log.to_csv(f'{player}/logs/training_log.csv', index=False)
                        del t_log
                    #GARBEGE CLEAN UP
                    train_data = pd.DataFrame()
                    if os.path.exists(f'{player}/weights') == False:
                        os.makedirs(f'{player}/weights') #Create folder
                    if len(os.listdir(f'{player}/weights')) == 0:
                        self.tools.overwrite_model(
                            f'{n_player}/weights',
                            f'{player}/weights'
                        ) #Move model data over if none exists
                    g_num += 1

    def replay_game(
        game_id,
        game_log,
        id_header = 'Game-ID',
        state_header = 'state',
        action_header = 'action'
    ):
        """
        Input: game_id - string representing the ID of the game you wish to watch the replay of
               game_log - dataframe containing the compleate game log
               id_header - string representing the header to find the game id in the game log (Default = 'ID') [OPTIONAL]
        Description: watch a replay of a previously played chess game
        Output: None
        """
        game_data = game_log[game_log[id_header] == game_id]
        if len(game_data) > 0:
            chess_game = Chess()
            plumbing = Plumbing()
            s_headers = [h for h in game_data if state_header in h]
            cur_move = 0
            for i, row in game_data.iterrows():
                g_state = row[s_headers].to_numpy()[1:].reshape((8,8)).tolist()
                chess_game.p_move = 1 if row[f'{state_header}0'] == 0 else -1
                chess_game.board = plumbing.decode_state(g_state)
                cur, next = plumbing.parse_action(int(row[action_header]))
                if chess_game.move(cur, next):
                    chess_game.display()
                    if chess_game.p_move > 0:
                        print(f'w {cur.lower()}-->{next.lower()} | GAME:{game_id} MOVE:{cur_move} HASH:{chess_game.EPD_hash()}\n')
                    else:
                        print(f'b {cur.lower()}-->{next.lower()} | GAME:{game_id} MOVE:{cur_move} HASH:{chess_game.EPD_hash()}\n')
                    state = chess_game.check_state(chess_game.EPD_hash())
                    if state == '50M' or state == '3F' or state == 'PP':
                        print(f'Game Code Found = {state}\n')
                    cur_move += 1
                else:
                    print('INVALID MOVE ENCOUNTERED\n')
                    break
            print(chess_game.is_end())
        else:
            print(f'Error - could not find game data with the game ID {game_id}')
