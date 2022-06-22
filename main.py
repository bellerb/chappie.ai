import os
import json
import pandas as pd
import torch

from ai.model import Representation, Backbone
from tools.toolbox import ToolBox
from skills.chess.game_interface import chess

if __name__ == '__main__':
    '''
    game_log = pd.read_csv(f'skills/chess/data/models/test_v7/logs/game_log.csv')
    game_id = game_log.drop_duplicates(subset=['Game-ID'],keep='last')['Game-ID'].iloc[-2]
    chess.replay_game(
        game_id,
        game_log
    )
    quit()
    '''
    tools = ToolBox()
    print(
'''
-------------------------------------------------
   Hi I'm Chappie, what would you like to do?
-------------------------------------------------
'''
    )
    o_bank = [
        'Chess (c)'
    ]
    task = tools.give_options(o_bank)
    if task == 0:
        print(
'''
-------------------------------------------------
 You've selected chess.
 What would you like to do in chess?
-------------------------------------------------
'''
        )
        c_tasks = [
            'Play     (p)',
            'Train    (t)',
            'Evaluate (e)'
        ]
        task = tools.give_options(c_tasks)
        i = 0
        model_list = []
        for m in os.listdir(f'skills/chess/data/models'):
            if m != '.DS_Store' and '(temp)' not in m:
                model_list.append(f'{m} ({i})')
                i += 1
        #PLAY CHESS ---------------------------------------
        if task == 0:
            print(
'''
-------------------------------------------------
 Playig chess.
 Which colour would you like to be?
-------------------------------------------------
'''
            )
            players = ['human']
            col_choice = ['White (w)','Black (b)']
            p_col = tools.give_options(col_choice)
            print(
'''
-------------------------------------------------
Please select a bot to play?
-------------------------------------------------
'''
            )
            m_choice = tools.give_options(model_list)
            player = f'skills/chess/data/models/{model_list[m_choice].split("(")[0].strip()}'
            if p_col == 0:
                players.append(f'{player}/parameters.json')
            else:
                players.insert(0, f'{player}/parameters.json')
            if os.path.exists(f'{player}/logs/game_log(human).csv'):
                game_num = len(pd.read_csv(f'{player}/logs/game_log(human).csv'))
            else:
                game_num = 0
            chess = chess()
            print(
'''
-------------------------------------------------
Starting game...
-------------------------------------------------
'''
            )
            state, log = chess.play_game(
                'TEST',
                0,
                #EPD = '1b4k1/Q7/p2np1/P1P2p2/1P3P2/1R5R/q6P/5rK1 b - -',
                players = players,
                SILENT = False,
                train = False,
                game_num = game_num
            )
            print(state)
            if os.path.exists(f'{player}/logs') == False:
                os.makedirs(f'{player}/logs') #Create folder
            if os.path.exists(f'{player}/logs/game_log(human).csv'):
                g_log = pd.read_csv(f'{player}/logs/game_log(human).csv')
            else:
                g_log = pd.DataFrame()
            g_log = g_log.append(log, ignore_index=True)
            g_log.to_csv(f'{player}/logs/game_log(human).csv', index=False)
        #TRAIN CHESS ---------------------------------------
        elif task == 1:
            print(
'''
-------------------------------------------------
Training chess bots.
Please select which bot you would like to train?
-------------------------------------------------
'''
            )
            m_choice = tools.give_options(model_list)
            player = f'skills/chess/data/models/{model_list[m_choice].split("(")[0].strip()}'
            chess = chess()
            chess.traing_session(
                loops = 1,
                games = 5,
                boards = 1,
                best_of = 3,
                player = player,
                SILENT = False,
                tie_min = float('inf'),
                full_model = False,
                #game_max = 200
            )
        #EVAL CHESS ---------------------------------------
        elif task == 2:
            print(
'''
-------------------------------------------------
Evaluating chess bots.
What kind of evaluation would you like to do?
-------------------------------------------------
'''
            )
            e_type = tools.give_options(['Single Game       (s)', 'Tourmanent Style  (t)'])
            if e_type == 0:
                players = []
                print(
'''
-------------------------------------------------
Single game selected.
Please select a bot to play white?
-------------------------------------------------
'''
                )
                m_choice = tools.give_options(model_list)
                players.append(f'skills/chess/data/models/{model_list[m_choice].split("(")[0].strip()}/parameters.json')
                print(
'''
-------------------------------------------------
Please select a bot to play black?
-------------------------------------------------
'''
                )
                m_choice = tools.give_options(model_list)
                players.append(f'skills/chess/data/models/{model_list[m_choice].split("(")[0].strip()}/parameters.json')
                chess = chess()
                print(
'''
-------------------------------------------------
Starting game...
-------------------------------------------------
'''
                )
                state, log = chess.play_game(
                    'TEST',
                    0,
                    #EPD = '1b4k1/Q7/p2np1/P1P2p2/1P3P2/1R5R/q6P/5rK1 b - -',
                    players = players,
                    SILENT = False,
                    tie_min = float('inf'),
                    game_num = 674,
                    #game_max = 200
                )
                print(state)
            elif e_type == 1:
                print(
'''
-------------------------------------------------
Starting games...
-------------------------------------------------
'''
                )
                t_results = pd.DataFrame()
                for i in range(len(model_list) - 1):
                    for j in range(i + 1, len(model_list)):
                        for x in range(10):
                            chess_game = chess()
                            if x % 2 == 0:
                                players = [
                                    f'skills/chess/data/models/{model_list[i].split("(")[0].strip()}/parameters.json',
                                    f'skills/chess/data/models/{model_list[j].split("(")[0].strip()}/parameters.json'
                                ]
                            else:
                                players = [
                                    f'skills/chess/data/models/{model_list[j].split("(")[0].strip()}/parameters.json',
                                    f'skills/chess/data/models/{model_list[i].split("(")[0].strip()}/parameters.json'
                                ]
                            state, log = chess_game.play_game(
                                'TEST',
                                0,
                                #EPD = '1b4k1/Q7/p2np1/P1P2p2/1P3P2/1R5R/q6P/5rK1 b - -',
                                players = players,
                                SILENT = True,
                                tie_min = float('inf'),
                                train = False,
                                game_num = 674,
                                #game_max = 200
                            )
                            t_results = t_results.append({
                                'white':players[0],
                                'black':players[1],
                                'state':state
                            },ignore_index=True)
                print(t_results)
                t_results.to_csv('tournament_results.csv', index=False)
