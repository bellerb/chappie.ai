import os
import time
import json
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm

from ai.bot import Agent
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
        'Chess                 (c)'
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
            'Play                  (p)',
            'Train                 (t)',
            'Evaluate              (e)'
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
            col_choice = [
                'White                 (w)',
                'Black                 (b)'
            ]
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
            g_log = pd.concat([g_log, log], ignore_index=True)
            g_log.to_csv(f'{player}/logs/game_log(human).csv', index=False)
        #TRAIN CHESS ---------------------------------------
        elif task == 1:
            print(
'''
-------------------------------------------------
What kind of training would you like to do?
-------------------------------------------------
'''
            )
            t_methods = [
                'Supervised            (s)',
                'Offline Reinforcement (r)'
            ]
            t_choice = tools.give_options(t_methods)
            if t_choice == 0:
                print(
'''
-------------------------------------------------
Which model would you like to train?
-------------------------------------------------
'''
                )
                i = 0
                model_list = []
                for m in os.listdir('skills/chess/data/models'):
                    if m != '.DS_Store' and '(temp)' not in m and os.path.exists(f'skills/chess/data/models/{m}/logs/game_log.csv'):
                        model_list.append(f'{m} ({i})')
                        i += 1
                m_choice = tools.give_options(model_list)
                print()
                player = model_list[m_choice].split("(")[0].strip()
                with tqdm(total=10, desc=f'Loading model {player}') as pbar:
                    pbar.update(5)
                    pbar.refresh()
                    agent = Agent(
                        param_name = f'skills/chess/data/models/{player}/parameters.json',
                        train = False
                    )
                    pbar.update(5)
                print(
'''
-------------------------------------------------
Which layer do you want to train?
-------------------------------------------------
'''
                )
                layer_list = [
                    'Representation        (h)',
                    'Backbone              (b)'
                ]
                if agent.E_DB is not None:
                    layer_list.append('Cca                   (c)')
                layer_list += [
                    'Value                 (v)',
                    'Policy                (p)',
                    'State                 (s)',
                    'Reward                (r)'
                ]
                l_choice = tools.give_options(layer_list)
                print()
                layer_name = layer_list[l_choice].split("(")[0].strip().lower()
                with tqdm(total=len(layer_list) + 2, desc=f'Loading {layer_name} layer') as pbar:
                    if 'Representation' in layer_list[l_choice] or 'Backbone' in layer_list[l_choice] or 'Cca' in layer_list[l_choice]:
                        agent.mse = torch.nn.MSELoss() #Mean squared error loss
                        agent.bce = torch.nn.BCELoss() #Binary cross entropy loss
                    elif 'Policy' not in layer_list[l_choice]:
                        agent.mse = torch.nn.MSELoss() #Mean squared error loss
                    else:
                        agent.bce = torch.nn.BCELoss() #Binary cross entropy loss
                    pbar.update(1)
                    pbar.refresh()
                    #Load model weights
                    for l in layer_list:
                        longform_layer, shortform_layer = l.split('(')
                        longform_layer = longform_layer.strip().lower()
                        shortform_layer = shortform_layer.replace(')', '').strip()
                        agent.init_model_4_training(
                            f'{shortform_layer}_optimizer',
                            f'{shortform_layer}_scheduler',
                            longform_layer,
                            f'{shortform_layer}_step',
                            f'{shortform_layer}_gamma'
                        )
                        pbar.update(1)
                        pbar.refresh()
                    start_time = time.time() #Get time of starting process
                    with open(f'skills/chess/data/models/{player}/logs/game_log.csv') as f:
                        data_header = f.readline().split(',')
                        data_size = sum(1 for _ in f)
                    pbar.update(1)
                    pbar.refresh()
                print(
f'''
-------------------------------------------------
Would you like to use train on all the data?
Original data size: {data_size}
-------------------------------------------------
'''
                )
                training_size_list = [
                    'Yes                   (y)',
                    'No                    (n)',
                ]
                s_choice = tools.give_options(training_size_list)
                if (s_choice == 0):
                    data = pd.read_csv(f'skills/chess/data/models/{player}/logs/game_log.csv')
                elif (s_choice == 1): 
                    data_reduction = input(
'''
-------------------------------------------------
How much training data do you wish to use?
-------------------------------------------------\n
'''
                    )
                    while True:
                        try:
                            data_reduction = int(data_reduction)
                            break
                        except:
                            data_reduction = input(
'''
-------------------------------------------------
Sorry invalid input [integer]
-------------------------------------------------\n
'''
                            )
                    data = tools.read_n_from_bottom_csv(
                        f'skills/chess/data/models/{player}/logs/game_log.csv', 
                        data_reduction, 
                        headers = data_header
                    )
                    print(f'\nTraining data size reduced to: {len(data)}\n')
                EPOCHS = input(
'''
-------------------------------------------------
How many EPOCHs do you wish to run?
-------------------------------------------------\n
'''
                )
                while True:
                    try:
                        EPOCHS = int(EPOCHS)
                        break
                    except:
                        EPOCHS = input(
'''
-------------------------------------------------
Sorry invalid input [integer]
-------------------------------------------------\n
'''
                        )
                BSZ = input(
'''
-------------------------------------------------
What batch size do you wish to use?
-------------------------------------------------\n
'''
                )
                while True:
                    try:
                        BSZ = int(BSZ)
                        break
                    except:
                        BSZ = input(
'''
-------------------------------------------------
Sorry invalid input [integer]
-------------------------------------------------\n
'''
                        )
                print(
f'''
-------------------------------------------------
Beginning Training
=================================================
method:           {t_methods[t_choice].split(" ")[0].strip().lower()}
model:            {player}
layer:            {layer_name}
sample size:      {len(data)}
epochs:           {EPOCHS}
batch size:       {BSZ}
-------------------------------------------------
'''
                )
                log = agent.train_layer(
                    layer_name,
                    {'epoch':EPOCHS, 'bsz':BSZ},
                    data
                )
                del data
                if os.path.exists(f'skills/chess/data/models/{player}/weights') is False:
                    os.makedirs(f'skills/chess/data/models/{player}/weights') #Create folder
                torch.save({
                    'state_dict': agent.m_weights[layer_name]['model'].state_dict(),
                }, f"skills/chess/data/models/{player}/weights/{agent.m_weights[layer_name]['param']}")
                if os.path.exists(f'{player}/logs/training_log.csv'):
                    t_log = pd.read_csv(f'{player}/logs/training_log.csv')
                else:
                    t_log = pd.DataFrame()
                t_log = pd.concat([t_log, pd.DataFrame(log)], ignore_index=True)
                t_log.to_csv(f'skills/chess/data/models/{player}/logs/training_log.csv', index=False)
            elif t_choice == 1:
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
                    games = 50,
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
                            '''
                            t_results = t_results.append({
                                'white':players[0],
                                'black':players[1],
                                'state':state
                            },ignore_index=True)
                            '''
                            t_results = pd.concat([t_results, {
                                'white':players[0],
                                'black':players[1],
                                'state':state
                            }], ignore_index=True)
                print(t_results)
                t_results.to_csv('tournament_results.csv', index=False)
