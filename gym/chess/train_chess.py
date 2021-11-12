import os
import time
import math
import json
import torch
import random
import pandas as pd
from copy import deepcopy
from shutil import copyfile

from ai.bot import Agent
from tasks.chess.chess import Chess
from gym.chess.game_plumbing import Plumbing

class train:
    """
    Reinforcement training for AI
    """
    def play_game(
        game_name,
        epoch,
        train = False,
        players = [
            'model-new.pth.tar',
            'param':'model-active.pth.tar'
        ]
    ):
        """
        Input: game_name - string representing the game board name
               epoch - integer representing the current epoch
               train - boolean representing the training control (Default=False) [OPTIONAL]
               white - string representing the file name for the white player (Default='model-active.pth.tar') [OPTIONAL]
               black - string representing the file name for the black player (Default='model-new.pth.tar') [OPTIONAL]
               search_amount - integer representing the amount of searches the ai's should perform (Default=50) [OPTIONAL]
               max_depth - integer representing the max depth each search can go (Default=5) [OPTIONAL]
               best_of - integer representing the amount of games played in a bracket (Default=5) [OPTIONAL]
        Description: Plays game for training
        Output: tuple containing game state, training data and which of the players won
        """
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
                m_check = False
                while True:
                    if True in [True for s in human_code if str(p).lower() == s] and len(log) <= 1:
                        a_players[i] = 'human'
                    elif len(log) <= 1:
                        a_players[i] = deepcopy(Agent(param_name = p, train = train)))
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
                        if m_check == False:
                            enc_state = Plumbing.encode_state(chess_game)
                            u_bank = a_players[i].choose_action(enc_state)
                        if train == True:
                            pass #Add valid moves parser for training
                        a_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
                        b_a = random.choice(a_bank)
                        a_map = np.zeros(4096)
                        a_map[b_a] = 1
                        a_map = a_map.reshape((8,8,8,8))
                        a_index = [(cy, cx, ny, nx) for cy, cx, ny, nx in zip(*np.where(a_map == 1))]
                        cur = f'{game.x[a_index[1]]}{game.y[a_index[0]]}'
                        next = f'{game.x[a_index[3]]}{game.y[a_index[2]]}'

                    valid = False
                    if chess_game.move(cur,next) == False:
                        print('Invalid move')
                        a[b_a] = float('-inf')
                    else:
                        valid = True
                        cur_pos = chess_game.board_2_array(cur)
                        next_pos = chess_game.board_2_array(next)
                        #Better game logging
                        log.append({**{f'state{i}':float(s) for i,s in enumerate(plumbing.encode_state(chess_game)[0])},
                                    **{f'action{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)}})
                        print(f'w {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n') if chess_game.p_move > 0 else print(f'b {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')

                        if True not in [True for s in human_code if str(p).lower() == s]
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
                            for i,x in enumerate(state):
                                game_train_data[f'value{i}'] = [x]*len(log)
                                game_train_data[f'value{i}'] = game_train_data[f'value{i}'].astype(float)
                            if 'imag_log' in locals():
                                game_train_data = game_train_data.append(imag_log,ignore_index=True)
                            game_train_data = game_train_data.astype(float)
                            end = True
                        break
                if end == True:
                    break
                if valid == True:
                    chess_game.p_move = chess_game.p_move * (-1)
            if end == True:
                break
    return state, game_train_data, a_colour

if __name__ == '__main__':
    GAMES = 10 #Games to play on each board
    BOARDS = 1 #Amount of boards to play on at a time
    BEST_OF = 5 #Amount of games played when evaluating the models

    #Player parameters
    folder = 'gym/chess/data' #Folder name where data is saved
    players = [
        f'{folder}/active_param.json',
        f'{folder}/new_param.json'
    ]

    #Training parameters
    bsz = 50 #Batch size
    lr = 0.0005 #Learning rate
    total_loss = 0.0 #Initalize total loss

    #Begin training games
    train_data = pd.DataFrame()
    game_results = {
        'new':0,
        'tie':0,
        'active':0
    }
    for epoch in range(GAMES):
        print(f'STARTING GAMES\n')
        state, g_results, a_colour = train.play_game(
            0,
            epoch,
            train = True,
            players = players
        )
        #WHO WINS
        if ((state == [0,0,1] and a_colour == 'b') or (state == [1,0,0] and a_colour == 'w')) and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            print('ACTIVE AI WINS\n')
            game_results['active'] += 1
        elif ((state == [0,0,1] and a_colour == 'w') or (state == [1,0,0] and a_colour == 'b')) and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            print('NEW AI WINS\n')
            game_results['new'] += 1
        elif state == [1,0,0] and a_colour == 'b' and str(white).lower() != 'ai' and str(black).lower() == 'ai':
            print('YOU WIN\n')
            game_results['active'] += 1
        elif state == [0,0,1] and a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() != 'ai':
            print('YOU WIN\n')
            game_results['active'] += 1
        elif state == [1,0,0] and a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() != 'ai':
            print('NEW AI WINS\n')
            game_results['new'] += 1
        elif state == [0,0,1] and a_colour == 'b' and str(white).lower() != 'ai' and str(black).lower() == 'ai':
            print('NEW AI WINS\n')
            game_results['new'] += 1
        else:
            print('TIE GAME\n')
            game_results['tie'] += 1

        train_data = train_data.append(g_results,ignore_index=True)
        train_data = train_data.drop_duplicates()
        print(epoch, game_results, '\n')
        if sum([v for v in game_results.values()]) >= BEST_OF and game_results['new']/max(sum([game_results['new'],game_results['active']]),1) >= 0.51 and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            print(f"NEW MODEL OUTPERFORMED ACTIVE MODEL ({round(game_results['new']/max(sum([game_results['new'],game_results['active']]),1),3)*100}%)\n")
            copyfile(os.path.join(folder,new_weights),os.path.join(folder,active_weights)) #Overwrite active model with new model
        if sum([v for v in game_results.values()]) >= BEST_OF:
            game_results = {
                'new':0,
                'tie':0,
                'active':0
            }
        #Load current new model
        model = TransformerModel(
            m_param['input_size'], #Size of input layer 8x8 board
            m_param['ntokens'], #The size of vocabulary
            m_param['emsize'], #Embedding dimension
            m_param['nhead'], #The number of heads in the multiheadattention models
            m_param['nhid'], #The dimension of the feedforward network model in nn.TransformerEncoder
            m_param['nlayers'], #The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            m_param['dropout'] #The dropout value
        ).to(device) #Initialize the transformer model
        filepath = os.path.join(folder, new_weights)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
        #Initailize training
        criterion = torch.nn.BCELoss() #Binary cross entropy loss
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) #Optimization algorithm using stochastic gradient descent
        model.train() #Turn on the train mode
        start_time = time.time() #Get time of starting process
        train_data = train_data.sample(frac=1).reset_index(drop=True) #Shuffle training data
        train_data = torch.tensor(train_data.values) #Set training data to a tensor
        #Start training model
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bsz)):
            data, v_targets, p_targets = TransformerModel.get_batch(train_data, i, bsz) #Get batch data with the selected targets being masked
            output = model(data) #Make prediction using the model
            v_loss = criterion(output[0], v_targets) #Apply loss function to results
            p_loss = criterion(output[1], p_targets) #Apply loss function to results
            loss = v_loss + p_loss
            loss.backward() #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item() #Increment total loss
        #Updated new model
        filepath = os.path.join(folder, new_weights)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({
            'state_dict': model.state_dict(),
        }, filepath)
        print(f'{epoch} | {BOARDS} games | {time.time() - start_time} ms | {train_data.size(0)} samples | {total_loss/(GAMES*BOARDS)} loss\n')
        #Garbage cleanup
        total_loss = 0
        train_data = pd.DataFrame()
        del data
        del v_targets
        del p_targets
        del output
        del v_loss
        del p_loss
        del loss
        del checkpoint
        del g_results
        del model
        del criterion
        del optimizer
