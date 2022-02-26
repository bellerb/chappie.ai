import os
from skills.chess.game_interface import chess

def give_options(o_bank):
    choice = -1
    while True:
        u_in = input(''.join(f'* {o}\n' for o in o_bank) + '\n')
        for i, o in enumerate(o_bank):
            o_hold = str(o).lower().split(' ')
            if str(u_in).lower() == str(o_hold[0]).lower() or str(u_in).lower() == str(o_hold[0]).lower()+' '+str(o_hold[1]).lower() or str(u_in).lower() == str(o_hold[-1]).replace('(','').replace(')','').lower():
                choice = i
                break
        if choice == -1:
            print(
'''
-------------------------------------------------
 Invalid option, plase select an option.
-------------------------------------------------
'''
            )
        else:
            break
    return choice

if __name__ == '__main__':
    print(
'''
-------------------------------------------------
   Hi I'm Chappie, what would you like to do?
-------------------------------------------------
'''
    )
    i = 0
    model_list = []
    for m in os.listdir(f'skills/chess/data/models'):
        if m != '.DS_Store' and '(temp)' not in m:
            model_list.append(f'{m} ({i})')
            i += 1
    o_bank = [
        'Chess (c)'
    ]
    task = give_options(o_bank)
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
        task = give_options(c_tasks)
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
            p_col = give_options(col_choice)
            if p_col == 0:
                players.append('skills/chess/data/models/test_V3/parameters.json')
            else:
                players.insert(0, 'skills/chess/data/models/test_V3/parameters.json')
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
                SILENT = False
            )
            print(state)
        elif task == 1:
            print(
'''
-------------------------------------------------
Training chess bots.
Please select which bot you would like to train?
-------------------------------------------------
'''
            )
            p_col = give_options(model_list)
            player = f'skills/chess/data/models/{model_list[p_col].split("(")[0].strip()}'
            chess = chess()
            chess.traing_session(
                loops = 2,
                games = 20,
                boards = 1,
                best_of = 3,
                player = player,
                SILENT = False
            )
        elif task == 2:
            players = []
            print(
'''
-------------------------------------------------
Evaluating chess bots.
Please select a bot to play white?
-------------------------------------------------
'''
            )
            p_col = give_options(model_list)
            players.append(f'skills/chess/data/models/{model_list[p_col].split("(")[0].strip()}/parameters.json')
            print(
'''
-------------------------------------------------
Please select a bot to play black?
-------------------------------------------------
'''
            )
            p_col = give_options(model_list)
            players.append(f'skills/chess/data/models/{model_list[p_col].split("(")[0].strip()}/parameters.json')
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
                SILENT = False
            )
            print(state)
