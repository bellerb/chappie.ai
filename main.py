from skills.chess.game_interface import chess

if __name__ == '__main__':
    print(
    '''
    --------------------------------------------
     Hi I'm Chappie, what would you like to do?
    --------------------------------------------
    '''
    )
    o_bank = [
        'Play Chess (p)',
        'Train Chess (t)'
    ]
    task = -1
    while True:
        u_in = input(''.join(f'* {o}\n' for o in o_bank))
        for i, o in enumerate(o_bank):
            o_hold = str(o).lower().split(' ')
            if str(u_in).lower() == str(o_hold[0]).lower() or str(u_in).lower() == str(o_hold[0]).lower()+' '+str(o_hold[1]).lower() or str(u_in).lower() == str(o_hold[-1]).replace('(','').replace(')','').lower():
                task = i
                break
        if task == -1:
            print(
    '''
    --------------------------------------------
     Invalid option, plase select an option.
    --------------------------------------------
    '''
            )
        else:
            break

    if task == 0:
        chess = chess()
        chess.play_game(
            'TEST',
            0,
            #EPD = '1b4k1/Q7/p2np1/P1P2p2/1P3P2/1R5R/q6P/5rK1 b - -',
            players = [
                #'skills/chess/data/models/test',
                'skills/chess/data/models/test_V2',
                'human'
            ]
        )
    elif task == 1:
        chess = chess()
        chess.traing_session(
            loops = 2,
            games = 1,
            boards = 1,
            best_of = 3,
            #player = 'skills/chess/data/models/test',
            player = 'skills/chess/data/models/test_V2',
            SILENT = False
        )
