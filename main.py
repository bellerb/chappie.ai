from skills.chess.game_interface import chess

chess = chess(train = True)
'''
chess.play_game(
    'TEST',
    0,
    EPD = '1b4k1/Q7/p2np1/P1P2p2/1P3P2/1R5R/q6P/5rK1 b - -',
    players = [
        'human',
        'human'
    ]
)
'''
chess.traing_session(
    games = 5,
    boards = 1,
    best_of = float('inf'),
    players = [
        {
            'param':'skills/chess/data/new_param.json',
            'train':True
        }
    ]
)
