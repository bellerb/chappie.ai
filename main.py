from skills.chess.game_interface import chess

chess = chess(train = True)
'''
chess.play_game(
    'TEST',
    0,
    players = [
        'skills/chess/data/active_param.json',
        'human'
    ]
)
'''
chess.traing_session(
    games = 5,
    boards = 1,
    best_of = 5,
    players = [
        {
            'param':'skills/chess/data/new_param.json',
            'train':True
        }
    ]
)
