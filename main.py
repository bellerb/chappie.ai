from gym.chess.train_chess import chess

chess = chess(train = True)
chess.play_game(
    'TEST',
    0,
    players = [
        'gym/chess/data/active_param.json',
        'human'
    ]
)
