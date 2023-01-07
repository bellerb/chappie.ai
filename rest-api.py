import os
import json
import numpy as np
from copy import deepcopy
from flask import Flask, jsonify
from datetime import datetime

from tasks.games.chess.chess import Chess
from skills.chess.game_plumbing import Plumbing

FILE_NAME = "rest-api-data.json"

app = Flask(__name__)
game = Chess()
plumbing = Plumbing()


@app.route('/')
def index():
    return jsonify({
        'name': 'Chappie',
        'version': '0.0.1',
        'release': datetime.now()
    })


@app.route('/chess/new-game', methods=['GET'])
def chess_new_game():
    game.reset()
    # Reset game log
    with open(FILE_NAME, 'w') as json_file:
        game_data = {'LOG': []}
        json.dump(game_data, json_file, indent=4)
    return jsonify({
        'board': game.board,
        'p_move': game.p_move,
        'state': game.is_end(),
        'valid_moves': game.possible_board_moves()
    })


@app.route('/chess/current-game', methods=['GET'])
def chess_current_game():
    # Load saved game
    if os.path.isfile(FILE_NAME):
        with open(FILE_NAME, 'r') as json_file:
            game_data = json.load(json_file)
        if 'EPD' in game_data:
            game.reset(EPD=game_data['EPD'])
    state = game.is_end()
    return jsonify({
        'board': game.board,
        'p_move': game.p_move,
        'state': state,
        'valid_moves': game.possible_board_moves()
    })


@app.route('/chess/move-piece/<cur>/<next>', methods=['GET'])
def chess_move_piece(cur, next):
    if game.move(cur, next):
        state = game.is_end()
        if state == [0, 0, 0] and game.check_state(game.EPD_hash()) == 'PP':
            game.pawn_promotion()
        # Log game
        if os.path.isfile(FILE_NAME):
            with open(FILE_NAME, 'r') as json_file:
                game_data = json.load(json_file)
        else:
            game_data = {'LOG': []}
        a_map = np.zeros((8, 8, 8, 8))
        a_map[game.y.index(cur[1])][game.x.index(
            cur[0].lower())][game.y.index(next[1])][game.x.index(next[0].lower())] = 1
        a_map = a_map.flatten()
        b_a = np.where(a_map == 1)[0][0]
        enc_state = plumbing.encode_state(game)
        player = deepcopy(game.p_move)
        game.p_move *= -1
        game_data['EPD'] = game.EPD_hash()
        game_data['LOG'].append({
            **{f'state{i}': float(s) for i, s in enumerate(enc_state[0])},
            **{f'prob{x}': 1 if x == ((cur[0]+(cur[1]*8))*64)+(next[0]+(next[1]*8)) else 0 for x in range(4096)},
            **{'action': int(b_a)},
            'player': player
        })
        with open(FILE_NAME, 'w') as json_file:
            json.dump(game_data, json_file, indent=4)
    else:
        state = game.is_end()
        print("INVALID MOVE")
    return jsonify({
        'board': game.board,
        'p_move': game.p_move,
        'state': state,
        'valid_moves': game.possible_board_moves()
    })


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080))
    )
