import os
import torch
import numpy as np
import pandas as pd
from shutil import copyfile
from copy import deepcopy

class Plumbing():
    def __init__(
        self,
        filename = 'skills/chess/data/token_bank.csv',
        move_count = 5
    ):
        """
        Input: None
        Description: Plumbing initail variables
        Output: None
        """
        self.notation = {
            1 : 'p',
            2 : 'n',
            3 : 'b',
            4 : 'r',
            5 : 'q',
            6 : 'k'
        } #Map of notation to part number
        self.x_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] #Board x representation
        self.y_map = ['8', '7', '6', '5', '4', '3', '2', '1'] #Board y representation
        self.token_bank = pd.read_csv(filename) #All tokens
        self.move_count = move_count
        self.move_que = [] #Previous moves

    def encode_state(self, game):
        """
        Input: game - object containing the game current state
        Description: encode the game board as tokens for the NN
        Output: list containing integers representing a tokenized game board
        """
        temp_board = deepcopy(game.board)
        for y, row in enumerate(temp_board):
            for x, peice in enumerate(row):
                if peice !=  0:
                    if peice > 0:
                        temp_board[y][x] = f'{self.notation[abs(peice)]}w'
                    else:
                        temp_board[y][x] = f'{self.notation[abs(peice)]}b'
                else:
                    temp_board[y][x] = 'mt'
        if len(temp_board) > 0:
            flat = [x for y in temp_board for x in y]
            if game.p_move == 1:
                flat.insert(0,'wm')
            else:
                flat.insert(0,'bm')
            result = [self.token_bank['token'].eq(t).idxmax() for t in flat]
        else:
            result = []
        if self.move_count > 1:
            if len(self.move_que) == 0:
                self.move_que = [len(self.token_bank) - 1] * ((len(result) * self.move_count) + 2)
            move_que = result + [self.token_bank['token'].eq('SEP').idxmax()] + self.move_que[:(len(result) * (self.move_count - 1)) + 1]
            return torch.tensor([move_que])
        return torch.tensor([result])

    def decode_state(self, state):
        """
        Input: state - list containing encoded game state
        Description: decode encoded game state for board representation
        Output: list containing integers representing the current game state
        """
        result = [[]]
        for y, row in enumerate(state):
            for x, enc in enumerate(row):
                token = self.token_bank.loc[enc]['token']
                if token != 'mt':
                    numeric = list(self.notation.keys())[list(self.notation.values()).index(token[0])]
                    result[-1].append(numeric if token[-1] == 'w' else numeric * (-1))
                else:
                    result[-1].append(0)
            if y < len(state) - 1:
                result.append([])
        return result

    def parse_action(self, action):
        """
        Input: action - integer representing the choosen action to perform
        Description: parse current action in terms of cordinate system for easy understanding
        Output: tuple containing current position and next position of peice
        """
        a_map = np.zeros(4096)
        a_map[action] = 1
        a_map = a_map.reshape((8,8,8,8))
        a_index = [(cy, cx, ny, nx) for cy, cx, ny, nx in zip(*np.where(a_map == 1))][0]
        c_p = f'{self.x_map[a_index[1]]}{self.y_map[a_index[0]]}'
        n_p = f'{self.x_map[a_index[3]]}{self.y_map[a_index[2]]}'
        return (c_p, n_p)
