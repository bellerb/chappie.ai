from copy import deepcopy

class Chess:
    """
    Game Engine for playing chess in the console
    """

    def __init__(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        """
        Input: EPD - string representing the EPD hash you want to start the game with
                     (Default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -') [OPTIONAL]
        Description: Chess initail variables
        Output: None
        """
        self.x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] #Board x representation
        self.y = ['8', '7', '6', '5', '4', '3', '2', '1'] #Board y representation
        self.notation = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6} #Map of notation to part number
        self.parts = {1:'Pawn', 2:'Knight', 3:'Bishop', 4:'Rook', 5:'Queen', 6:'King'} #Map of number to part
        self.c_escape = {} #Possible check escapes
        self.reset(EPD=EPD) #Reset game board and state

    def reset(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        """
        Input: EPD - string representing the EPD hash you want to start the game with
                     (Default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -') [OPTIONAL]
        Description: reset game board to desired EPD hash
        Output: None
        """
        self.log = [] #Game log
        self.init_pos = EPD #Inital position
        self.EPD_table = {} #EPD hashtable
        self.p_move = 1 #Current players move white = 1 black = -1
        self.castling = [1, 1, 1, 1] #Castling control
        self.en_passant = None #En passant control
        self.prev_move = None #Previous move
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]] #Generate empty chess board
        self.load_EPD(EPD) #Load in game starting position

    def display(self):
        """
        Input: None
        Description: display the current game state
        Output: None
        """
        result = '  a b c d e f g h  \n  ----------------\n'
        for c, y in enumerate(self.board):
            result += f'{8-c}|'
            for x in y:
                if x != 0:
                    n = getattr(Chess, self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.lower() if x < 0 else getattr(Chess, self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.upper()
                    if n == '':
                        n = 'p' if x < 0 else 'P'
                    result += n
                else:
                    result += '.'
                result += ' '
            result += f'|{8-c}\n'
        result += '  ----------------\n  a b c d e f g h\n'
        print(result)

    def board_2_array(self, cord):
        """
        Input: cord - string representing the game board cordinate you want to convert
        Description: convert board string cordinate to a matrix index of the cordinate
        Output: tuple of x,y cordinates for game board matrix
        """
        cord = list(cord)
        if len(cord) == 2 and str(cord[0]).lower() in self.x and str(cord[1]) in self.y:
            return self.x.index(str(cord[0]).lower()), self.y.index(str(cord[1]))
        else:
            return None

    def EPD_hash(self):
        """
        Input: None
        Description: represent current game state as a EPD hash
        Output: string representing the current game in EPD hash format
        """
        result = ''
        for i, rank in enumerate(self.board):
            e_count = 0
            for square in rank:
                if square == 0:
                    e_count += 1
                else:
                    if e_count > 0:
                        result += str(e_count)
                    e_count = 0
                    p_name = self.parts[int(square) if square > 0 else int(square)*(-1)] #Get name of part
                    p_notation = getattr(Chess, p_name)().notation
                    if p_notation == '':
                        p_notation = 'p'
                    if square < 0:
                        p_notation = str(p_notation).lower()
                    else:
                        p_notation = str(p_notation).upper()
                    result += p_notation
            if e_count > 0:
                result += str(e_count)
            if i < 7:
                result += '/'
        if self.p_move == -1:
            result += ' w'
        else:
            result += ' b'
        result += ' '
        if sum(self.castling) == 0:
            result += '-'
        else:
            if self.castling[0] == 1:
                result += 'K'
            if self.castling[1] == 1:
                result += 'Q'
            if self.castling[2] == 1:
                result += 'k'
            if self.castling[3] == 1:
                result += 'q'
        result += ' '
        if self.en_passant == None:
            result += '-'
        else:
            result += f'{self.x[self.en_passant[0]]}{self.y[self.en_passant[1]]}'
        return result

    def load_EPD(self, EPD):
        """
        Input: EPD - string representing the current game in EPD hash format
        Description: update game state to requirements in supplied EPD hash
        Output: boolean representing the outcome of the function
        """
        data = EPD.split(' ')
        if len(data) == 4:
            for x, rank in enumerate(data[0].split('/')):
                y = 0
                for p in rank:
                    if p.isdigit():
                        for i in range(int(p)):
                            self.board[x][y] = 0
                            y += 1
                    else:
                        self.board[x][y] = self.notation[str(p).lower()]*(-1) if str(p).islower() else self.notation[str(p).lower()]
                        y += 1
            self.p_move = 1 if data[1] == 'w' else -1
            if 'K' in data[2]:
                self.castling[0] = 1
            else:
                self.castling[0] = 0
            if 'Q' in data[2]:
                self.castling[1] = 1
            else:
                self.castling[1] = 0
            if 'k' in data[2]:
                self.castling[2] = 1
            else:
                self.castling[2] = 0
            if 'q' in data[2]:
                self.castling[3] = 1
            else:
                self.castling[3] = 0
            self.en_passant = None if data[3] == '-' else self.board_2_array(data[3])
            return True
        else:
            return False

    def log_move(self, part, cur_cord, next_cord, cur_pos, next_pos, n_part=None):
        """
        Input: part - integer representing the peice that was moved
               cur_cord - string representing the current cordinate of the peice
               next_pos - string representing the next cordinate of the peice
               n_part - string representing the new part pawn became, used for pawn promotion (Default = None) [OPTIONAL]
        Description: log player move using chess notation (English)
        Output: None
        """
        #to remove ambiguity where multiple pieces could make the move add starting identifier after piece notation ex Rab8
        if part == 6 * self.p_move and next_pos[0] - cur_pos[0] == 2:
            move = '0-0'
        elif part == 6 * self.p_move and next_pos[0] - cur_pos[0] == -2:
            move = '0-0-0'
        elif part == 1 * self.p_move and n_part != None:
            move = f'{str(next_cord).lower()}={str(n_part).upper()}'
        else:
            p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
            move = str(getattr(Chess, p_name)().notation).upper() #Get part notation
            if self.board[next_pos[1]][next_pos[0]] != 0 or (next_pos == self.en_passant and (part == 1 or part == -1)): #Check if there is a capture
                move += 'x' if move != '' else str(cur_cord)[0] + 'x'
            move += str(next_cord).lower()
        self.log.append(move)

    def move(self, cur_pos, next_pos):
        """
        Input: cur_cord - string representing the current cordinate of the peice
               next_pos - string representing the next cordinate of the peice
        Description: move peice on game board
        Output: boolean representing the state of the function
        """
        cp = self.board_2_array(cur_pos)
        np = self.board_2_array(next_pos)
        if self.valid_move(cp, np) is True:
            part = self.board[cp[1]][cp[0]]
            if np == self.en_passant and (part == 1 or part == -1):
                self.board[self.en_passant[1]-(self.p_move*(-1))][self.en_passant[0]] = 0
            self.log_move(part, cur_pos, next_pos, cp, np)
            self.prev_move = self.board
            if (part == 1 and np[1] == 4) or (part == -1 and np[1] == 3):
                self.en_passant = (np[0], np[1]+1) if part == 1 else (np[0], np[1]-1)
            elif part == 6*self.p_move and np[0]-cp[0] == 2:
                self.board[np[1]][np[0]-1] = 4*self.p_move
                self.board[np[1]][np[0]+1] = 0
            elif part == 6*self.p_move and np[0]-cp[0] == -2:
                self.board[np[1]][np[0]+1] = 4*self.p_move
                self.board[np[1]][np[0]-2] = 0
            else:
                self.en_passant = None
            if part == 6*self.p_move:
                if self.p_move == 1:
                    self.castling[0] = 0
                    self.castling[1] = 0
                else:
                    self.castling[2] = 0
                    self.castling[3] = 0
            elif part == 4*self.p_move:
                if self.p_move == 1:
                    if cp == (0, 7):
                        self.castling[1] = 0
                    else:
                        self.castling[0] = 0
                else:
                    if cp == (0, 0):
                        self.castling[3] = 0
                    else:
                        self.castling[2] = 0
            self.board[cp[1]][cp[0]] = 0
            self.board[np[1]][np[0]] = part
            hash = self.EPD_hash()
            if hash in self.EPD_table:
                self.EPD_table[hash] += 1
            else:
                self.EPD_table[hash] = 1
            #self.p_move = self.p_move * (-1)
            return True
        return False

    def valid_move(self, cur_pos, next_pos):
        """
        Input: cur_cord - string representing the current cordinate of the peice
               next_pos - string representing the next cordinate of the peice
        Description: determine if player move is valid game move
        Output: boolean representing the state of the player move
        """
        if cur_pos != None and next_pos != None:
            part = self.board[cur_pos[1]][cur_pos[0]]
            if part * self.p_move > 0 and part != 0:
                p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
                v_moves = getattr(Chess, p_name).movement(self, self.p_move, cur_pos, capture=True)
                if len(self.log) > 0 and '+' in self.log[-1]:
                    v_moves = [m for m in v_moves if cur_pos in self.c_escape and m in self.c_escape[cur_pos]]
                if next_pos in v_moves:
                    return True
        return False

    def possible_board_moves(self, capture=True):
        """
        Input: capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
        Description: determine all possible board moves for current game state
        Output: dictionary containing all possible moves by peice on the board
        """
        moves = {}
        for y, row in enumerate(self.board):
            for x, part in enumerate(row):
                if part != 0:
                    p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
                    p_colour = 1 if part > 0 else -1
                    v_moves = getattr(Chess, p_name).movement(self, p_colour, [x, y], capture=capture)
                    if len(self.log) > 0 and '+' in self.log[-1]:
                        v_moves = [m for m in v_moves if (x, y) in self.c_escape and m in self.c_escape[(x, y)]]
                    moves[f'{str(self.x[x]).upper() if p_colour > 0 else str(self.x[x]).lower()}{self.y[y]}'] = v_moves
        return moves

    def is_checkmate(self, moves):
        """
        Input: moves - dictionary containing all possible moves for current game state
               check - boolean representing if check has been found or not (Default=False) [OPTIONAL]
        Description: determine if the current game state results in a check mate or not
        Output: list representing current state of the game
        """
        self.c_escape = {}
        k_pos = () #King position
        p_blocks = [] #Possible blocks
        u_moves = {} #User potential moves
        #Sort all possible moves
        for p, a in moves.items():
            pos = self.board_2_array(p)
            if (str(p[0]).isupper() and self.p_move == -1) or (str(p[0]).islower() and self.p_move == 1):
                if self.board[pos[1]][pos[0]] == self.King().value * (self.p_move*(-1)):
                    k_pos = (pos, a)
                else:
                    for m in a:
                        if (pos, m) not in p_blocks:
                            p_blocks.append((pos, m))
            else:
                if pos not in u_moves:
                    u_moves[pos] = a
        p_moves = [m for a in u_moves.values() for m in a]
        #Check if checkmate is in posible moves
        if len(k_pos) > 0 and k_pos[0] not in p_moves:
            return [0, 0, 0]
        elif len(k_pos) == 0:
            for y, row in enumerate(self.board):
                if self.King().value * (self.p_move*(-1)) in row:
                    k_pos = ((row.index(self.King().value*(self.p_move*(-1))), y), [])
                    break
        if len(k_pos) > 0 and k_pos[0] in p_moves:
            for m in p_blocks:
                i_game = deepcopy(self)
                i_game.p_move = i_game.p_move * (-1)
                i_game.move(f'{self.x[m[0][0]]}{self.y[m[0][1]]}', f'{self.x[m[1][0]]}{self.y[m[1][1]]}') #Move king
                i_game.p_move = i_game.p_move * (-1)
                i_moves = i_game.possible_board_moves(capture=True) #Imaginary moves
                if True not in [True for k in i_moves if k_pos[0] in i_moves[k]]: #Check if moved king still in check
                    #if len(self.log) > 0 and self.log[-1][-1] is not '+':
                        #self.log[-1] += '+' #Check
                    if m[0] in self.c_escape:
                        self.c_escape[m[0]].append(m[1])
                    else:
                        self.c_escape[m[0]] = [m[1]]
                    #return [0, 0, 0]
            for m in k_pos[1]:
                if m not in p_moves:
                    i_game = deepcopy(self)
                    i_game.p_move = i_game.p_move * (-1)
                    i_game.move(f'{self.x[k_pos[0][0]]}{self.y[k_pos[0][1]]}', f'{self.x[m[0]]}{self.y[m[1]]}') #Move king
                    i_game.p_move = i_game.p_move * (-1)
                    i_moves = i_game.possible_board_moves(capture=True) #Imaginary moves
                    if True not in [True for k in i_moves if m in i_moves[k]]: #Check if moved king still in check
                        #if len(self.log) > 0 and self.log[-1][-1] is not '+':
                            #self.log[-1] += '+' #Check
                        if k_pos[0] in self.c_escape:
                            self.c_escape[k_pos[0]].append(m)
                        else:
                            self.c_escape[k_pos[0]] = [m]
                        #return [0, 0, 0]
            if len(self.c_escape) > 0:
                self.log[-1] += '+' #Check
                return [0, 0, 0]
            elif self.p_move == -1:
                self.log[-1] += '#'
                return [0, 0, 1] #Black wins
            else:
                self.log[-1] += '#'
                return [1, 0, 0] #White wins
        else:
            return [1, 0, 0] if self.p_move == 1 else [0, 0, 1]

    def pawn_promotion(self, n_part=None):
        """
        Input: n_part - string representing the new part you want to promote your pawn to (Default=None) [OPTIONAL]
        Description: update game board with new part for pawn promotion
        Output: boolean representing the state of the function
        """
        if n_part == None:
            while True:
                n_part = input('\nPawn Promotion - What peice would you like to switch too:\n\n*Queen[q]\n*Bishop[b]\n*Knight[n]\n*Rook[r]\n')
                if str(n_part).lower() not in ['q', 'b', 'n', 'r', 'queen', 'bishop', 'knight', 'rook']:
                    print('\nInvalid Option')
                else:
                    break
            if len(n_part) > 1:
                n_part = getattr(Chess, str(n_part).capitalize())().notation
        part = self.notation[str(n_part).lower()]*self.p_move
        pos = self.board_2_array(self.log[-1].replace('+', '').split('x')[-1])
        if pos != None:
            self.board[pos[1]][pos[0]] = part
            self.log[-1] += f'={str(n_part).upper()}'
            return True
        else:
            return False

    def fifty_move_rule(self, moves, choice=None):
        """
        Input: moves - dictionary containing all possible moves for current game state
               choice - boolean representing if you want a draw or not (Default=None) [OPTIONAL]
        Description: check current state of the game for the fifty move rule
        Output: boolean representing the state of the function
        """
        if len(self.log) > 100:
            for m in self.log[-100:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        if choice is None:
            while True:
                choice = input('Fifty move rule - do you want to claim a draw? [Y/N]')
                if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                    return True
                elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                    return False
            print('Unsupported answer')
        else:
            return choice
        return False

    def seventy_five_move_rule(self, moves):
        """
        Input: moves - dictionary containing all possible moves for current game state
        Description: check current state of the game for the seventy five move rule
        Output: boolean representing the state of the function
        """
        if len(self.log) > 150:
            for m in self.log[-150:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        return True

    def three_fold_rule(self, hash, choice=None):
        """
        Input: hash - string representing the game state you want to check for in game EPD hash table
               choice - boolean representing the option of if you want to ask for a draw or not (Default = None) [OPTIONAL]
        Description: check current state of the game for the three fold rule
        Output: boolean representing the state of the function
        """
        if hash in self.EPD_table:
            if self.EPD_table[hash] == 3:
                if choice is None:
                    while True:
                        choice = input('Three fold rule - do you want to claim a draw? [Y/N]')
                        if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                            return True
                        elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                            return False
                        print('Unsupported answer')
                else:
                    return choice
        return False

    def five_fold_rule(self, hash):
        """
        Input: hash - string representing the game state you want to check for in game EPD hash table
        Description: check current state of the game for the five fold rule
        Output: boolean representing the state of the function
        """
        if hash in self.EPD_table:
            if self.EPD_table[hash] >= 5:
                return True
        return False

    def is_dead_position(self, moves):
        """
        Input: moves - dictionary containing all possible moves for current game state
        Description: check to see if the current state is a dead position
        Output: boolean representing the state of the function
        """
        #King and bishop against king and bishop with both bishops on squares of the same colour
        a_pieces = []
        for y in self.board:
            for x in y:
                if x != 0:
                    a_pieces.append(x)
                if len(a_pieces) > 4:
                    return False
        if len(a_pieces) == 2 and -6 in a_pieces and 6 in a_pieces:
            return True
        elif len(a_pieces) == 3 and ((-6 in a_pieces and 3 in a_pieces and 6 in a_pieces) or (-6 in a_pieces and -3 in a_pieces and 6 in a_pieces)):
            return True
        elif len(a_pieces) == 3 and ((-6 in a_pieces and 2 in a_pieces and 6 in a_pieces) or (-6 in a_pieces and -2 in a_pieces and 6 in a_pieces)):
            return True
        return False

    def is_stalemate(self, moves):
        """
        Input: moves - dictionary containing all possible moves for current game state
        Description: check to see if the current state is a stalemate
        Output: boolean representing the state of the function
        """
        if False not in [False for p, a in moves.items() if len(a) > 0 and ((self.p_move == 1 and str(p[0]).isupper()) or (self.p_move == -1 and str(p[0]).islower()))]:
            return True
        return False

    def is_draw(self, moves, hash, choice=None):
        """
        Input: moves - dictionary containing all possible moves for current game state
               hash - string representing the current game state EPD hash
               choice - boolean representing the option of if you want to ask for a draw or not (Default = None) [OPTIONAL]
        Description: check to see if the current state of game is a draw
        Output: boolean representing the state of the function
        """
        if self.is_stalemate(moves) == True:
            return True
        elif self.is_dead_position(moves) == True:
            return True
        elif self.seventy_five_move_rule(moves) == True:
            return True
        elif self.five_fold_rule(hash) == True:
            return True
        elif self.fifty_move_rule(moves, choice) == True:
            return True
        elif self.three_fold_rule(hash, choice) == True:
            return True
        return False

    def is_end(self, choice=None):
        """
        Input: choice - boolean representing the option of if you want to ask for a draw or not (Default = None) [OPTIONAL]
        Description: check to see if it's the end of the game
        Output: list containing the state of the game
        """
        w_king = False
        b_king = False
        for y, row in enumerate(self.board):
            for x, peice in enumerate(row):
                if self.board[y][x] == self.King().value * (-1):
                    b_king = True
                elif self.board[y][x] == self.King().value:
                    w_king = True
        if w_king == False and b_king == False:
            return [0, 1, 0]
        elif w_king == False:
            return [0, 0, 1]
        elif b_king == False:
            return [1, 0, 0]
        moves = self.possible_board_moves(capture=True)
        check_mate = self.is_checkmate(moves)
        hash = self.EPD_hash()
        if sum(check_mate) > 0:
            return check_mate
        elif self.is_draw(moves, hash, choice) == True:
            return [0, 1, 0]
        return [0, 0, 0]

    def check_state(self, hash):
        """
        Input: hash - string representing the game state you want to check for in game EPD hash table
        Description: check current state of the game
        Output: boolean representing the state of the game or string representing additional action needed
        """
        if len(self.log) > 0 and self.p_move == 1 and (self.log[-1][0].isupper() == False or self.log[-1][0] == 'P') and True in [True for l in self.log[-1] if l == '8']:
            return 'PP' #Pawn promotion
        elif len(self.log) > 0 and self.p_move == -1 and (self.log[-1][0].isupper() == False or self.log[-1][0] == 'P') and True in [True for l in self.log[-1] if l == '1']:
            return 'PP' #Pawn promotion
        elif hash in self.EPD_table and self.EPD_table[hash] == 3:
            return '3F' #3 Fold
        elif len(self.log) > 100:
            for m in self.log[-100:]:
                if 'x' in m or m[0].islower():
                    return None
            return '50M' #50 move
        else:
            return None

    class King:
        """
        Chess peice object for the king
        """

        def __init__(self):
            """
            Input: None
            Description: King initail variables
            Output: None
            """
            self.value = 6 #Numerical value of piece
            self.notation = 'K' #Chess notation

        def movement(game, player, pos, capture=True):
            """
            Input: player - integer representing which player the peice belongs to
                   pos - tuple containing the current position of the peice
                   capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
            Description: show possible moves for peice
            Output: list of possible moves for the peice
            """
            result = []
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+1][pos[0]]*player < 0 or game.board[pos[1]+1][pos[0]] == 0):
                result.append((pos[0], pos[1]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-1][pos[0]]*player < 0 or game.board[pos[1]-1][pos[0]] == 0):
                result.append((pos[0], pos[1]-1))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]][pos[0]+1]*player < 0 or game.board[pos[1]][pos[0]+1] == 0):
                result.append((pos[0]+1, pos[1]))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]][pos[0]-1]*player < 0 or game.board[pos[1]][pos[0]-1] == 0):
                result.append((pos[0]-1, pos[1]))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]+1][pos[0]+1]*player < 0 or game.board[pos[1]+1][pos[0]+1] == 0):
                result.append((pos[0]+1, pos[1]+1))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]+1][pos[0]-1]*player < 0 or game.board[pos[1]+1][pos[0]-1] == 0):
                result.append((pos[0]-1, pos[1]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]-1][pos[0]+1]*player < 0 or game.board[pos[1]-1][pos[0]+1] == 0):
                result.append((pos[0]+1, pos[1]-1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]-1][pos[0]-1]*player < 0 or game.board[pos[1]-1][pos[0]-1] == 0):
                result.append((pos[0]-1, pos[1]-1))
            if (pos == (4, 7) or pos == (4, 0)) and game.board[pos[1]][pos[0]+1] == 0 and game.board[pos[1]][pos[0]+2] == 0 and ((game.castling[0] == 1 and game.p_move == 1) or (game.castling[2] == 1 and game.p_move == -1)):
                result.append((pos[0]+2, pos[1]))
            if (pos == (4, 7) or pos == (4, 0)) and game.board[pos[1]][pos[0]-1] == 0 and game.board[pos[1]][pos[0]-2] == 0 and ((game.castling[1] == 1 and game.p_move == 1) or (game.castling[3] == 1 and game.p_move == -1)):
                result.append((pos[0]-2, pos[1]))
            return result

    class Queen:
        """
        Chess peice object for the queen
        """

        def __init__(self):
            """
            Input: None
            Description: Queen initail variables
            Output: None
            """
            self.value = 5 #Numerical value of piece
            self.notation = 'Q' #Chess notation

        def movement(game, player, pos, capture=True):
            """
            Input: player - integer representing which player the peice belongs to
                   pos - tuple containing the current position of the peice
                   capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
            Description: show possible moves for peice
            Output: list of possible moves for the peice
            """
            result = []
            check = [True, True, True, True, True, True, True, True]
            for c in range(1, 8, 1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+c][pos[0]]*player < 0 or game.board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[0], pos[1]+c))
                    if game.board[pos[1]+c][pos[0]]*player < 0 and capture == True:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-c][pos[0]]*player < 0 or game.board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[0], pos[1]-c))
                    if game.board[pos[1]-c][pos[0]]*player < 0 and capture == True:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]][pos[0]+c]*player < 0 or game.board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c, pos[1]))
                    if game.board[pos[1]][pos[0]+c]*player < 0 and capture == True:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]][pos[0]-c]*player < 0 or game.board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c, pos[1]))
                    if game.board[pos[1]][pos[0]-c]*player < 0 and capture == True:
                        check[3] = False
                else:
                    check[3] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]+c][pos[0]+c]*player < 0 or game.board[pos[1]+c][pos[0]+c] == 0) and check[4] == True:
                    result.append((pos[0]+c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]+c]*player < 0 and capture == True:
                        check[4] = False
                else:
                    check[4] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]+c][pos[0]-c]*player < 0 or game.board[pos[1]+c][pos[0]-c] == 0) and check[5] == True:
                    result.append((pos[0]-c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]-c]*player < 0 and capture == True:
                        check[5] = False
                else:
                    check[5] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]-c][pos[0]+c]*player < 0 or game.board[pos[1]-c][pos[0]+c] == 0) and check[6] == True:
                    result.append((pos[0]+c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]+c]*player < 0 and capture == True:
                        check[6] = False
                else:
                    check[6] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]-c][pos[0]-c]*player < 0 or game.board[pos[1]-c][pos[0]-c] == 0) and check[7] == True:
                    result.append((pos[0]-c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]-c]*player < 0 and capture == True:
                        check[7] = False
                else:
                    check[7] = False
                if True not in check:
                    break
            return result

    class Rook:
        """
        Chess peice object for the rook
        """

        def __init__(self):
            """
            Input: None
            Description: Rook initail variables
            Output: None
            """
            self.value = 4 #Numerical value of piece
            self.notation = 'R' #Chess notation

        def movement(game, player, pos, capture=True):
            """
            Input: player - integer representing which player the peice belongs to
                   pos - tuple containing the current position of the peice
                   capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
            Description: show possible moves for peice
            Output: list of possible moves for the peice
            """
            result = []
            check = [True, True, True, True]
            for c in range(1, 8, 1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+c][pos[0]]*player < 0 or game.board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[0], pos[1]+c))
                    if game.board[pos[1]+c][pos[0]]*player < 0 and capture == True:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-c][pos[0]]*player < 0 or game.board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[0], pos[1]-c))
                    if game.board[pos[1]-c][pos[0]]*player < 0 and capture == True:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]][pos[0]+c]*player < 0 or game.board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c, pos[1]))
                    if game.board[pos[1]][pos[0]+c]*player < 0 and capture == True:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]][pos[0]-c]*player < 0 or game.board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c, pos[1]))
                    if game.board[pos[1]][pos[0]-c]*player < 0 and capture == True:
                        check[3] = False
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    class Bishop:
        """
        Chess peice object for the bishop
        """

        def __init__(self):
            """
            Input: None
            Description: Bishop initail variables
            Output: None
            """
            self.value = 3 #Numerical value of piece
            self.notation = 'B' #Chess notation

        def movement(game, player, pos, capture=True):
            """
            Input: player - integer representing which player the peice belongs to
                   pos - tuple containing the current position of the peice
                   capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
            Description: show possible moves for peice
            Output: list of possible moves for the peice
            """
            result = []
            check = [True, True, True, True]
            for c in range(1, 8, 1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]+c][pos[0]+c]*player < 0 or game.board[pos[1]+c][pos[0]+c] == 0) and check[0] == True:
                    result.append((pos[0]+c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]+c]*player < 0 and capture == True:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]+c][pos[0]-c]*player < 0 or game.board[pos[1]+c][pos[0]-c] == 0) and check[1] == True:
                    result.append((pos[0]-c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]-c]*player < 0 and capture == True:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]-c][pos[0]+c]*player < 0 or game.board[pos[1]-c][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]+c]*player < 0 and capture == True:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]-c][pos[0]-c]*player < 0 or game.board[pos[1]-c][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]-c]*player < 0 and capture == True:
                        check[3] = False
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    class Knight:
        """
        Chess peice object for the knight
        """

        def __init__(self):
            """
            Input: None
            Description: Knight initail variables
            Output: None
            """
            self.value = 2 #Numerical value of piece
            self.notation = 'N' #Chess notation

        def movement(game, player, pos, capture=True):
            """
            Input: player - integer representing which player the peice belongs to
                   pos - tuple containing the current position of the peice
                   capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
            Description: show possible moves for peice
            Output: list of possible moves for the peice
            """
            result = []
            for i in [-1, 1]:
                if pos[0]-i >= 0 and pos[0]-i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (game.board[pos[1]-(2*i)][pos[0]-i]*player < 0 or game.board[pos[1]-(2*i)][pos[0]-i] == 0):
                    result.append((pos[0]-i, pos[1]-(2*i)))
                if pos[0]+i >= 0 and pos[0]+i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (game.board[pos[1]-(2*i)][pos[0]+i]*player < 0 or game.board[pos[1]-(2*i)][pos[0]+i] == 0):
                    result.append((pos[0]+i, pos[1]-(2*i)))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]-i >= 0 and pos[1]-i <= 7 and (game.board[pos[1]-i][pos[0]-(2*i)]*player < 0 or game.board[pos[1]-i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i), pos[1]-i))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]+i >= 0 and pos[1]+i <= 7 and (game.board[pos[1]+i][pos[0]-(2*i)]*player < 0 or game.board[pos[1]+i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i), pos[1]+i))
            return result

    class Pawn:
        """
        Chess peice object for the pawn
        """

        def __init__(self):
            """
            Input: None
            Description: Pawn initail variables
            Output: None
            """
            self.value = 1 #Numerical value of piece
            self.notation = '' #Chess notation

        def movement(game, player, pos, capture=True):
            """
            Input: player - integer representing which player the peice belongs to
                   pos - tuple containing the current position of the peice
                   capture - boolean representing control of if you do not allow moves past peice capture (Default=True) [OPTIONAL]
            Description: show possible moves for peice
            Output: list of possible moves for the peice
            """
            result = []
            init = 1 if player < 0 else 6
            amt = 1 if pos[1] != init else 2
            for i in range(amt):
                if pos[1]-((i+1)*player) >= 0 and pos[1]-((i+1)*player) <= 7 and game.board[pos[1]-((i+1)*player)][pos[0]] == 0:
                    result.append((pos[0], pos[1]-((i+1)*player)))
                else:
                    break
            if pos[1]-player <= 7 and pos[1]-player >= 0 and pos[0]+1 <= 7 and pos[0]+1 >= 0 and game.board[pos[1]-player][pos[0]+1]*player < 0:
                result.append((pos[0]+1, pos[1]-player))
            if pos[1]-player >= 0 and pos[1]-player <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and game.board[pos[1]-player][pos[0]-1]*player < 0:
                result.append((pos[0]-1, pos[1]-player))
            if pos[1]-player <= 7 and pos[1]-player >= 0 and pos[0]+1 <= 7 and pos[0]+1 >= 0 and (pos[0]+1, pos[1]-player) == game.en_passant:
                result.append((pos[0]+1, pos[1]-player))
            if pos[1]-player >= 0 and pos[1]-player <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (pos[0]-1, pos[1]-player) == game.en_passant:
                result.append((pos[0]-1, pos[1]-player))
            return result

if __name__ == '__main__':
    #chess_game = Chess(EPD='4kb2/rpp1p3/6p1/6Np/3Q1B2/4P2b/PPP2PPP/RN1R2K1 w - -')
    chess_game = Chess(EPD='1b4k1/Q7/p2np1/P1P2p2/1P3P2/1R5R/q6P/5rK1 b - -')
    #chess_game = Chess()
    while True:
        if chess_game.p_move == 1:
            print('\nWhites Turn [UPPER CASE]\n')
        else:
            print('\nBlacks Turn [LOWER CASE]\n')
        chess_game.display()
        cur = input('What piece do you want to move?\n')
        next = input('Where do you want to move the piece to?\n')
        if chess_game.move(cur, next) == False:
            if len(chess_game.log) > 0 and '+' in chess_game.log[-1]:
                print('Invalid move, you are in check')
            else:
                print('Invalid move')
        else:
            state = chess_game.is_end()
            if sum(state) > 0:
                print('\n*********************\n      GAME OVER\n*********************\n')
                chess_game.display()
                print('Game Log:\n---------\n')
                print(f'INITIAL POSITION = {chess_game.init_pos}')
                print(f'MOVES = {chess_game.log}')
                print('\nGame Result:\n------------\n')
                if state == [0, 0, 1]:
                    print('BLACK WINS\n')
                elif state == [1, 0, 0]:
                    print('WHITE WINS\n')
                else:
                    print('TIE GAME\n')
                break

            chess_game.p_move = chess_game.p_move * (-1)
