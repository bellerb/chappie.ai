class Plumbing():
    def __init__(
        self,
        folder='ai_ben/data',
        filename='token_bank.csv'
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
        self.token_bank = pd.read_csv(f'{folder}/{filename}') #All tokens

    def encode_state(self, game):
        """
        Input: game - object containing the game current state
        Description: encode the game board as tokens for the NN
        Output: list containing integers representing a tokenized game board
        """
        temp_board = deepcopy(game.board)
        for y,row in enumerate(temp_board):
            for x,peice in enumerate(row):
                if peice != 0:
                    temp_board[y][x] = f'{self.notation[abs(peice)]}w' if peice > 0 else f'{self.notation[abs(peice)]}b'
                else:
                    temp_board[y][x] = 'PAD'
        if len(temp_board) > 0:
            flat = [x for y in temp_board for x in y]
            result = [self.token_bank['token'].eq(t).idxmax() for t in flat]
            result.insert(0, 1) if game.p_move == 1 else result.insert(0, 2)
        else:
            result = []
        return torch.tensor([result])

    def multi_process(func, workers=None):
        """
        Input: func - list of dicitonary's containing the functions you want to run in parallel
        Description: run multiple funcitons in parallel
        Output: dictionary containing the output from all the supplied functions
        """
        data = {}
        with ProcessPoolExecutor(max_workers = workers) as ex:
            future_func = {ex.submit(f['func'], *f['args']):f['name'] for f in func}
            for future in as_completed(future_func):
                data[future_func[future]] = future.result()
            return data
