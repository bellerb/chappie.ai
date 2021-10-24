"""
Monte Carlo Tree Search algorithm used to search game tree for the best move
"""
class MCTS:
    """
    Input: max_depth - integer representing the max depth in the graph the angent is alowed (Default=5) [OPTIONAL]
           folder - string representing the location of the folder storing the model parameters (Default='ai_ben/data') [OPTIONAL]
           filename - string representing the model name (Default='model.pth.tar') [OPTIONAL]
    Description: MCTS initail variables
    Output: None
    """
    def __init__(self,model,Cpuct=0.77,max_depth=5,user=None:
        self.tree = {} #Game tree
        self.user = user #What player is searching
        self.Cpuct = Cpuct #Exploration hyper parameter [0-1]
        self.depth = 0 #Curent node depth
        self.max_depth = max_depth #Max allowable depth
        self.model = model #Model used for exploitation

    """
    Node for each state in the game tree
    """
    class Node:
        """
        Input: None
        Description: MCTS initail variables
        Output: None
        """
        def __init__(self):
            self.Q = 0 #Reward
            self.P = 0 #Policy
            self.N = 0 #Visits

    """
    Input: game - object containing the game current state
    Description: Search the game tree using upper confidence value
    Output: tuple of integers representing node reward and policy values
    """
    def search(self,game,parent_hash):
        self.depth += 1
        if parent_hash not in self.tree:
            self.tree[parent_hash] = self.Node()
        self.tree[parent_hash].N += 1
        if self.tree[parent_hash].Q == 0:
            #Use NN
            enc_state = self.plumbing.encode_state(game)
            v,p = self.Model(enc_state)
            return v, p
        else:
            if self.depth == self.max_depth:
                return self.tree[parent_hash].Q, self.tree[parent_hash].P
            b_action = None
            #CHECK UCB VALUES FOR BEST ACTION
            if b_action != None:
                v,p = self.search(b_action)
                return v,p
            return self.tree[parent_hash].Q, self.tree[parent_hash].P
