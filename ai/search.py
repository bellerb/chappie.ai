class MCTS:
    """
    Monte Carlo Tree Search algorithm used to search game tree for the best move
    """
    def __init__(
        self, 
        prediction,
        dynamics,
        user = None,
        c1 = 1.25,
        c2 = 19652,
        max_depth = 5
    ):
        """
        Input: prediction - NN used to predict p, v values
               dynamics - NN used to predict the next states hidden values and the reward
               user - integer representing which user the agent is (default = None) [OPTIONAL]
               c1 - float representing a search hyperparameter (default = 1.25) [OPTIONAL]
               c2 - float representing a search hyperparameter (default = 19652) [OPTIONAL]
               max_depth - integer representing the max allowable depth to search (default = 5) [OPTIONAL]
        Description: MCTS initail variables
        Output: None
        """
        self.tree = {} #Game tree
        self.depth = 0 #Curent node depth
        self.max_depth = max_depth #Max allowable depth
        self.c1 = c1 #Exploration hyper parameter 1
        self.c2 = c2 #Exploration hyper parameter 2
        self.f = dynamics #Model used for dynamics
        self.g = prediction #Model used for prediction

    class Node:
        """
        Node for each state in the game tree
        """
        def __init__(self):
            """
            Input: None
            Description: Node initail variables
            Output: None
            """
            self.Q = 0 #Value
            self.R = 0 #Reward
            self.P = 0 #Policy
            self.N = 0 #Visits
            self.Parent = () #Parent Node
            
    def state_hash(s):
        """
        Input: s - tensor representing hidden state of task
        Description: generate unique hash of the supplied hidden state
        Output: integer representing unique hash of the supplied hidden state
        """
        result = hash(str(s))
        return result
    
    def parent_sum(self, n, c = False):
        """
        Input: n - object containing (s, a) edge between nodes
               c - boolean representing if hyperparameter is to be used (default = False) [OPTIONAL]
        Description: find the amount of times parent nodes have been visited
        Output: float containing the resultant value
        """
        result = 0.
        if n.Parent is not ():
            if c == False:
                result += (self.tree[n.Parent].N + self.parent_sum(self.tree[n.Parent]))
            else:
                result += (self.tree[n.Parent].N + self.c2 + 1 + self.parent_sum(self.tree[n.Parent], c = True))
        return result

    def UCB(self, s):
        """
        Input: s - integer representing the task state hash
        Description: return best action state using upper confidence value
        Output: integer containing the best action
        """
        u = {}
        for a in range(self.prediction.action_space):
            x = (self.parent_sum(self.tree[(s, a)])**(0.5))/1+self.tree[(s, a)].N #First part of exploration
            x = x * (self.c1 + math.log(self.parent_sum(self.tree[(s, a)], c = True)/self.c2)) #Second part of exploration
            u[a] = self.tree[(s, a)].Q + (self.tree[(s, a)].P  * x)
        a_bank = [k for k,v in u.items() if v == max(u.values())]
        if len(a_bank) > 0:
            return random.choice(a_bank)
        else:
            return None
        
    def search(self, s, a = None):
        """
        Input: s - tensor representing hidden state of task
               a - integer representing which action is being performed (default = None) [OPTIONAL]
        Description: Search the task action tree using upper confidence value
        Output: predicted value
        """
        s_hash = self.state_hash(s) #Create hash of state [sk-1] for game tree
        if (s_hash, a) not in self.tree:
            self.tree[(s_hash, a)] = Node() #Initialize new game tree node
        else:
            self.tree[(s_hash, a)].N += 1
        if a is not None:
            r_k, s = self.g(s, a) #Reward and state prediction using dynamics function
            self.tree[(s_hash, a)].R = r_k
        if self.tree[(s_hash, a)].Q == 0:
            v, p = self.f(s) #Value and policy prediction using prediction function
            self.tree[(s_hash, a)].Q = v
            sk_hash = self.state_hash(s) #Create hash of state [sk] for game tree
            for a_k, p_a in enumerate(p):
                self.tree[(sk_hash, a_k)] = Node()
                self.tree[(sk_hash, a_k)].P = p_a
                self.tree[(sk_hash, a_k)].Parent = (s_hash, a)
            return self.tree[(s_hash, a)].Q
        a_k = self.UCB(s_hash)
        if self.depth < self.max_depth:
            self.depth += 1
            self.tree[(s_hash, a)].Q = self.search(s, a_k)
        return self.tree[(s_hash, a)].Q
