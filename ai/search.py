import math
import random

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
        d_a = .3,
        e_f = .25,
        g_d = 1.,
        Q_max = 1,
        Q_min = -1,
        max_depth = float('inf')
    ):
        """
        Input: prediction - NN used to predict p, v values
               dynamics - NN used to predict the next states hidden values and the reward
               user - integer representing which user the agent is (default = None) [OPTIONAL]
               c1 - float representing a search hyperparameter (default = 1.25) [OPTIONAL]
               c2 - float representing a search hyperparameter (default = 19652) [OPTIONAL]
               d_a = float representing the dirichlet alpha you wish to use (default = 0.3) [OPTIONAL]
               e_f = float representing the exploration fraction you wish to use with dirichlet alpha noise (default = 0.25) [OPTIONAL]
               g_d = float representing the gamma discount to apply to v_k+1 when backpropagating tree (default = 1) [OPTIONAL]
               Q_max = float representing the max value (default = 1) [OPTIONAL]
               Q_min = float representing the min value (default = -1) [OPTIONAL]
               max_depth - integer representing the max allowable depth to search tree (default = inf) [OPTIONAL]
        Description: MCTS initail variables
        Output: None
        """
        self.tree = {} #Game tree
        self.depth = 0 #Curent node depth
        self.max_depth = max_depth #Max allowable depth
        self.c1 = c1 #Exploration hyper parameter 1
        self.c2 = c2 #Exploration hyper parameter 2
        self.d_a = d_a #Dirichlet alpha
        self.e_f = e_f #Exploration fraction
        self.g_d = g_d #Gamma discount
        self.Q_max = Q_max #Max value
        self.Q_min = Q_min #Min value
        self.g = dynamics #Model used for dynamics
        self.f = prediction #Model used for prediction

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
            self.N = 0 #Visits
            self.Q = 0 #Value
            self.R = 0 #Reward
            self.P = None #Policy

    def state_hash(self, s):
        """
        Input: s - tensor representing hidden state of task
        Description: generate unique hash of the supplied hidden state
        Output: integer representing unique hash of the supplied hidden state
        """
        result = hash(str(s))
        return result

    def dirichlet_noise(self, p):
        """
        Input: p - list of floats [0-1] representing action probability distrabution
        Description: add dirichlet noise to probability distrabution
        Output: list of floats [0-1] representing action probability with added noise
        """
        d = np.random.dirichlet([self.d_a] * len(p))
        return (d * self.e_f) + ((1 - self.e_f) * p)

    def pUCT(self, s):
        """
        Input: s - tensor representing hidden state of task
        Description: return best action state using polynomial upper confidence trees
        Output: list containing pUCT values for all acitons
        """
        p_visits = sum([self.tree[(s, b)].N for b in range(self.f.action_space)]) #Sum of all potential nodes
        u_bank = {}
        for a in range(self.f.action_space):
            U = self.tree[(s, a)].P * ((p_visits**(0.5))/(1+self.tree[(s, a)].N)) #First part of exploration
            U *= self.c1 + (math.log((p_visits + (self.f.action_space * self.c2) + self.f.action_space) / self.c2)) #Second part of exploration
            Q_n = (self.tree[(s, a)].Q - self.Q_min) / (self.Q_max - self.Q_min) #Normalized value
            u_bank[a] = Q_n + U
        return u_bank

    def search(self, s, a = None, train = False):
        """
        Input: s - tensor representing hidden state of task
               a - integer representing which action is being performed (default = None) [OPTIONAL]
               train - boolean representing if search is being used in training mode (default = False) [OPTIONAL]
        Description: Search the task action tree using upper confidence value
        Output: predicted value
        """
        s_hash = self.state_hash(s) #Create hash of state [sk-1] for game tree
        if (s_hash, a) not in self.tree:
            self.tree[(s_hash, a)] = self.Node() #Initialize new game tree node
        if a is not None:
            r_k, s = self.g(s, a) #Reward and state prediction using dynamics function
            self.tree[(s_hash, a)].R = r_k
        sk_hash = self.state_hash(s) #Create hash of state [sk] for game tree
        if self.tree[(s_hash, a)].N == 0:
            v_k, p = self.f(s) #Value and policy prediction using prediction function
            if a is None and train == True:
                p = self.dirichlet_noise(p) #Add dirichlet noise to p @ s0
            self.tree[(s_hash, a)].Q = v_k
            #EXPANSION ---
            for a_k, p_a in enumerate(p):
                self.tree[(sk_hash, a_k)] = self.Node()
                self.tree[(sk_hash, a_k)].P = p_a
            self.tree[(s_hash, a)].N += 1
            return self.tree[(s_hash, a)].Q
        u_bank = self.pUCT(sk_hash) #Find best action to perform @ [sk]
        a_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
        a_k = random.choice(a_bank)
        if self.depth < self.max_depth:
            self.depth += 1
            #BACKUP ---
            g = self.tree[(s_hash, a)].R + self.g_d * self.search(s, a_k) #Discounted value at current node
            q_m = (self.tree[(s_hash, a)].N * self.tree[(s_hash, a)].Q + g) / self.tree[(s_hash, a)].N #Mean value
            self.tree[(s_hash, a)].Q = q_m
        self.tree[(s_hash, a)].N += 1
        return self.tree[(s_hash, a)].Q
