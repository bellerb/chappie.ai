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
    def __init__(self,model,Cpuct=0.77,max_depth=5,user=None):
        self.tree = {} #Game tree
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
            self.Parent = '' #Parent Node

    """
    Input: Task - object containing the task the agent is trying to do
    Description: return best action state using upper confidence value
    Output: string containing the best action state
    """
    def UCB(self,Task):
        u = {}
        actions = Task.actions
        for a in actions:
            u[a] = self.tree[a].Q + (self.Cpuct * self.tree[a].P * (math.sqrt(math.log(self.tree[self.tree[a].Parent].N)/(1+self.tree[a].N))))
        a_bank = [k for k,v in u.items() if v == max(u.values())]
        if len(a_bank) > 0:
            return random.choice(a_bank)
        else:
            return None
    """
    Input: Task - object containing the task the agent is trying to do
    Description: Search the task action tree using upper confidence value
    Output: predicted reward
    """
    def search(self,Task):
        if Task.end == True:
            return self.UCB(Task)
        if Task.state() not in self.tree:
            self.tree[Task.state()] = Node()
        if self.tree[Task.state()].Q == 0:
            q,p = self.model.predict(Task.enc_state(Task.state()))
            self.tree[Task.state()].Q = q
            for i,a in enumerate(Task.actions()):
                self.tree[a] = Node()
                self.tree[a].P = p[i]
                self.tree[a].Parent = Task.state()
            return self.tree[Task.state()].Q
        Task.state() = self.UCB(Task)
        if self.depth < self.max_depth:
            self.depth += 1
            self.tree[Task.state()].Q = self.search(Task)
        return self.tree[Task.state()].Q
