from search import MCTS
from model import ChappieZero

class Agent():
    def __init__(self,game,folder='/data',model_name='model-active.pth.tar',param_name='model-parameters.json'):
        #Model parameters
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use
        with open(os.path.join(folder,param_name)) as f:
            if os.path.exists(f):
                m_param = json.load(f)
            else:
                raise Exception('ERROR - Supplied model parameter file does not exist.')
        #Build model
        self.Model = ChappieZero(
            input_size,
            latent_size,
            reward_size,
            policy_size,
            ntoken = 30,
            embedding_size=64,
            padding_idx=29,
            encoder_dropout=0.5,
            latent_inner=64,
            latent_heads=1,
            latent_dropout=0.5,
            perceiver_inner=64,
            recursions=1,
            transformer_blocks=1,
            cross_heads=1,
            self_heads=1,
            cross_dropout=0.5,
            self_dropout=0.5,
            reward_inner=64,
            reward_heads=1,
            reward_dropout=0.5,
            policy_inner=64,
            policy_heads=1,
            policy_dropout=0.5
        ).to(self.Device)
        while open(os.path.join(folder,model_name)) as f:
            if os.path.exists(f):
                checkpoint = torch.load(f,map_location=self.Device)
                self.Model.load_state_dict(checkpoint['state_dict'])
        #Inialize search
        self.MCTS = MCTS(self.Model)

    def choose_action(self,game):
        self.MCTS.search(self,game,parent_hash)
        pass
