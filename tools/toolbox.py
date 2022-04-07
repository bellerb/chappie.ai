import torch
from time import sleep
import pandas as pd
from os import listdir
from os.path import exists, isfile
from shutil import copyfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class ToolBox:
    def build_embedding_db(representation, backbone, f_name = None, s_header = 'state'):
        """
        Input: representation - model used to create game state representations
               backbone - model used as the backbone of our multi task model
               f_name - string representing the game state data (Default = None) [OPTIONAL]
               s_header - string representing the main name used in game state token database (Default = 'state') [OPTIONAL]
        Description: builds embedding database from tokens
        Output: dataframe containing embeddings
        """
        if f_name is not None and isfile(f_name):
            t_db = pd.read_csv(f_name)
            t_db = t_db[[h for h in t_db if s_header in h]].drop_duplicates()
        else:
            t_db = None
        if t_db is not None:
            headers = [h for h in t_db]
            e_db = []
            for i, row in t_db.iterrows():
                hold = torch.tensor(row[headers].values).to(torch.long).reshape(1, len(headers))
                hold = representation(hold)
                hold = backbone(hold, torch.zeros(1, 1).to(torch.long))
                e_db.append(hold.tolist())
            del hold
            e_db = pd.DataFrame(e_db)
        else:
            e_db = None
        return e_db

    def get_kNN(chunks, e_db, k = 2):
        """
        Input: chunks - tensor containing initial data
               e_db - dataframe containing embeddings
        Description: find k-nearest-neighbours of input tensor
        Output: tensor containing the k-nearest-neighbours of input tensor
        """
        neighbours = torch.tensor([])
        for i, chunk in enumerate(chunks):
            e_db['L2'] = e_db.apply(lambda x:torch.linalg.norm(chunk - torch.tensor(x[0][chunk.size(0) * i:chunk.size(0) * (i + 1)])).item(), axis=1)
            kNN = torch.tensor([e_db.nsmallest(k, ['L2'])[0].tolist()])
            neighbours = torch.cat([neighbours, kNN])
        return neighbours

    def multi_process(func, workers = None):
        """
        Input: func - list of dicitonary's containing the functions you want to run in parallel
        Description: run multiple funcitons in parallel
        Output: dictionary containing the output from all the supplied functions
        """
        data = {}
        with ProcessPoolExecutor(max_workers = workers) as ex:
            future_func = {}
            for f in func:
                if isinstance(f['args'], tuple) or isinstance(f['args'], list):
                    future_func[ex.submit(f['func'], *f['args'])] = f['name']
                else:
                    future_func[ex.submit(f['func'], f['args'])] = f['name']
            for future in as_completed(future_func):
                data[future_func[future]] = future.result()
        return data

    def multi_thread(func, workers = None):
        """
        Input: func - list of dicitonary's containing the functions you want to run in parallel
        Description: run multiple funcitons in parallel
        Output: dictionary containing the output from all the supplied functions
        """
        data = {}
        with ThreadPoolExecutor(max_workers = workers) as ex:
            future_func = {}
            for f in func:
                if isinstance(f['args'], tuple) or isinstance(f['args'], list):
                    future_func[ex.submit(f['func'], *f['args'])] = f['name']
                else:
                    future_func[ex.submit(f['func'], f['args'])] = f['name']
            for future in as_completed(future_func):
                data[future_func[future]] = future.result()
        return data

    def overwrite_model(p1, p2):
        """
        Input: p1 - string representing player 1 paramater file
               p2 - string representing player 2 paramater file
        Description: overwrite player 2 model with player 1 model
        Output: None
        """
        if exists(p1):
            if exists(p2) == False:
                os.makedirs(p2) #Create folder
            for i, m in enumerate(listdir(p1)):
                copyfile(
                    f"{p1}/{m}",
                    f"{p2}/{m}"
                ) #Overwrite active model with new model

    def give_options(o_bank):
        """
        Input: o_bank - list of strings representing options
        Description: get user input from option bank
        Output: integer representing the index of the option
        """
        choice = -1
        while True:
            u_in = input(''.join(f'* {o}\n' for o in o_bank) + '\n')
            for i, o in enumerate(o_bank):
                o_hold = str(o).lower().split(' ')
                if str(u_in).lower() == str(o_hold[0]).lower() or str(u_in).lower() == str(o_hold[0]).lower()+' '+str(o_hold[1]).lower() or str(u_in).lower() == str(o_hold[-1]).replace('(','').replace(')','').lower():
                    choice = i
                    break
            if choice == -1:
                print(
    '''
    -------------------------------------------------
     Invalid option, plase select an option.
    -------------------------------------------------
    '''
                )
            else:
                break
        return choice

    def update_ELO(p1, p2, k = 32, tie = False):
        """
        Input: p1 - float representing the winning players current ELO
               p2 - float representing the loosing players current ELO
               k - integer representing ELO hyperparameter (Default = 32) [OPTIONAL]
               tie - boolean control for if game was a tie or not (Default = False) [OPTIONAL]
        Description: update players ELO after game
        Output: tuple containing updated player 1 and player 2 ELO
        """
        R_p1 = 10 ** (p1 / 400)
        R_p2 = 10 ** (p2 / 400)

        E_p1 = R_p1 / (R_p1 + R_p2)
        E_p2 = R_p2 / (R_p2 + R_p1)

        if tie == False:
            S_p1 = 1
            S_p2 = 0
        else:
            S_p1 = S_p2 = 0.5

        ELO_p1 = p1 + (k * (S_p1 - E_p1))
        ELO_p2 = p2 + (k * (S_p2 - E_p2))
        return (ELO_p1, ELO_p2)
