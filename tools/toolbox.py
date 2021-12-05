from time import sleep
from os import listdir
from os.path import exists
from shutil import copyfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class ToolBox:
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
