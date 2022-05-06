import sys 

sys.path.append(r'/data/zmyuan/HDAM/HDAM/Genetic')
import Genetic
from Genetic import population, tools, resnet, genetic_proess
# from Model.Training import Training, Data_loader
import torch
import multiprocessing as mp
import torch


if __name__ == '__main__':
    

    mp.set_start_method('spawn')
    torch.cuda.current_device()

    config_file = r'/data/zmyuan/HDAM/HDAM/Genetic/Setting.ini'
    configer = tools.Configer(config_file) 
    params = configer.get_params()

    log = tools.Recorder(params['DEFAULT']['log_name'], params['DEFAULT']['log_path'])
    logger = log.log()
    
    genetic = genetic_proess.Genetic(int(params['RUNNING']['is_running']), int(params['GENETIC']['generation']), 
                                     int(params['GENETIC']['resnet_type']), int(params['GENETIC']['popu_size']), 
                                     float(params['GENETIC']['cross_prob']), float(params['GENETIC']['muta_prob']), 
                                     params['TRAINING'], logger, params['DEFAULT']['models_path'], 
                                     params['DEFAULT']['cache_file'], params['DEFAULT']['all_indi_file'], 
                                     params['DEFAULT']['last_ind_file'], config_file)
    
    genetic.process() 
    #0.125/0.5/0||0.125/0.25/0.125/0.125||0.5/0.25/0/0/0.5/0.25||0/0.5/0.5
