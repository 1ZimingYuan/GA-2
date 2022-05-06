""" 
This script aims to describ the fitness evaluation.
"""
from asyncio.log import logger
import os, sys
sys.path.append(r'/data/zmyuan/HDAM')
import torch 
# from torch.utils.tensorboard import SummaryWriter
from population import Individual, Population

import resnet 
# from torchtoolbox.transform import Cutout
from multiprocessing import Process, Queue
from subprocess import Popen, PIPE
import time
from Model.Training import Data_loader, Training
from tools import Recorder
from logging import Logger


data_dir = r'/data/datasets/cifar-10'




class Evaluate:
    def __init__(self, cifar: int, data_dir: str, valid_size: float, batch_size: int, argument: bool, max_epoch: int, valid_test_epoch: int, save_path: str, logger:Logger, model_log_path:str, cache_file:str, all_indi_file:str) -> None: # tb_path: str
        self.cifar = cifar
        self.data_dir = data_dir
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.argument = argument
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = max_epoch
        self.valid_test_epoch = valid_test_epoch
        self.loss_fun = torch.nn.CrossEntropyLoss()
        # self.writer = SummaryWriter(tb_path)
        self.save_path = save_path
        self.logger = logger
        self.model_log_path = model_log_path
        self.cache_file, self.all_indi_file = cache_file, all_indi_file

    def train(self, indi: Individual, models_path:str):#q_indid_acc: Queue, 
        while True: # 每次GPU限制运行两个网络
            gpu_info = Popen('nvidia-smi', stdout = PIPE)
            gpu_info = gpu_info.stdout.read().decode('UTF-8')
            gpu_info = gpu_info.split(os.linesep)
            if len(gpu_info[18,-2]) > 1:
                time.sleep(60)
            else:
                break   
        
        mo_log = Recorder(f'Model_{indi.gen_id}_{indi.ind_id}', models_path, f'{indi.gen_id}_{indi.ind_id}').log()
        mo_log.info(f'PID:{os.getpid()}——Training individual:{indi}...')
        net = resnet._resnet(indi, self.cifar).to(self.device)
        mo_log.info(f'{indi.combination}')
        train_loader, valid_loader, test_loader = Data_loader.loader(self.cifar, self.data_dir, self.valid_size, self.batch_size, self.argument)
        trainer = Training(train_loader, valid_loader, test_loader, self.device, net, self.max_epoch, self.valid_test_epoch, self.loss_fun, self.save_path, mo_log) #self.writer, 
        try:
            indi.fitness = trainer.process()
            with open(self.all_indi_file, 'a') as f: # 记录所有个体，用于断点运行
                    f.write(str(indi)+f', {indi.fitness}'+os.linesep)
            with open(self.cache_file, 'a') as f: # 记录cache
                    f.write(str(indi).split('-')[-1]+f', {indi.fitness}'+os.linesep)
        except Exception as e:
            mo_log.info(f"'{e}' occurs! '{str(indi)}' cannot be trained! Please check!")
        else:
            # q_indid_acc.put([str(indi), indi.fitness])
            mo_log.info(f"The training of '{str(indi)}' done! Accuracy: {indi.fitness*100}%")

        # count = 1
        # while True:
        #     mo_log.info(f"{count}th try to train the '{str(indi)}'.")
        #     try:
        #         acc = trainer.process()
        #     except RuntimeError as e: #************************显存不够错误***********************
        #         mo_log.info(f"{e} occurs! '{str(indi)}' Sleep for 10s!")
        #         count+=1
        #         time.sleep(10)
        #         continue
        #     except Exception as e: #************************记录日志，继续运行************************
        #         mo_log.info(f"'{e}' occurs! '{str(indi)}' cannot be trained!")
        #         acc = -2
        #         break             
        #     else:
        #         q_indid_acc.put([str(indi), acc])
        #         mo_log.info(f"The training of '{str(indi)}' done! Accuracy: {acc*100}%")
        #         break



    def __decode(self, population: Population):
        self.logger.info(f"Decoding the population into CNNs...")
        models = []
        for i in  population:
            models.append([str(i), resnet._resnet(i, self.cifar)])
        self.logger.info(f"Decoding done!")
        return models

    def __call__(self, population: Population, check:bool = False): #多进程
        with open(self.cache_file, 'r') as f:
            cache = dict([[v.split(',')[0], float(v.split(',')[1].strip(os.linesep))] for v in f.readlines()])
        sub_pids = []
        # indid_accs = []
        # ind_accs = Queue() # 接受精度和个体id
        num_cache = 0 #记录存在于缓存中不用训练的个体个数
        if check:
             with open(self.all_indi_file, 'r') as f:
                res = [v.split(',')[0] for v in f.readlines()]
        for indi in population:
            key = str(indi).split('-')[-1]
            if key in cache:
                num_cache+=1
                indi.fitness = cache[key]
                with open(self.all_indi_file, 'a') as f: #如果是断点运行，则需要检查该个体是不是已经记录在all_indi_file。如果是，则无任何操作继续下一个个体，否则记录到all_indi_file里面。
                    if not check: #是不是断点运行
                        f.write(str(indi)+f', {indi.fitness}'+os.linesep)
                    else:
                        if str(indi) in res:
                            pass
                        else:
                            f.write(str(indi)+f', {indi.fitness}'+os.linesep)
                self.logger.info(f'No need to train individual:{indi.gen_id}_{indi.ind_id}, its acc ({indi.fitness}) is in cache.')
                continue
            p = Process(target=self.train, args=(indi, self.model_log_path))#indid_accs,
            p.start()
            sub_pids.append(p)
            self.logger.info(f'PID:{p.pid}——Training or waiting to train individual:{indi.gen_id}_{indi.ind_id}...')
            time.sleep(5)
        

        if not (len(sub_pids) == len(population)-num_cache):
             self.logger.info(f"The length of 'q_pids' and 'q_indid_accs' must equal the length of models (individuals)!")
             raise RuntimeError("The length of 'q_pids' and 'q_indid_accs' must equal the length of models (individuals)!")
             
        while True:
            # while not ind_accs.empty():
            #     re = ind_accs.get(False)
            #     self.logger.info(f'The training of "{re[0]}" is done! Acc(Fitness): {re[1]*100:5.2f}%.')
            #     indid_accs.append(re)
            sub_pids = [pc for pc in sub_pids if pc.is_alive()]
            if sub_pids:
               time.sleep(60) 
            else:
                with open(self.cache_file, 'a') as f:
                    for ind in population:
                        if str(ind) not in cache: # 只记录不重复（新）的个体
                            f.write(str(ind)+f', {ind.fitness}'+os.linesep)

                no_fit = []
                for indi in population:
                    if indi.fitness<=0:
                        no_fit.append(str(indi))
                if no_fit:
                    self.logger.info("The evaluation process is done! But there remains some individuals' fitnesses are None! Please check!")
                    raise ValueError("The evaluation process is done! But there remains some individuals' fitnesses are None! Please check!")
               
                # if not (len(indid_accs) == len(population)):
                #     self.logger.info("The training of all models is done, but the the length of array:'[indid, acc]' does not equal that of models(individuals)!")  
                #     raise RuntimeError("The training of all models is done, but the the length of array:'[indid, acc]' does not equal that of models(individuals)!") # 如果训练进程全部执行完毕，则此时的‘精度-个体ID’长度应该和models的长度一致结束所有训练
                # else:
                #     with open(catch_file, 'a') as f:
                #         for ind in population:
                            
                        # for re in indid_accs:
                        #     f.write(str(re)[1:-1]+os.linesep)
                break 
        # dict_acc_indids = dict(indid_accs) 
        # for ind in population:
        #     ind.fitness = dict_acc_indids[str(ind)]

if __name__ == '__main__': ...