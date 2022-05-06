""" 
This script describes the whole HDAM process. 
"""
from importlib.resources import path
import population
from population import Individual, Population, Crossover, Mutation
from evaluation import Evaluate
from selection import Roulette
import multiprocessing as mp
import time
from tools import Configer, Recorder
from logging import Logger
import os
from configparser import ConfigParser
# sys.path.a.ppend('/data/zmyuan/HDAM')





class Genetic:
    def __init__(self, is_running: int, generation: int, resnet_type: int, popu_size: int, cross_pro: float, mutation_pro: float, train_setting: dict, logger:Logger, models_path:str, cache_file:str, all_indi_file:str, last_pop_file:str, conf_file:str) -> None:
        """
        generation (int):种群代数
        resnet_type (int):Resnet类型
        popu_size (int):种群大小
        cross_pro (float):交叉概率
        mutation_pro (float):变异概率
        train_setting (dict):训练设置
        """
        self.is_running = is_running
        self.__gen_num = generation
        self.__pop_size = popu_size
        self.__logger = logger
        self.train_setting = train_setting
        self.__evaluate = Evaluate(int(train_setting['cifar']), train_setting['data_dir'], float(train_setting['valid_size']), 
                                   int(train_setting['batch_size']), bool(train_setting['argument']), int(train_setting['max_epoch']), 
                                   int(train_setting['valid_test_epoch']), train_setting['save_path'], self.__logger, models_path, cache_file, all_indi_file)
        self.__corssover = Crossover(self.__logger, cross_pro)
        self.__mutation = Mutation(self.__logger, mutation_pro)
        self.__select = Roulette()
        self.__logger = logger 
        self.__resnet_type = resnet_type
        self.__last_pop_file = last_pop_file
        self.__conf_file = conf_file

    def save_last_pop(self, pop: Population) -> None:
        with open(self.__last_pop_file, 'w') as f:
            for ind in pop:
                f.write(str(ind)+os.linesep)

    def set_is_running(self) -> None:
        config = ConfigParser()
        config.read(self.__conf_file)
        config.set('RUNNING', 'is_running', '1')
        config.write(open(self.__conf_file, 'w'))

    def process(self,):
        try:
            self.__logger.info('Genetic process begins!')
            if not self.is_running:
                start = 1
                pop = Population(0, self.__resnet_type, self.__pop_size, self.__logger)
                pop.initialize()
                self.__logger.info(f"Evaluating the first initialized population.")
                self.save_last_pop(pop)
                self.__evaluate(pop)
            else:
                last_pop = []
                with open(self.__last_pop_file, 'r') as f:
                    last_pop = f.readlines()
                    gen = int(last_pop[0].split('-')[0])
                    start = gen+1
                    pop = Population(gen, self.__resnet_type, self.__pop_size, self.__logger)
                    pop.create_from(last_pop)
                self.__logger.info(f"Evaluating the {gen}th population created from 'last pop'.")
                self.save_last_pop(pop)
                self.__evaluate(pop, check=True)
            
            for gen_id in range(start, self.__gen_num+1): # gen_id从1开始，表示第几次进化，形成的第几代种群
                offspring = self.__corssover(gen_id, pop)
                self.__mutation(offspring)
                self.__evaluate(offspring)
                pop = self.__select(pop, offspring)
                self.save_last_pop(pop)
            self.__logger.info('Genetic process terminates!')

        except Exception as e:
            self.set_is_running()
            self.__logger.info(e)

if __name__ == '__main__': ...