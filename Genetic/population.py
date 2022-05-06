""" 
This script records how to encode the receptive field into an individual.
"""
from hashlib import new
import numpy as np
from typing import Tuple
from numpy import random
from copy import deepcopy
import pickle
from logging import Logger
import os
from torch import combinations


#                    8, 8, 8               8, 4, 4, 4            4, 2, 2, 2, 2, 2       2, 1, 1, 
block_range = {50:{0:[1/8, 0, 1/4, 1/2], 1:[1/8, 0, 1/4, 1/2], 2:[1/4, 0, 1/2], 3:[1/2, 0]},
               101:{},
               152:{}}

block_type = {50:[3, 4, 6, 3],
              101:[],
              152:[]}

class Population:
    
    def __init__(self, gen_id: int, resnet_type:int, pop_size:int, logger:Logger) -> None:
        """ 
        gen_id (int): (当前)种群代数,从0开始计算
        resnet_type (int):resnet类型
        pop_size (int):种群大小
        logger:日志记录器
        """
        self.pop_size = pop_size
        self.individuals = []
        self.res_type = resnet_type
        self.start = 0
        self.logger = logger
        self.gen_id = gen_id
    
    def initialize(self,) -> None:#初始化初代种群
        self.logger.info('Initialize the first population...')
        for i in range(self.pop_size):
            indi = Individual(self.gen_id, i, self.res_type, self.logger)
            indi.initialize()
            self.individuals.append(indi)
        self.logger.info('First initialization of population termoinates!')
            
    
    def extend(self, lis: list or tuple) -> None:
        if not (isinstance(lis, list), isinstance(lis, tuple)):
            self.logger.info(f"TypeError: The object used to be extended must be the type 'list' or 'tuple' instead  of '{type(lis)}'!")
            raise TypeError(f"The object used to be extended must be the type 'list' or 'tuple' instead of '{type(lis)}'!")
        for i in lis:
            if not isinstance(i, Individual):
                self.logger.info(f"TypeError: The element type of the object used to be extended must be 'Individual' instead of '{type(i)}'!")
                raise TypeError(f"The element type of the object used to be extended must be 'Individual' instead of '{type(i)}'!")
        if len(self.individuals)+len(lis)>self.pop_size:
            self.logger.info(f"RuntimeError: The population size would exceed the limit after this 'extend' operation.")
            raise RuntimeError(f"The population size would exceed the limit after this 'extend' operation.")
        self.individuals.extend(lis)


    def __getitem__(self, key: int):
        return self.individuals[key]

    def __len__(self):
        return len(self.individuals)

    def __repr__(self) -> str:
        pass

    def create_from(self, last_pop:list):
        self.logger.info(f'Creating {self.gen_id}th population from the last population...')
        for i in range(last_pop):
            temp = []
            ind_id = int(i.split('-')[1]), 
            com = i.split('-')[-1].split('||')
            for k in com:
                tem = [float(j.strip(os.linesep)) for j in k.split(r'/')]
                temp.append(tem)
            indi = Individual(self.gen_id, ind_id, self.res_type, self.logger)
            indi.create_from(temp)
            self.individuals.append(indi)
        self.logger.info('Creating termoinates!')
    
    


class Individual:
    def __init__(self, gen_id: int, ind_id, res_type: int, logger) -> None:
        """ 
        gen_id (int): 表示当前个体属于第几代种群中的个体。
        ind_id (int):表示当前个体在该种群中的序号。
        res_type (int): 当前个体属于属于什么resnet类型。
        logger:日志记录器。
        """
        # assert len(blocks) == 4, 'The length of blocks should be 4!'
        assert res_type in [50, 101, 152], "The resnet type must be in [50, 101, 152]!"
        self.combination = []#[[0.1,0.2,0.3],[0.4,0.5,0.6,0.7],[0.2,0.3,0.1,0.4,0.6,0.5],[0.1,0.2,0.3]]
        self.res_type = res_type
        self.blocks = block_type[self.res_type] #different blocks indicates different resnet. [3, 4, 6, 3]
        self.fitness = -1
        self.logger = logger
        self.gen_id = gen_id
        self.ind_id = ind_id

    def initialize(self,) -> None:
        self.logger.info(f"***Initializing the individual '{self.ind_id}' of generation {self.gen_id} ...") 
        for k, v  in enumerate(self.blocks):
            tem = []
            for i in range(v):
                ind = np.random.randint(1 if i != 0 else 0, len(block_range[self.res_type][k]))
                tem.append(block_range[self.res_type][k][ind])
            self.combination.append(tem)
        self.logger.info(f"***Initialization of individual '{self}' terminates!")
    
    def create_from(self, com:list) -> None:
        self.combination = com

    def __getitem__(self, index):
        return self.combination[index]
    
    def __repr__(self) -> str:
        temp = []
        for i in self.combination:
            tem = [str(j) for j in i]
            temp.append('/'.join(tem))
        return str(self.gen_id)+'-'+str(self.ind_id)+'-'+'ResNet'+str(self.res_type)+'-'+'||'.join(temp)
    
    def __getstate__(self,):
        log = pickle.dumps(self.logger)
        combination = pickle.dumps(self.combination)
        blocks = pickle.dumps(self.blocks)
        state = {'gen_id': self.gen_id, 'ind_id':self.ind_id, 'res_type':self.res_type, 'combination':combination, 'fitness':self.fitness, 'blocks':blocks, 'logger':log}
        return state
    
    def __setstate__(self, state):
        self.combination = pickle.loads(state['combination']) 
        self.res_type = state['res_type']
        self.blocks = pickle.loads(state['blocks']) 
        self.fitness = state['fitness']
        self.logger = pickle.loads(state['logger']) 
        self.gen_id = state['gen_id']
        self.ind_id = state['ind_id']


class Crossover:
    def __init__(self, logger, cpr: int=0.8) -> None:
        """ 
        logger:日志记录器。
        cpr:交叉概率。
        """
        self.logger = logger
        self.cpr = cpr
    
    def crossover(self, parent1: Individual, parent2:Individual) -> Tuple[Individual]:
        off1, off2 = deepcopy(parent1), deepcopy(parent2) # gen_id 和 ind_id需要改变
        if random.random() < self.cpr:
            p1, p2 = [], []
            for i in parent1:
                p1.extend(i)
            for i in parent2:
                p2.extend(i)
            if not (len(p1) == len(p2)):
               self.logger.info('The length of p1 does not equal that of p2!')
               raise RuntimeError('The length of p1 does not equal that of p2!')
            cross_point = random.randint(1, len(p1))
            self.logger.info(f"Do crossover and the crossover point in parent1 and parent2 is {cross_point}.")
            temp = p1[cross_point:]
            p1[cross_point:] = p2[cross_point:]
            p2[cross_point:] = temp
            off1_, off2_ = [], []
            k = parent1.blocks
            t = 0
            for i in k:
                off1_.append(p1[t: i+t])
                off2_.append(p2[t: i+t])
                t+=i
            off1.combination = off1_
            off2.combination = off2_ ##check if the same individual exists

            off1.fitness = off2.fitness = -1 ##if not exist, reset the fitness
        else:
            self.logger.info(f"Do not crossover and parent1 and parent2 are copyied as children.")
            # off1, off2 = deepcopy(parent1), deepcopy(parent2) ##logging
        
        return off1, off2
    
    def binary(self, population: Population) -> Individual: #二进制锦标赛选择
        p1 = population[random.randint(len(population))]
        p2 = population[random.randint(len(population))]
        return p1 if p1.fitness>p2.fitness else p2        

    def __call__(self, gen_id: int, population: Population,) -> Population:
        new_population = Population(gen_id, population.res_type, population.pop_size, self.logger)
        self.logger.info(f"Begin the {gen_id}th crossover...")
        for i in range(len(population)//2):
            parent1 = self.binary(population)
            parent2 = self.binary(population)
            self.logger.info(f"Begin the {i+1}th crossover_operation and the individuals {parent1.gen_id}_{parent1.ind_id} and {parent2.gen_id}_{parent2.ind_id} are selected as the parents...")
            new_population.extend(self.crossover(parent1, parent2))
            self.logger.info(f"End the {i+1}th crossover_operation!")        
        self.logger.info(f"End the {gen_id}th crossover and {gen_id}th population before mutation is produced!")
        ind_start = 0 
        for indi in new_population: 
            indi.gen_id = gen_id
            indi.ind_id = ind_start
            ind_start+=1
        assert ind_start != len(new_population)-1, f'The ind_start({ind_start}) does not equal the length({len(new_population)}) of new population! '
        return new_population
    
    def __repr__(self) -> str:
        return 'Crossover'

class Mutation:
    def __init__(self, logger, mpr: int=0.2) -> None:
        self.logger = logger
        self.mpr = mpr
        
    def mutation(self, child: Individual) -> None:
        off_ = []
        for i in child.combination:
            off_.extend(i)
        if random.random() < self.mpr:
            mut_index = random.randint(len(off_))
            tem = child.blocks
            t, tt = [], 0
            for i in tem:
                t.append(tt+i)
                tt+=i
            k = list(range(sum(tem)))
            mut_type = off_[mut_index]
            while mut_type == off_[mut_index]:#确保变异后的个体和原个体不一样
                if mut_index in k[:t[0]]:
                    mut_type = block_range[child.res_type][0][random.randint(len(block_range[child.res_type][0]))]
                elif mut_index in k[t[0]:t[1]]:
                    mut_type = block_range[child.res_type][1][random.randint(len(block_range[child.res_type][1]))]
                elif mut_index in k[t[1]:t[2]]:
                    mut_type = block_range[child.res_type][2][random.randint(len(block_range[child.res_type][2]))]
                else:
                    mut_type = block_range[child.res_type][3][random.randint(len(block_range[child.res_type][3]))]

            self.logger.info(f'Implement mutation operator. Position: {mut_index}. {off_[mut_index]}-->{mut_type}.')
            off_[mut_index] = mut_type
        else:
            self.logger.info('Do not implement mutation operator.')
        off = []
        tt = 0
        for i in child.blocks:
            off.append(off_[tt:tt+i])
            tt+=i
        child.combination = off ##check if the same individual exists
        child.fitness = -1 ##if not exist, reset the fitness
    
    def __call__(self, population:Population) -> None:
        self.logger.info(f"Begin the {population.gen_id}th mutation...")
        for indi in population:
            self.mutation(indi)

    def __repr__(self) -> str:
        return 'Mutation'


if __name__ == "__main__":
    pass