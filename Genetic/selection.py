from population import Population
from numpy import random 

#Q:line23-27 根据适应度值进行轮盘赌判断是不是不太适宜，因为他们的适应度值基本上都是一样的。

class Roulette:
    

    def __call__(parent: Population, off: Population): #environment selection
        assert len(parent) == len(off), 'The population size of parent and offspring are be the same!'

        all_popu = [] #Population()
        all_popu.extend(parent)
        all_popu.extend(off)
        sum_fit = 0
        for i in all_popu:
            sum_fit += i.fitness

        new_off = []
        for _ in range(len(parent)//2):
            rand_digit = random.random_sample() * sum_fit
            tem = 0
            for j in all_popu:
                tem += j.fitness
                if tem>rand_digit:
                    new_off.append(j)
                    break
        new_popu = Population(len(parent))
        new_popu.individuals = new_off 
        return new_popu