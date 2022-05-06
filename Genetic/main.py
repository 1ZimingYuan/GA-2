import multiprocessing as mp
import time
def act1():
        st = time.time()
        while True:
            end = time.time()
            if end-st < 0.5:
                print('act2',end-st)
            else:
                break
def act2():
    st = time.time()
    while True:
        end = time.time()
        if end-st < 0.5:
            print('act1',end-st)
        else:
            break
if __name__ == '__main__':
    mp.set_start_method('spawn')
    # torch.multiprocessing.set_start_method('spawn')
    # config_file = r'/data/zmyuan/HDAM/HDAM/Genetic/Setting.ini'
    # configer = Configer(config_file) 
    # params = configer.get_params()
    # # print(params.keys())

    # log = Logger(params['DEFAULT']['log_name'], params['DEFAULT']['log_path'])
    # logger = log.log()
    
    # genetic = Genetic(int(params['GENETIC']['generation']), int(params['GENETIC']['resnet_type']), int(params['GENETIC']['popu_size']), float(params['GENETIC']['cross_prob']), float(params['GENETIC']['muta_prob']), params['TRAINING'], logger)
    
    p1 = mp.Process(target=act1,)
    p2 = mp.Process(target=act2,)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    # genetic.process_ind()