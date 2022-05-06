import logging
from datetime import datetime
import configparser

#配置文件
class Configer:
    def __init__(self, path: str, section: str= 'DEFAULT') -> None:
        self.conf_file_path = path
        self.section = section
        
    def get_params(self, ) -> dict:
        config = configparser.ConfigParser()
        config.read(self.conf_file_path)
        params = {}
        for section in config.sections():
            params[section] = dict(config[section])
        params['DEFAULT'] = config.defaults()
        return params

#日志
class Recorder:
    def __init__(self, name, path, ind = None) -> None:
        self.name = name
        self.path = path
        self.time = datetime.now().strftime(r'%m-%d-%H-%M-%S')
        self.ind = ind

    def log(self,):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        file_log_path = self.path+r'/' + (self.time if not self.ind else self.ind) +'.log'
        filehandler = logging.FileHandler(file_log_path)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)

        if not self.ind:
            screamhandler = logging.StreamHandler()
            screamhandler.setLevel(logging.INFO)
            screamhandler.setFormatter(formatter)
            logger.addHandler(screamhandler)

        
        logger.addHandler(filehandler)

        return logger

    def __getstate__(self,):
        state = {'name': self.name, 'path': self.path, 'time': self.time, 'ind':self.ind}
        return state
    
    def __setstate__(self, state):
        self.name = state['name']
        self.path = state['path']
        self.time = state['time']
        self.ind = state['ind']  