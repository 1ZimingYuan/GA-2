from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchtoolbox.transform import Cutout
from torch import optim
from numpy import random
import numpy as np
import torch
from typing import Tuple
import time
import os
import sys




#数据读取与预处理
class Data_loader:
    @classmethod
    def loader(cls, cifar:int, data_dir:str, valid_size:float, batch_size:int, argument:bool, do_shuffle:bool=True, num_workers:int = 0, pin_memery:bool =  False) -> Tuple[DataLoader]:
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if cifar==10 else ([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        normalize = transforms.Normalize(mean, std) 
        train_transform = transforms.Compose([transforms.RandomCrop(32),
                                                  transforms.RandomHorizontalFlip(),
                                                  Cutout(),
                                                  transforms.ToTensor(),
                                                  normalize] if argument else [transforms.ToTensor(), normalize]) 
        valid_transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = datasets.CIFAR10(root=data_dir, train=True,
                                      transform = train_transform, download= False) if cifar==10 else datasets.CIFAR100(root=data_dir, train=True,
                                      transform = train_transform, download= False)
        
        test_data = datasets.CIFAR10(root=data_dir,train=False, 
                                     transform=valid_transform, download=False) if cifar==10 else datasets.CIFAR100(root=data_dir, train=False,
                                      transform = train_transform, download= False)

        index_list = list(range(len(train_data)))

        if do_shuffle:
            random.shuffle(index_list)
        
        valid = int(np.floor(valid_size*len(index_list)))
        train_index, valid_index = index_list[valid:], index_list[:valid]
        train_sampler, valid_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(valid_index)

        train_loader = DataLoader(train_data, 
                                  batch_size=batch_size,  
                                  sampler=train_sampler, 
                                  num_workers=num_workers,  
                                  pin_memory=pin_memery)

        valid_loader = DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  sampler=valid_sampler, 
                                  num_workers=num_workers,
                                  pin_memory=pin_memery)
        
        test_loader = DataLoader(test_data, 
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memery)
        
        return train_loader, valid_loader, test_loader
           

#训练模型
class Training:
    def __init__(self, train_loader, valid_loader, test_loader, device, net, max_epoch, valid_test_epoch, loss_fun, savepath, mo_log) -> None:#writer, 
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.device = device
        self.net = net#.to(device)
        self.max_epoch = max_epoch
        self.valid_test_epoch = valid_test_epoch
        self.loss = loss_fun
        # self.writer = writer
        self.path = savepath

        self.log = mo_log
    
    #训练模型
    def train(self, epoch):
        self.net.train()
        correct, total, loss_c = 0., 0., 0.
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            _, indices = torch.max(outputs, 1)
            correct+=(indices==labels).sum()
            total+=labels.size(0)
            loss = self.loss(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # loss_c+=loss.item()
        self.log.info(f'Epoch:{epoch+1}, Train acc:    {correct/total*100:5.2f}%')
        # self.writer.add_scalar('tra_acc', correct/total, epoch)
    
    #验证模型
    def valid(self, epoch):
        self.net.eval()
        with torch.no_grad():
            correct, total= 0., 0.
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, indices = torch.max(outputs, 1)
                correct+=(indices == labels).sum()
                total+=labels.size()[0]
                # loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                # loss.backward()
                self.optimizer.step()
            self.log.info(f'Epoch:{epoch+1}, Validate acc: {correct/total*100:5.2f}%')
            
            # self.writer.add_scalar('val_acc', correct/total, epoch)
    
    #测试和保存模型
    def test(self, epoch):
        self.net.eval()
        with torch.no_grad():
            correct, total = 0., 0.
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, indices = torch.max(outputs, 1)
                correct+=(indices == labels).sum()
                total+=labels.size()[0]
            self.log.info(f'Epoch:{epoch+1}, Test acc:     {correct/total*100:5.2f}%')
            
            # self.writer.add_scalar('test_acc', correct/total, epoch)

        if self.test_acc < correct/total:
            self.test_acc = correct/total
            for i in os.listdir(self.path):
                os.remove(os.path.join(self.path, i))
            
            check_point = {
                'epoch': epoch,
                'test_acc': self.test_acc*100,
                'model_state_dict': self.net.state_dict(),
                'optimize_state_dict': self.optimizer.state_dict(),
            }
            torch.save(check_point, os.path.join(self.path, f'test_{epoch}.tar'))


    
    #执行训练
    def process(self):
        start = time.time()
        self.test_acc = -1.0
        for epoch in range(self.max_epoch):
            if epoch == 0: lr =0.01
            if epoch > 0: lr = 0.1
            if epoch > 40 : lr = 0.01
            if epoch > 70 : lr = 0.001
            if epoch > 100: lr = 0.0008
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            self.train(epoch)
            if (epoch+1) % self.valid_test_epoch == 0 or epoch == self.valid_test_epoch - 1:
                self.valid(epoch)
                self.test(epoch)
            end =  time.time()
            # self.writer.close()
        return self.test_acc

if __name__ == '__main__':
    # from torchvision.models import resnet
    sys.path.append(r'/data/zmyuan/HDAM/HDAM')
    sys.path.append(r'/data/zmyuan/HDAM/HDAM/Genetic')
    from Genetic import resnet
    from Genetic.tools import Recorder
    from Genetic.population import Individual
    train_loader, val_loader, test_loader = Data_loader.loader(10, r'/data/datasets/cifar-10', 0.1, 256, True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_log = Recorder('123', r'/data/zmyuan/HDAM/HDAM/Genetic/Logfile/models_log', r'main').log()
    indi = Individual(0,0,50,main_log)
    indi.combination = [[0.25, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    net = resnet._resnet(indi, 10).to(device)#resnet.resnet50().to(device)
    loss = torch.nn.CrossEntropyLoss()
    log = Recorder('main', r'/data/zmyuan/HDAM/HDAM/Genetic/Logfile/models_log', r'model').log()
    trainer = Training(train_loader, val_loader, test_loader, device, net, 100, 1, loss, r'/data/zmyuan/HDAM/HDAM/Genetic/Logfile/save', log)
    acc = trainer.process()
    print(acc)
    pass