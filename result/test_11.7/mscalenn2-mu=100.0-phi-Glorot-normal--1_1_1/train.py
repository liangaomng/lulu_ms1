
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from .model_utils import *
# from .data_gen import target_func
import shutil
torch.set_default_dtype(torch.float32)

# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 deeponet_FNN_siren_GD_ddp.py

                 
def target_func(x, mu ):

    out = np.sum(np.exp(-x**2)*np.sin(mu*x**2),axis=-1,keepdims=True)

    return out




class MyDataset(torch.utils.data.Dataset):

    #父类是torch.utils.data.Dataset，也可以是object，对父类没有要求
        def __init__(self,coordinate,label):
            self.coordinate=coordinate
            self.label=label
        def __getitem__(self,index):#迭代数据
            coordinate=self.coordinate[index]
            label=self.label[index]
            return coordinate,label
        def __len__(self):#返回数据的总长度
            return len(self.coordinate)


def train(args,save_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model=="mscalenn2":

        sub_omegas = args.sub_omegas
        model = MscaleNN2(
                        sub_layer_sizes = args.sub_layer_sizes,
                        activation = args.activation,
                        kernel_initializer = args.kernel_initializer,
                        sub_omegas = sub_omegas,
                        )


    if args.model=="fnn":
        model = FNN(
                    layer_sizes = args.sub_layer_sizes,
                    activation = args.activation,
                    kernel_initializer = args.kernel_initializer,
                    )
    
    # send your model to GPU 
    model = model.to(device)
    lr = args.lr

    if args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = lr)

    n_epoch = args.n_epoch

    loss_list = []
    lr_list = []
    mse_list = []

    test_coordinate = np.linspace(args.x_min,args.x_max,501)[:,np.newaxis].astype(np.float32)
    test_label = target_func(test_coordinate, args.mu)


    if args.ifrandomsample == "False":
        print("ifrandomsample:",args.ifrandomsample)
        print('Start generating data!',flush=True)
        print("case",args.case_num)
        os.system('python -m src.data_gen --case_num={}'.format(args.case_num))
        print('Over!')

        data_path = "./data/case{}".format(args.case_num)

        train_coordinates = np.load(data_path + '/train_coordinates.npy').astype(np.float32)
        train_label = np.load(data_path + '/train_label.npy').astype(np.float32)

        train_dataset= MyDataset(train_coordinates,train_label)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batchsize,
                                                shuffle=True)

        for k in range(n_epoch):

            model.train()

            loss_train = 0
            count_num = 0

            for _,(coordinate,label) in enumerate(train_loader):


                predict = model(coordinate.to(device)) #直接会对第一维进行broadcast
                
                label = label.to(device)
                
                diff = predict - label
            
                loss = torch.mean(torch.square(diff))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * diff.size(0)
                count_num += diff.size(0)

                test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast
                mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2)
                
            # scheduler.step()
            loss_list.append(loss_train/count_num)
            # lr_list.append(scheduler.get_last_lr()[0])
            mse_list.append(mse_loss)
            print("traing epoch: {} , loss: {} , mse: {}".format(k,loss_train/count_num,mse_loss))

            if k % 1000 == 0:

                test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast
                mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2)

                fig ,ax = plt.subplots()
                plt.plot(test_coordinate,test_predict.detach().cpu().numpy(),'*',label='predict')
                plt.plot(test_coordinate, test_label,'-',label='label')
                plt.xlabel('x',fontsize=16)
                plt.ylabel('y',fontsize=16)
                plt.tight_layout()
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                plt.tick_params(labelsize=16,width=2,colors='black')
                plt.legend(fontsize=16)
                plt.title("MSE={}".format(mse_loss))
                fig.savefig('{}/target_func_epoch{}.png'.format(save_path,k), bbox_inches='tight',format='png', dpi=600)
                plt.close(fig)

                plt.semilogy(loss_list,label='train')
                plt.semilogy(mse_list,label='MSE')
                plt.title('loss')
                plt.legend(fontsize=16)
                plt.savefig(save_path + '/loss.png')
                plt.close()
                np.save(save_path + '/loss.npy', loss_list)
                np.save(save_path + '/mse.npy', mse_list)

                plt.semilogy(lr_list)
                plt.title('lr')
                plt.savefig(save_path + '/lr.png')
                plt.close()

                np.save(save_path + '/lr.npy', lr_list)

                if k == 0:
                    initial_loss = loss_list[-1]
                    best_loss = initial_loss
                    torch.save(model.state_dict(),  './{}/model_epoch{}.pth'.format(save_path, k))
                else:
                    new_loss = loss_list[-1]
                    if new_loss < best_loss:
                        best_loss = new_loss
                        torch.save(model.state_dict(), './{}/model_epoch{}.pth'.format(save_path, k))

    if args.ifrandomsample == "True":
        print("ifrandomsample:",args.ifrandomsample)

        for k in range(n_epoch):

            model.train() 

            loss_train = 0
            count_num = 0  

            for batch_i in range(1):

                x_min = np.array([args.x_min]*args.dim)
                x_max = np.array([args.x_max]*args.dim)
                
                coordinate = np.random.uniform(x_min,x_max,[args.num,args.dim])
                label = target_func(coordinate,mu =args.mu)                
                
                predict = model(torch.tensor(coordinate,dtype=torch.float32).to(device)) #直接会对第一维进行broadcast 

                diff = predict - torch.tensor(label,dtype=torch.float32).to(device)
            
                loss = torch.mean(torch.square(diff))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * diff.size(0)
                count_num += diff.size(0)  

                test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast      
                mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2) 

            loss_list.append(loss_train/count_num)

            mse_list.append(mse_loss)

            print("epoch: {} , loss: {} , mse: {}".format(k,loss_train/count_num,mse_loss))

            # if k % 1000 == 0:
            if k == 0:
                initial_loss = loss_list[-1]
                best_loss = initial_loss
                torch.save(model.state_dict(),  './{}/model.pth'.format(save_path))
            else:
                new_loss = loss_list[-1]
                if new_loss < best_loss:
                    best_loss = new_loss
                    torch.save(model.state_dict(), './{}/model.pth'.format(save_path))
                

                
        plt.semilogy(loss_list,label='train')
        plt.semilogy(mse_list,label='MSE')
        plt.title('loss')
        plt.legend(fontsize=16)
        plt.savefig(save_path + '/loss.png')
        plt.close()
        np.save(save_path + '/loss.npy', loss_list)
        np.save(save_path + '/mse.npy', mse_list)

        plt.semilogy(lr_list)
        plt.title('lr')
        plt.savefig(save_path + '/lr.png')
        plt.close()

        np.save(save_path + '/lr.npy', lr_list)

        if args.model=="mscalenn2-multi-alphai-ci":
            np.save(save_path + '/ci.npy', model.c_s.detach().cpu().numpy())

                

            # test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast      
                
                # mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2)

                # fig ,ax = plt.subplots()
                # plt.plot(test_coordinate,test_predict.detach().cpu().numpy(),'*',label='predict') 
                # plt.plot(test_coordinate, test_label,'-',label='label')           
                # plt.xlabel('x',fontsize=16)
                # plt.ylabel('y',fontsize=16)
                # ax.get_xaxis().get_major_formatter().set_useOffset(False)
                # plt.tick_params(labelsize=16,width=2,colors='black')
                # plt.legend(fontsize=16)
                # plt.title("MSE={}".format(mse_loss))    
                # fig.savefig('{}/target_func_epoch{}.png'.format(save_path,k), bbox_inches='tight',format='png', dpi=600)     
                # plt.close(fig)

import ast
import re
def redu_subnets4name(input:list)->str:#简化记录的name
    '''
    Args:
        input_str: [1,..3]
        reduce the name 4 "1_3_1" means :start 1 end3 step=1
    Returns:"1_3_1"

    '''
    start, end = input[0], input[-1]

    # 计算步长，如果序列长度大于1，则通过相邻元素的差值计算步长
    step = input[1] - input[0] if len(input) > 1 else 1

    # 生成并返回格式化的字符串
    return f"{start}_{end}_{step}"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pytorch distributed")

    parser.add_argument('--dim', type = int, default = 1) 
    parser.add_argument('--method', choices = ['uniform', 'linspace'], default = 'uniform') 
    parser.add_argument('--num', type = int, default = 5000) 
    parser.add_argument('--mu', type = float, default = 70) 
    parser.add_argument('--x_min', type = float, default = -1) 
    parser.add_argument('--x_max', type = float, default = 1)
    
    parser.add_argument('--ifrandomsample', choices = ['True', 'False'], default = 'False')
    parser.add_argument('--batchsize', type = int, default = 5000)
    parser.add_argument('--case_num', type = int, default = 0)
    parser.add_argument('--n_epoch', type = int, default = 1) 
    parser.add_argument('--lr', type = float, default = 1.e-3) 
    parser.add_argument('--optim', choices = ['Adam'], default = 'Adam') 

    parser.add_argument('--sub_omegas',default =  "[1,2,4,8,16,32]",help='list str of sub_omegas')
    parser.add_argument('--sub_layer_sizes', default = [1,150,150,150,1])
    
    parser.add_argument('--model', choices = ["mscalenn2" ,"mscalenn2-multi-alphai","mscalenn2-multi-alphai-ci","fnn"], default = 'mscalenn2')     
    parser.add_argument('--activation', choices = ["phi" ,"ricker","sRELU"], default = 'phi')     
    parser.add_argument('--kernel_initializer', choices = ["Glorot-normal" ,"Glorot-uniform","He-normal","He-uniform"], default = 'He-normal')     

    # str to list
    args = parser.parse_args()
    args.sub_omegas = ast.literal_eval(args.sub_omegas)#4list

    print("redu_name",redu_subnets4name(args.sub_omegas))

    # loca=time.strftime('%Y-%m-%d %H:%M:%S')
    save_path = './result/{}-mu={}-{}-{}--{}'.format(args.model,
                                                     args.mu,
                                                     args.activation,
                                                     args.kernel_initializer,
                                                     redu_subnets4name(args.sub_omegas))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.abspath(__file__)
    shutil.copy(filename,save_path) 
    shutil.copy("./src/model_utils.py",save_path)

    path = '{}/config_{}.yaml'.format(save_path,args.case_num)

    # except FileNotFoundError:
    with open(path, 'a') as f:
        config = {
            'sampling': {
                'dim': args.dim,
                'method': args.method,
                'num': args.num,
                'mu': args.mu,
                'x_min': args.x_min,
                'x_max': args.x_max,
            },
            'training': {        
                'ifrandomsample': args.ifrandomsample,
                'batchsize': args.batchsize,
                'model': args.model,
                'lr': args.lr,
                'optim': args.optim,
                'n_epoch': args.n_epoch,
                'sub_omegas': args.sub_omegas,
                'sub_layer_sizes': args.sub_layer_sizes,
                'activation': args.activation,
                'kernel_initializer': args.kernel_initializer
            }
        }
        f.write(yaml.dump(config, allow_unicode = True))

    #copy the config.yaml to the config folder
    shutil.copy(path,'./config/')
    print("moved config.yaml to the config folder")

    train(args,save_path)
    