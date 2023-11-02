
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


def mixcoe(o_n=500,MixKnum=3):
    coe=np.ones([1,o_n])
    Unit_each=int(np.floor(o_n/MixKnum))
    for i_z in range(1,MixKnum-1):
        coe[0,Unit_each*i_z:Unit_each*(i_z+1)]=i_z
    coe[0,Unit_each*(MixKnum-1):]=MixKnum-1
    coe = torch.tensor(coe,dtype=torch.float32)
    return coe

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

    if args.model=="mscalenn2-multi-alphai":

        sub_omegas = args.sub_omegas
        model = MscaleNN2_multi_alphai(
                        sub_layer_sizes = args.sub_layer_sizes,
                        activation = args.activation,
                        kernel_initializer = args.kernel_initializer,
                        sub_omegas = sub_omegas,
                        )  

    if args.model=="mscalenn2-multi-alphai-ci":

        sub_omegas = args.sub_omegas
        model = MscaleNN2_multi_alphai_ci(
                        sub_layer_sizes = args.sub_layer_sizes,
                        activation = args.activation,
                        kernel_initializer = args.kernel_initializer,
                        sub_omegas = sub_omegas,
                        )  
        
    # initialize your model 
    if args.model=="mscalenn":
        first_omega = torch.tensor(150,dtype=torch.float32).to(device)
        hidden_omega = torch.tensor(1,dtype=torch.float32).to(device)
        model = MscaleNN(
                        layer_sizes = args.layer_sizes,
                        activation = args.activation,
                        kernel_initializer = args.kernel_initializer,
                        first_omega = first_omega,
                        hidden_omega = hidden_omega,
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
    # if args.schedu == "StepLR":
    #     scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)#step 0.1下降太小了且也震荡,反正就是没啥用;0.2也没啥用。所以可能就是P.E.的问题？

    n_epoch = args.n_epoch

    loss_list = []
    lr_list = []
    mse_list = []

    test_coordinate = np.linspace(args.x_min,args.x_max,501)[:,np.newaxis].astype(np.float32)
    test_label = target_func(test_coordinate, args.mu)

    # #load data
    # if args.ifrandomsample == "False":
    #     print("ifrandomsample:",args.ifrandomsample)
    #     print('Start generating data!')
    #     os.system('python -m src.data_gen --case_num={}'.format(args.case_num))
    #     print('Over!')

    #     data_path = "./data/case{}".format(args.case_num)
    #     train_coordinates = np.load(data_path + '/train_coordinates.npy').astype(np.float32)
    #     train_label = np.load(data_path + '/train_label.npy').astype(np.float32)

    #     train_dataset= MyDataset(train_coordinates,train_label)
        
    #     train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                             batch_size=args.batchsize,
    #                                             shuffle=True)

    #     for k in range(n_epoch):

    #         model.train() 

    #         loss_train = 0
    #         count_num = 0  

    #         for _,(coordinate,label) in enumerate(train_loader):

                
    #             predict = model(coordinate.to(device)) #直接会对第一维进行broadcast 
                
    #             label = label.to(device)
                
    #             diff = predict - label
            
    #             loss = torch.mean(torch.square(diff))
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             loss_train += loss.item() * diff.size(0)
    #             count_num += diff.size(0)  

    #             test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast      
    #             mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2) 
                
    #         # scheduler.step()
    #         loss_list.append(loss_train/count_num)
    #         # lr_list.append(scheduler.get_last_lr()[0])
    #         mse_list.append(mse_loss)

    #         if k % 1000 == 0:

    #             test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast      
    #             mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2)

    #             fig ,ax = plt.subplots()
    #             plt.plot(test_coordinate,test_predict.detach().cpu().numpy(),'*',label='predict') 
    #             plt.plot(test_coordinate, test_label,'-',label='label')           
    #             plt.xlabel('x',fontsize=16)
    #             plt.ylabel('y',fontsize=16)
    #             ax.get_xaxis().get_major_formatter().set_useOffset(False)
    #             plt.tick_params(labelsize=16,width=2,colors='black')
    #             plt.legend(fontsize=16)
    #             plt.title("MSE={}".format(mse_loss))    
    #             fig.savefig('{}/target_func_epoch{}.png'.format(save_path,k), bbox_inches='tight',format='png', dpi=600)     
    #             plt.close(fig)

    #             plt.semilogy(loss_list,label='train')
    #             plt.semilogy(mse_list,label='MSE')
    #             plt.title('loss')
    #             plt.legend(fontsize=16)
    #             plt.savefig(save_path + '/loss.png')
    #             plt.close()
    #             np.save(save_path + '/loss.npy', loss_list)
    #             np.save(save_path + '/mse.npy', mse_list)

    #             plt.semilogy(lr_list)
    #             plt.title('lr')
    #             plt.savefig(save_path + '/lr.png')
    #             plt.close()

    #             np.save(save_path + '/lr.npy', lr_list)

    #             if args.model=="mscalenn2-multi-alphai-ci":
    #                 np.save(save_path + '/ci.npy', model.c_s.detach().cpu().numpy())

                

    #             if k == 0:
    #                 initial_loss = loss_list[-1]
    #                 best_loss = initial_loss
    #                 torch.save(model.state_dict(),  './{}/model_epoch{}.pth'.format(save_path, k))
    #             else:
    #                 new_loss = loss_list[-1]
    #                 if new_loss < best_loss:
    #                     best_loss = new_loss
    #                     torch.save(model.state_dict(), './{}/model_epoch{}.pth'.format(save_path, k))

    if args.ifrandomsample == "True":
        print("ifrandomsample:",args.ifrandomsample)

        print(n_epoch)

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
                
            # scheduler.step()
            loss_list.append(loss_train/count_num)
            # lr_list.append(scheduler.get_last_lr()[0])
            mse_list.append(mse_loss)

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
                test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast      
                
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
                # fig.savefig('{}/target_func.png'.format(save_path), bbox_inches='tight',format='png', dpi=600)     
                # plt.close(fig)


                
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

                

            

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pytorch distributed")

    parser.add_argument('--dim', type = int, default = 1) 
    parser.add_argument('--method', choices = ['uniform', 'linspace'], default = 'uniform') 
    parser.add_argument('--num', type = int, default = 5000) 
    parser.add_argument('--mu', type = float, default = 70) 
    parser.add_argument('--x_min', type = float, default = -1) 
    parser.add_argument('--x_max', type = float, default = 1) 
    
    
    parser.add_argument('--ifrandomsample', choices = ['True', 'False'], default = 'True') 
    parser.add_argument('--batchsize', type = int, default = 5000) 
    parser.add_argument('--n_epoch', type = int, default = 1) 
    parser.add_argument('--lr', type = float, default = 1.e-3) 
    parser.add_argument('--optim', choices = ['Adam'], default = 'Adam') 
    # parser.add_argument('--schedu', choices = ['StepLR'], default = 'StepLR') 
    # parser.add_argument('--step_size', type = int, default = 200) 
    # parser.add_argument('--gamma', type = float, default = 0.7)  
    parser.add_argument('--sub_omegas', default = [1,2,4,8,16,32]) 
    parser.add_argument('--sub_layer_sizes', default = [1,900,900,900,1])
    
    parser.add_argument('--model', choices = ["mscalenn2" ,"mscalenn2-multi-alphai","mscalenn2-multi-alphai-ci","fnn"], default = 'fnn')     
    parser.add_argument('--activation', choices = ["phi" ,"ricker","sRELU"], default = 'phi')     
    parser.add_argument('--kernel_initializer', choices = ["Glorot-normal" ,"Glorot-uniform","He-normal","He-uniform"], default = 'He-normal')     

    args = parser.parse_args()

    # loca=time.strftime('%Y-%m-%d %H:%M:%S')
    save_path = './result/{}-mu={}-{}-{}'.format(args.model,args.mu,args.activation,args.kernel_initializer)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.abspath(__file__)
    shutil.copy(filename,save_path) 
    shutil.copy("./src/model_utils.py",save_path)     

    path = '{}/config.yaml'.format(save_path)
    
    # try:
    #     with open(path, 'r') as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)
    #         args.dim = config['sampling']['dim']
    #         args.method = config['sampling']['method']
    #         args.num = config['sampling']['num']
    #         args.mu = config['sampling']['mu']
    #         args.x_min = config['sampling']['x_min']
    #         args.x_max = config['sampling']['x_max']

    #         args.ifrandomsample = config['training']['ifrandomsample']
    #         args.batchsize = config['training']['batchsize']
    #         args.model = config['training']['model']
    #         args.lr = config['training']['lr']
    #         args.optim = config['training']['optim']
    #         # args.schedu = config['training']['schedu']
    #         # args.step_size = config['training']['step_size']
    #         # args.gamma = config['training']['gamma']
    #         args.n_epoch = config['training']['n_epoch']
    #         args.sub_omegas = config['training']['sub_omegas']
    #         args.sub_layer_sizes = config['training']['sub_layer_sizes']
    #         args.activation = config['training']['activation']
    #         args.kernel_initializer = config['training']['kernel_initializer']
    #         # args.first_omega = config['training']['first_omega']
    #         # args.hidden_omega = config['training']['hidden_omega']


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

    # print(args.model)
    # print(args.activation)
    # print(args.kernel_initializer)
    train(args,save_path)
    