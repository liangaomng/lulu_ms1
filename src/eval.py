
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
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


def plot_segx(args,fig_path,yaxis_data,test_label,test_predict):
    #yaxis_data = yy[:,30]
    #test_label = np.squeeze(test_label)[:,30]
    # test_predict = np.squeeze(test_predict.detach().cpu().numpy())[:,30]
    

    #绘制主图
    fig ,ax = plt.subplots()
    ax.plot(yaxis_data,test_predict,'*',label = 'Test-predict') 
    ax.plot(yaxis_data,test_label,'-',label = 'Test-true') 
    plt.xlabel('x',fontsize=16)
    plt.ylabel('u',fontsize=16)
    plt.legend()
    mse_loss = np.mean((test_predict-test_label)**2) 
    plt.title("{}-MSE={}".format(args.model,mse_loss)) 
    plt.tick_params(labelsize=16,width=2,colors='black')

    

    #嵌入局部放大图的坐标系
    # axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
    #            bbox_to_anchor=(0.3, 0.1, 1, 1), 
    #            bbox_transform=ax.transAxes)
    axins = ax.inset_axes((0.1, 0.05, 0.25, 0.1))

    axins.plot(yaxis_data,test_predict,'*') 
    axins.plot(yaxis_data,test_label,'-') 


    # 设置放大区间
    zone_left = 20
    zone_right = 120

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = yaxis_data[zone_left]-(yaxis_data[zone_right]-yaxis_data[zone_left])*x_ratio
    xlim1 = yaxis_data[zone_right]+(yaxis_data[zone_right]-yaxis_data[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((test_label[zone_left:zone_right], test_predict[zone_left:zone_right]))
    # ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    # # ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
    # ylim1 = ylim0 + (np.max(y)-np.min(y))*y_ratio
    ylim0 = - 0.55
    ylim1 = - 0.3


    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")

    # 画两条线
    xy = (xlim0,ylim0)
    xy2 = (xlim0,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
    axins.add_artist(con)

    xy = (xlim1,ylim0)
    xy2 = (xlim1,ylim1)
    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
    axins.add_artist(con)

    fig.savefig('{}/target_func_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
    plt.close(fig)



def eval(args,save_path):

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
    print(save_path)

    model.load_state_dict(torch.load('{}/model.pth'.format(save_path)))

    fig_path = './best_figure'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    if args.dim == 1:
        test_coordinate = np.linspace(args.x_min,args.x_max,1001)[:,np.newaxis].astype(np.float32)
        test_label = target_func(test_coordinate, args.mu)
        test_predict = model(torch.from_numpy(test_coordinate).to(device)).detach().cpu().numpy() #直接会对第一维进行broadcast      
        mse_loss = np.mean((test_predict-test_label)**2) 
        # fig ,ax = plt.subplots()
        # plt.plot(test_coordinate,test_predict.detach().cpu().numpy(),'*',label='predict') 
        # plt.plot(test_coordinate, test_label,'-',label='label')           
        # plt.xlabel('x',fontsize=16)
        # plt.ylabel('y',fontsize=16)
        # ax.get_xaxis().get_major_formatter().set_useOffset(False)
        # plt.tick_params(labelsize=16,width=2,colors='black')
        # plt.legend(fontsize=16)
        # plt.title("MSE={}".format(mse_loss))    
        # fig.savefig('{}/target_func_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
        # plt.close(fig)

        plot_segx(args,fig_path,test_coordinate,test_label,test_predict)
                

    
    if args.dim == 2: #边界采样点不同

        x = np.linspace(args.x_min,args.x_max,151).astype(np.float32)
        y = np.linspace(args.x_min,args.x_max,151).astype(np.float32)
        xx,yy = np.meshgrid(x,y)
        test_coordinate = np.concatenate((xx[...,np.newaxis],yy[...,np.newaxis]),axis=-1)
        test_label = target_func(test_coordinate, args.capital_n)

        test_predict = model(torch.from_numpy(test_coordinate).to(device)) #直接会对第一维进行broadcast      
        mse_loss = np.mean((test_predict.detach().cpu().numpy()-test_label)**2)
   
        
        
        fig ,ax = plt.subplots()
        plt.scatter(xx,yy,c=np.squeeze(test_predict.detach().cpu().numpy()),cmap="jet")  
        plt.xlabel('x',fontsize=16)
        plt.ylabel('y',fontsize=16)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.axis('equal')
        plt.title('predict_func_{}'.format(args.model))
        plt.tick_params(labelsize=16,width=2,colors='black')
        cbar=plt.colorbar()
        fig.savefig('{}/predict_func_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
        plt.close(fig)

        fig ,ax = plt.subplots()
        # plt.contourf(xx,yy,np.squeeze(test_predict.detach().cpu().numpy()-test_label.detach().cpu().numpy()),cmap="jet")   
        plt.scatter(xx,yy,c=np.squeeze(test_predict.detach().cpu().numpy()-test_label),cmap="jet") 
        plt.xlabel('x',fontsize=16)
        plt.ylabel('y',fontsize=16)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.axis('equal')
        plt.title('error_func_{}'.format(args.model))
        plt.tick_params(labelsize=16,width=2,colors='black')
        cbar=plt.colorbar()
        fig.savefig('{}/error_func_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
        plt.close(fig)

        fig ,ax = plt.subplots()
        # plt.contourf(xx,yy,np.squeeze(test_label.detach().cpu().numpy()),cmap="jet") 
        plt.scatter(xx,yy,c=np.squeeze(test_label),cmap="jet")  
        plt.xlabel('x',fontsize=16)
        plt.ylabel('y',fontsize=16)
        # ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.axis('equal')
        plt.tick_params(labelsize=16,width=2,colors='black')
        plt.title('anaytic_func_{}'.format(args.model))
        cbar=plt.colorbar()
        fig.savefig('{}/anaytic_func_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
        plt.close(fig)

        fig ,ax = plt.subplots()
        plt.plot(yy[:,30],np.squeeze(test_label)[:,30],label = 'Test-true') 
        plt.plot(yy[:,30],np.squeeze(test_predict.detach().cpu().numpy())[:,30],label = 'Test-predict') 
        plt.xlabel('y',fontsize=16)
        plt.ylabel('u',fontsize=16)
        plt.legend()
        # ax.get_xaxis().get_major_formatter().set_useOffset(False)
        # ax.axis('equal')
        plt.tick_params(labelsize=16,width=2,colors='black')
        plt.title('anaytic-seg-x_{}'.format(args.model))
        fig.savefig('{}/anaytic-seg-x_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
        plt.close(fig)

        fig ,ax = plt.subplots()
        plt.plot(xx[90,:],np.squeeze(test_label)[90,:],label = 'Test-true')  
        plt.plot(xx[90,:],np.squeeze(test_predict.detach().cpu().numpy())[90,:],label = 'Test-predict') 
        plt.xlabel('x',fontsize=16)
        plt.ylabel('u',fontsize=16)
        # ax.get_xaxis().get_major_formatter().set_useOffset(False)
        # ax.axis('equal')
        plt.legend()
        plt.tick_params(labelsize=16,width=2,colors='black')
        plt.title('anaytic-seg-y_{}'.format(args.model))
        fig.savefig('{}/anaytic-seg-y_{}.png'.format(fig_path,args.model), bbox_inches='tight',format='png', dpi=600)     
        plt.close(fig)
            
                
        
      

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pytorch distributed")

    parser.add_argument('--mu', default = 70)  
    parser.add_argument('--model', choices = ["mscalenn2" ,"mscalenn2-multi-alphai","mscalenn2-multi-alphai-ci","fnn"], default = 'mscalenn2-multi-alphai')     
    parser.add_argument('--activation', choices = ["phi" ,"ricker","sRELU"], default = 'phi')     
    parser.add_argument('--kernel_initializer', choices = ["Glorot-normal" ,"Glorot-uniform","He-normal","He-uniform"], default = 'Glorot-normal')     

    args = parser.parse_args()

    save_path = './result/{}-mu={}-{}-{}'.format(args.model,args.mu,args.activation,args.kernel_initializer)
     
    path = '{}/config.yaml'.format(save_path)
    
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.dim = config['sampling']['dim']
        args.x_min = config['sampling']['x_min']
        args.x_max = config['sampling']['x_max']

        args.batchsize = config['training']['batchsize']
        args.model = config['training']['model']
        args.sub_omegas = config['training']['sub_omegas']
        args.sub_layer_sizes = config['training']['sub_layer_sizes']
        args.activation = config['training']['activation']
        args.kernel_initializer = config['training']['kernel_initializer']

        
    print(args.sub_layer_sizes)
    print(args.sub_omegas)
    eval(args,save_path)
    