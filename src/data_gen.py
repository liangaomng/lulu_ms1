import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import yaml

parser = argparse.ArgumentParser(description="config")

parser.add_argument('--case_num', type = int, default = 1) #选不同的配置参数训练

args = parser.parse_args()

#read config from path
path = './config/config_{}.yaml'.format(args.case_num)

with open(path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    args.dim =  config['sampling']['dim']
    args.method = config['sampling']['method']
    args.num = config['sampling']['num']
    args.mu =  config['sampling']['mu']
    args.x_min =  config['sampling']['x_min']
    args.x_max =  config['sampling']['x_max']
    
                  
def target_func(x, mu ):

    out = np.sum(np.exp(-x**2)*np.sin(mu*x**2),axis=-1,keepdims=True)

    return out

# x = np.linspace(0,1,11)
x_min = np.array([args.x_min]*args.dim)
# print(x_min)
x_max = np.array([args.x_max]*args.dim)
if args.method == "uniform":
    coordinates = np.random.uniform(x_min,x_max,[args.num+1,args.dim])
    # print(coordinates.shape)
if args.method == "linspace":
    if args.dim == 1:
        coordinates = np.linspace(args.x_min,args.x_max,args.num+1)[:,np.newaxis] #,endpoint=False
    if args.dim == 2:
        coordinates_x = np.linspace(args.x_min,args.x_max,args.num+1)
        coordinates_y = np.linspace(args.x_min,args.x_max,args.num+1)
        xx,yy = np.meshgrid(coordinates_x,coordinates_y)
        coordinates = np.concatenate((xx,yy),axis=-1).reshape(-1,2)


outputs = target_func(coordinates,mu =args.mu)
print("target_output_shape",outputs.shape)


def plot1d(fig_name,coordinates,outputs):

    fig ,ax = plt.subplots()
    plt.plot(coordinates, outputs,'o',label='train data') 

    x = np.linspace(args.x_min,args.x_max,5001)[:,np.newaxis]
    y = target_func(x,mu =args.mu)
    plt.plot(x, y,'-',label='target function') 
        
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.tick_params(labelsize=16,width=2,colors='black')
    plt.legend(fontsize=16)
    
    fig.savefig('%s/target_func.png'%fig_name, bbox_inches='tight',format='png', dpi=600)
    
    plt.close(fig)
    
def plot2d(fig_name,coordinates,outputs):

    fig ,ax = plt.subplots()
    x = np.linspace(args.x_min,args.x_max,101)
    y = np.linspace(args.x_min,args.x_max,101)
    xx,yy = np.meshgrid(x,y)
    inputs = np.concatenate((xx[...,np.newaxis],yy[...,np.newaxis]),axis=-1)
    print(inputs.shape)
    u = np.squeeze(target_func(inputs,mu =args.mu))
    plt.contourf(xx,yy,u,cmap="jet")
    plt.scatter(coordinates[:,0],coordinates[:,1],s=100,c='g',marker='o',label='train data') 
        
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.axis('equal')
    plt.tick_params(labelsize=16,width=2,colors='black')
    plt.legend(fontsize=16)
    
    fig.savefig('%s/target_func.png'%fig_name, bbox_inches='tight',format='png', dpi=600)
    
    plt.close(fig)
    

# train_data = np.concatenate((coordinates,outputs),axis=-1)

save_path = "./data/case{}".format(args.case_num)
print("before save")
if not os.path.exists(save_path):
    os.makedirs(save_path)
if args.dim==1:
    plot1d(save_path,coordinates,outputs)
if args.dim==2:
    plot2d(save_path,coordinates,outputs)
np.save(save_path + '/train_coordinates.npy',coordinates)
np.save(save_path + '/train_label.npy',outputs)
