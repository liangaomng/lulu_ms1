import os
import numpy as np
import shutil

import matplotlib.pyplot as plt


import matplotlib.colors as mcolors
print(mcolors.CSS4_COLORS['blue'])

#colors = [\'#1f77b4\', \'#ff7f0e\', \'#2ca02c\', \'#d62728\', \'#9467bd\',   # 使用颜色编码定义颜色
        #   \'#8c564b\', \'#e377c2\', \'#7f7f7f\', \'#bcbd22\', \'#17becf\']
# dirname = '/Users/科研资料/Helmholtz方程/多尺度神经网络的研究/updates on MscaleDNN/所有选中实验/ex1.fitting-ok'

dirname = '/Users/liangaoming/Desktop/neural_study/lulu_ms/result'

save_path = dirname + '/compare_fig'
if not os.path.exists(save_path):
    os.makedirs(save_path)
mu = 70

for activation in ["phi" ]:

    for kernel_initializer in ["Glorot-normal" ]:
            
        # for model in ["mscalenn2" ,"mscalenn2-multi-alphai","mscalenn2-multi-alphai-ci"]:
        loss_fnn = np.load('result/fnn-mu={}-{}-{}/loss.npy'.format(mu,activation,kernel_initializer))

        loss_ms2 = np.load('result/mscalenn2-mu={}-{}-{}/loss.npy'.format(mu,activation,kernel_initializer))
    
        loss_ms2_alpha = np.load('result/mscalenn2-multi-alphai-mu={}-{}-{}/loss.npy'.format(mu,activation,kernel_initializer))
        
        loss_ms2_alpha_c = np.load('result/mscalenn2-multi-alphai-ci-mu={}-{}-{}/loss.npy'.format(mu,activation,kernel_initializer))


        length = min(len(loss_fnn),len(loss_ms2),len(loss_ms2_alpha),len(loss_ms2_alpha_c))

        fig ,ax = plt.subplots(figsize=(6, 4))
        plt.semilogy(loss_fnn[0:length],label='fnn',color='#1f77b4')
        
        plt.semilogy(loss_ms2[0:length],label='ms2',color='#ff7f0e')
        plt.semilogy(loss_ms2_alpha[0:length],label='ms2_' + r'$\alpha$',color='#2ca02c')
        
        # plt.semilogy(loss_ms2_alpha_c[0:length],label='ms2_'+ r'$\alpha$'+'_c',color='r')
        # plt.semilogy(loss_ms2_alpha_rescale[0:length],label='ms2_' + r'$\alpha$' + '_rescale' )
        # plt.semilogy(loss_ms2_alpha_c_rescale[0:length],label='ms2_' + r'$\alpha$' + '_c_rescale')
        plt.title('loss-{}-{}'.format(activation,kernel_initializer),fontsize=15)
        plt.legend(fontsize=15)
        plt.yticks(size=15)#设置大小及加粗 #weight='bold' #fontproperties='Times New Roman',
        plt.xticks(size=15)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8 , box.height]) #图例放上面就box.height*0.8，放右边就box.width*0.8其它方式一样
        ax.legend(loc='center left', bbox_to_anchor=(0.95,0.8),ncol=1)
        plt.savefig(save_path+ '/loss-{}-{}.png'.format(activation,kernel_initializer))
        plt.close(fig)


        # mse_fnn = np.load('result/fnn-mu={}-{}-{}/mse.npy'.format(mu,activation,kernel_initializer))

        mse_ms2 = np.load('result/mscalenn2-mu={}-{}-{}/mse.npy'.format(mu,activation,kernel_initializer))
    
        # mse_ms2_alpha = np.load('result/mscalenn2-multi-alphai-mu={}-{}-{}/mse.npy'.format(mu,activation,kernel_initializer))
        #
        # mse_ms2_alpha_c = np.load('result/mscalenn2-multi-alphai-ci-mu={}-{}-{}/mse.npy'.format(mu,activation,kernel_initializer))

        # mse_ms2_alpha_rescale = np.load('./mscalenn2-multi-alphai-mu={}-{}-{}-rescaleout/mse.npy'.format(mu,activation,kernel_initializer))
        
        # mse_ms2_alpha_c_rescale = np.load('./mscalenn2-multi-alphai-ci-mu={}-{}-{}-rescaleout/mse.npy'.format(mu,activation,kernel_initializer))


        #length = min(len(mse_fnn),len(mse_ms2),len(mse_ms2_alpha),len(mse_ms2_alpha_c))
        length=4000
        fig ,ax = plt.subplots(figsize=(6, 4))
        #plt.semilogy(mse_fnn[0:length],label='fnn',color='#1f77b4')
        
        plt.semilogy(mse_ms2[0:length],label='ms2',color='#ff7f0e')
        #plt.semilogy(mse_ms2_alpha[0:length],label='ms2_' + r'$\alpha$',color='#2ca02c')
        
        # plt.semilogy(mse_ms2_alpha_c[0:length],label='ms2_'+ r'$\alpha$'+'_c',color='r')
        # plt.semilogy(mse_ms2_alpha_rescale[0:length],label='ms2_' + r'$\alpha$' + '_rescale' )
        # plt.semilogy(mse_ms2_alpha_c_rescale[0:length],label='ms2_' + r'$\alpha$' + '_c_rescale')
        plt.title('mse-{}-{}'.format(activation,kernel_initializer),fontsize=15)
        plt.legend(fontsize=15)
        plt.yticks( size=15)#设置大小及加粗 #weight='bold'
        plt.xticks( size=15)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8 , box.height]) #图例放上面就box.height*0.8，放右边就box.width*0.8其它方式一样
        ax.legend(loc='center left', bbox_to_anchor=(0.95,0.8),ncol=1)
        plt.savefig(save_path+ '/mse-{}-{}.png'.format(activation,kernel_initializer))
        plt.close(fig)



        # c = np.load('./result/mscalenn2-multi-alphai-ci-mu={}-{}-{}/ci.npy'.format(mu,activation,kernel_initializer))
        # plt.plot(c)
        # plt.title('ci-{}-{}-rescaleout'.format(activation,kernel_initializer),fontsize=15)
        # plt.yticks( size=15)#设置大小及加粗 #weight='bold'
        # plt.xticks( size=15)
        # plt.savefig(save_path + '/ci-{}-{}-rescaleout.png'.format(activation,kernel_initializer))
        # plt.close()







