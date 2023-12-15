import numpy as np
import torch
from .nn_class import  Multi_scale2,Single_MLP
from torch import optim
from torch import nn
from abc import abstractmethod
from src_lam.xls2_object import Return_expr_dict
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from .excel2yaml import Excel2yaml
from .Plot import Plot_Adaptive
from collections import deque
from .Analyzer import Analyzer4scale
class Expr():
    def __init__(self):
        self.model=None
        self.Read_set_path=None
        self.Save_Path=None
    @abstractmethod
    def _Random(self,seed):
        pass
    @abstractmethod
    def _read_arg_xlsx(self,path):
        pass

    @abstractmethod
    def Prepare_model_Dataloader(self,args):
        pass
    @abstractmethod
    def Valid(self,**kwargs):
        pass
    @abstractmethod
    def Train(self):
        pass
    @abstractmethod
    def Test4Save(self,**kwargs):
        pass
    @abstractmethod
    def Do_Expr(self):
        pass
    @abstractmethod
    def _CheckPoint(self,**kwargs):
        pass

class Base_Args():
    def __init__(self):
        self.model=None
        self.lr=None
        self.seed=None
        self.epoch=None
        self.Train_Dataset=None
        self.Valid_Dataset=None
        self.Test_Dataset=None
        self.Save_Path=None
        self.batch_size=None
        self._note=None
        self.Con_record=None

    @abstractmethod
    def Layer_set(self,layer_set:list):
        pass

    @abstractmethod
    def Act_set(self,act_set:list):
        pass

    @abstractmethod
    def Ini_Set(self,ini_set:list):
        pass
    @property
    def note(self):
        return self._note
    @note.setter
    def note(self,note):
        self._note=note

class Multi_scale2_Args(Base_Args):
    def __init__(self,scale_coff):
        super().__init__()
        self.subnets_number=len(scale_coff)#  确定后面数组的行，如果4个尺度，就是4行
        self.Scale_Coeff = scale_coff
        self.Act_Set_list = []
        self.Layer_Set_list = []
        self.Ini_Set_list = []
        self.Residual_Set_list = []
        self.Save_Path=None
        self.penalty=None
    def Layer_set(self,layer_set:list)->list: #单子网络
        self.Layer_Set_list=layer_set #[1, 10, 10, 10, 1]
    def Act_set(self,act_list)->np.ndarray:
        self.Act_Set_list=act_list #
    def Ini_Set(self,ini_set:list)->np.ndarray:
        #'Ini_Set': ['xavier_uniform']
        self.Ini_Set_list=ini_set
class Expr_Agent(Expr):
    def __init__(self,**kwargs):
        super().__init__()
        xls2_dict =Return_expr_dict.sheet2dict(kwargs["Read_set_path"])
        self.args = self._read_arg_xlsx(xls2_dict) # 读取参数
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Save_Path = kwargs["Loss_Save_Path"] #save pathlike  “expr1/expr_1.xlsx‘
        self.model = None
        self._Random(seed=self.args.seed)
        self._Check()
        self.Prepare_model_Dataloader()
        self.plot = Plot_Adaptive() # 画图
        Excel2yaml(kwargs["Read_set_path"]).excel2yaml() #convert 2yaml
    def _read_arg_xlsx(self,xls2_object:dict)->Multi_scale2_Args:

        args=Multi_scale2_Args(xls2_object["SET"].Scale_Coeff)

        args.model=xls2_object["SET"].Model[0]
        args.lr=xls2_object["SET"].lr[0]
        args.seed=int(xls2_object["SET"].SEED[0])
        args.epoch=int(xls2_object["SET"].Epoch[0])
        args.Train_Dataset=xls2_object["SET"].Train_Dataset[0]
        args.Valid_Dataset=xls2_object["SET"].Valid_Dataset[0]
        args.Test_Dataset=xls2_object["SET"].Test_Dataset[0]
        args.Save_Path=xls2_object["SET"].Save_Path[0]
        args.batch_size=int(xls2_object["SET"].Batch_size[0])
        args.Con_record=xls2_object["SET"].Con_record #list

        #  收集子网络的信息
        for i in range(int(args.subnets_number)):
            sub_key="Subnet"+str(i+1)
            args.Act_Set_list.append(xls2_object[sub_key].Act_Set)
            args.Layer_Set_list.append(xls2_object[sub_key].Layer_Set)
            args.Ini_Set_list.append(xls2_object[sub_key].Ini_Set)
            args.Residual_Set_list.append(xls2_object[sub_key].Residual)

        return args
    def _Random(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    def Prepare_model_Dataloader(self):

        if self.args.model == "mscalenn2":
            scale_omegas_coeff = self.args.Scale_Coeff #[1,2,3]
            layer_set=self.args.Layer_Set_list#[1, 10, 10, 10, 1]
            multi_net_act=self.args.Act_Set_list #[4,3] 4个子网络，每个3层激活

            multi_init_weight=self.args.Ini_Set_list #[4,3] 4个子网络，每个3层初始化
            sub_layer_number=len(scale_omegas_coeff)
            residual_en=self.args.Residual_Set_list #[4,3] 4个子网络，每个3层残差

            self.model = Multi_scale2(
                sub_layer_number = np.array(sub_layer_number),
                layer_set = np.array(layer_set[0]),#实际是4个list，每个list是一个子网络的神经元
                act_set = np.array(multi_net_act),
                ini_set = np.array(multi_init_weight),
                residual= residual_en[0],
                scale_number=scale_omegas_coeff
            )
        if self.args.model == "fnn":
            layer_set = self.args.Layer_Set_list#[1, 10, 10, 10, 1]
            residual_en = self.args.Residual_Set_list
            activation_set=self.args.Act_Set_list

            self.model = Single_MLP(
                input_size=layer_set[0],
                layer_set= layer_set,
                use_residual= residual_en,
                activation_set= np.array(activation_set)
            )
        #加速
        #self.model = torch.compile(self.model, mode="max-autotune")

        self._train_dataset = torch.load(self.args.Train_Dataset)
        self._valid_dataset = torch.load(self.args.Valid_Dataset)
        self._test_dataset = torch.load(self.args.Test_Dataset)
        self._train_loader = DataLoader(dataset=self._train_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=True)
        self._valid_loader = DataLoader(dataset=self._valid_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=True)
        self._test_loader = DataLoader(dataset=self._test_dataset,
                                        batch_size=self.args.batch_size,
                                        shuffle=True)
    def _Check(self):
        # 检查读取路径
        if not os.path.exists(self.args.Train_Dataset):
            raise FileNotFoundError("Train_Dataset not found")

        # 检查保存路径, 如果没有就创建一个
        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.loss_record_sheet = "LossRecord"  # 指定工作表名称


        try:
            self.loss_record_df = pd.read_excel(self.loss_record_file,
                                                sheet_name="Result")
        except Exception as e:
            self.loss_record_df = pd.DataFrame(columns=['epoch',
                                                        'train_loss',
                                                        'valid_loss',
                                                        'test_loss'])
    def _update_loss_record(self, epoch, train_loss=None, valid_loss=None, test_loss=None):
        # 创建一个新记录的DataFrame
        new_record_df = pd.DataFrame({
            'epoch': [epoch],
            'train_loss': [train_loss],
            'valid_loss': [valid_loss],
            'test_loss': [test_loss]
        })

        # 如果文件不存在，则初始化一个空的DataFrame
        if not os.path.isfile(self.Save_Path):
            self.loss_record_df = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'test_loss'])

        # 否则，读取现有文件
        else:
            self.loss_record_df = pd.read_excel(self.Save_Path, sheet_name=self.loss_record_sheet)

        # 将新记录追加到DataFrame中
        self.loss_record_df = pd.concat([self.loss_record_df, new_record_df], ignore_index=True).drop_duplicates(
            subset=['epoch'])

        try:
            with pd.ExcelWriter(self.Save_Path, mode='a', if_sheet_exists='overlay') as writer:
                self.loss_record_df.to_excel(writer, sheet_name=self.loss_record_sheet, index=False)
        except Exception as e:
            print("An error occurred:", e)
    def _Valid(self,**kwargs):

        epoch = kwargs["epoch"]
        num_epochs = kwargs["num_epochs"]
        self.model.eval()
        criterion = nn.MSELoss()

        with torch.no_grad():  # 在验证过程中不计算梯度
            sum_val_loss = 0.0
            for inputs, labels in  self._valid_loader:
                inputs= inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                val_loss = criterion(outputs, labels)
                sum_val_loss += val_loss.item()
        avg_val_loss = sum_val_loss / len( self._valid_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Val Loss: {avg_val_loss:.4f}')
        return avg_val_loss
    def Train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        self.model=self.model.to(self.device)

        print(f"we are using device {self.device}")
        for epoch in range(0,self.args.epoch,1):

            epoch_loss = 0.0
            for i, (x, y) in enumerate( self._train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            aver_loss = epoch_loss / len( self._train_loader)

            print('epoch: {}, train loss: {:.6f}'.format(epoch, aver_loss))
            if (epoch % 100 == 0):
                valid_loss=self._Valid(epoch=epoch,num_epochs=self.args.epoch)
                self._CheckPoint(epoch=epoch)
                test_loss =self._Test4Save(epoch=epoch)
                self._update_loss_record(epoch, train_loss=aver_loss, valid_loss=valid_loss, test_loss=test_loss)
    def _Test4Save(self,**kwargs):

        epoch = kwargs["epoch"]
        self.model.eval()
        criterion = nn.MSELoss()

        with torch.no_grad():
            sum_test_loss = 0.0

            for inputs, labels in self._test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                test_loss = criterion(outputs, labels)
                sum_test_loss += test_loss.item()

        avg_test_loss = sum_test_loss / len(self._test_loader)

        # 画loss的值

        self._save4plot(epoch, avg_test_loss)

        print(f'Test Loss: {avg_test_loss:.6f}')
        return avg_test_loss
    def _save4plot(self,epoch,avg_test_loss):
        # 创建两个子图，上下布局
        avg_test_loss= avg_test_loss
        # 加载测试数据集
        test_data = torch.load(self.args.Test_Dataset)

        # 将TensorDataset转换为numpy数组
        x_test = test_data.tensors[0].numpy()
        y_true = test_data.tensors[1].numpy()

        #analyzer
        analyzer=Analyzer4scale(model=self.model,
                                scale_coeffs=self.args.Scale_Coeff)
        # 读取损失记录
        loss_record_df = pd.read_excel(self.Save_Path, sheet_name=self.loss_record_sheet)

        # 获取模型预测
        pred = self.model(torch.from_numpy(x_test).float().to(self.device)).detach().cpu().numpy()
        if x_test.shape[1] == 1:
            fig,axes=self.plot.plot_1d(nrow=3,ncol=3,
                                  loss_record_df=loss_record_df,
                                  analyzer=analyzer,
                                  x_test=x_test,
                                  y_true=y_true,
                                  pred=pred,
                                  epoch=epoch,
                                  avg_test_loss=avg_test_loss,
                                  contribution_record=self.args.Con_record,)

        elif x_test.shape[1] == 2:
            #第一个图画预测热力图
            fig, ax = plt.subplots(3, 1, figsize=(20, 14))
            from scipy.interpolate import Rbf
            # 创建网格
            grid_x, grid_y = np.mgrid[x_test[:, 0].min():x_test[:, 0].max():100j,
                             x_test[:, 1].min():x_test[:, 1].max():100j]

            # 提取x和y坐标
            x = x_test[:, 0]
            y = x_test[:, 1]

            # 使用预测值创建Rbf插值函数
            rbf_pred = Rbf(x, y, pred[:, 0])
            grid_z_pred = rbf_pred(grid_x, grid_y)

            # 绘制预测热力图
            ax[0].imshow(grid_z_pred.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
            ax[0].set_title('Pred Interpolated Heatmap')
            ax[0].set_xlabel('X Coordinate')
            ax[0].set_ylabel('Y Coordinate')

            # 使用真实值创建Rbf插值函数
            rbf_true = Rbf(x, y, y_true[:, 0])
            grid_z_true = rbf_true(grid_x, grid_y)

            # 绘制真实热力图
            ax[1].imshow(grid_z_true.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
            ax[1].set_title('True Interpolated Heatmap')
            ax[1].set_xlabel('X Coordinate')
            ax[1].set_ylabel('Y Coordinate')

        # 保存整个图表
        fig.savefig('{}/combined_loss_{}.png'.format(self.args.Save_Path, epoch),
                    bbox_inches='tight', format='png')
        # # 关闭图表以释放内存
        # plt.close(fig)
    def _CheckPoint(self,**kwargs):
            epoch=kwargs["epoch"]
            dir_name=self.args.Save_Path+"/"+self.args.model+".pth"
            if os.path.exists(self.args.Save_Path):
                pass
            else:
                os.mkdir(self.args.Save_Path)

            torch.save(self.model.state_dict(), dir_name)
            print(f"save model at epoch {epoch}")
    def Do_Expr(self):
        self.Train()
        print("we have done the expr")

class PDE_Agent(Expr):
    def __init__(self,solver,**kwargs):

        xls2_dict = Return_expr_dict.sheet2dict(kwargs["Read_set_path"])
        self.args = self._read_arg_xlsx(xls2_dict)  # 读取参数
        self.model=None
        self.solver=solver
        self._Random(seed=self.args.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.plot = Plot_Adaptive() # 画图
        Excel2yaml(kwargs["Read_set_path"]).excel2yaml() #convert 2yaml
        self.Prepare_model()
    def Prepare_model(self):
        if self.args.model == "mscalenn2":
            scale_omegas_coeff = self.args.Scale_Coeff #[1,2,3]
            layer_set=self.args.Layer_Set_list#[1, 10, 10, 10, 1]
            multi_net_act=self.args.Act_Set_list #[4,3] 4个子网络，每个3层激活

            multi_init_weight=self.args.Ini_Set_list #[4,3] 4个子网络，每个3层初始化
            sub_layer_number=len(scale_omegas_coeff)
            residual_en=self.args.Residual_Set_list #[4,3] 4个子网络，每个3层残差

            self.model = Multi_scale2(
                sub_layer_number = np.array(sub_layer_number),
                layer_set = np.array(layer_set[0]),#实际是4个list，每个list是一个子网络的神经元
                act_set = np.array(multi_net_act),
                ini_set = np.array(multi_init_weight),
                residual= residual_en[0],
                scale_number=scale_omegas_coeff
            )
        if self.args.model == "fnn":
            layer_set = self.args.Layer_Set_list#[1, 10, 10, 10, 1]
            residual_en = self.args.Residual_Set_list
            activation_set=self.args.Act_Set_list

            self.model = Single_MLP(
                input_size=layer_set[0],
                layer_set= layer_set,
                use_residual= residual_en,
                activation_set= np.array(activation_set)
            )

    def _read_arg_xlsx(self,xls2_object:dict):

        args=Multi_scale2_Args(xls2_object["SET"].Scale_Coeff)
        args.model=xls2_object["SET"].Model[0]
        args.lr=xls2_object["SET"].lr[0]
        args.seed=int(xls2_object["SET"].SEED[0])
        args.epoch=int(xls2_object["SET"].Epoch[0])
        args.Save_Path=xls2_object["SET"].Save_Path[0]
        args.batch_size=int(xls2_object["SET"].Batch_size[0])
        args.penalty=float(xls2_object["SET"].Penalty[0])
        args.Con_record=xls2_object["SET"].Con_record #list
        args.mu=xls2_object["SET"].Mu[0]

        #  收集子网络的信息
        for i in range(int(args.subnets_number)):
            sub_key="Subnet"+str(i+1)
            args.Act_Set_list.append(xls2_object[sub_key].Act_Set)
            args.Layer_Set_list.append(xls2_object[sub_key].Layer_Set)
            args.Ini_Set_list.append(xls2_object[sub_key].Ini_Set)
            args.Residual_Set_list.append(xls2_object[sub_key].Residual)
        print("args",vars(args))

        return args
    def _Random(self,**kwargs):
        seed=kwargs["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def Train_PDE(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        boundary_loss=nn.MSELoss()
        for epoch in range(self.args.epoch):
            epoch_loss = 0.0
            #every epoch to sample in PDE_solver
            data=self.solver.sample(batch=int(self.args.batch_size))
            train_data=torch.from_numpy(data).float()

            inputs = train_data[:,:,0:2].to(self.device)
            labels = train_data[:,:,2:3].to(self.device)
            #预测
            outputs = self.model(inputs)

            boundary_pred=outputs[:,2400:,0]
            boundary_label =  labels[:,2400:,0].to(self.device)
            b_loss = self.args.penalty * boundary_loss(boundary_pred, boundary_label)

            loss1=criterion(labels[:,0:2400,0], outputs[:,0:2400,0])

            loss = loss1+b_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
            print(f"{epoch}_loss",epoch_loss)
            #画网格
            if (epoch % 1000 == 0):
                #self.solver.plot_contour(mu=7*np.pi)#真实解

                x=torch.linspace(-1,1,100)
                y=torch.linspace(-1,1,100)
                # 创建一个新的二维张量，其中每行包含来自x和y的相应元素
                x, y = torch.meshgrid(x, y)
                xy = torch.stack([x, y], dim=2).reshape(1,-1, 2)

                xy=xy.to(self.device)
                u=self.model(xy) #(1,1000,2)
                u=u.cpu().detach().numpy() #[]
                u=u.reshape(100,100)
                plt.imshow(u, extent=[-1, 1, -1, 1], origin='lower')
                plt.colorbar()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Heatmap of u(x, y)')
                plt.show()
            # if (epoch % 100 == 0):
            #     valid_loss=self._Valid(epoch=epoch)
            #     self._CheckPoint(epoch=epoch)
            #     test_loss =self._Test4Save(epoch=epoch)
            #     self._update_loss_record(epoch, train_loss=aver_loss, valid_loss=valid_loss, test_loss=test_loss)
    def _Valid(self,epoch):
        self.model.eval()
        valid_loss = 0.0
        criterion = nn.MSELoss()
        for inputs, labels in self._valid_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
        aver_loss = valid_loss / len(self._valid_loader)
        print('epoch: {}, valid loss: {:.6f}'.format(epoch, aver_loss))
        return aver_loss
    def _update_loss_record(self,epoch,**kwargs):
        train_loss=kwargs["train_loss"]
        valid_loss=kwargs["valid_loss"]
        test_loss=kwargs["test_loss"]
        self.args.train_loss_record.append(train_loss)
        self.args.valid_loss_record.append(valid_loss)
        self.args.test_loss_record.append(test_loss)
        self.args.epoch_record.append(epoch)
    def _Test4Save(self,epoch):
        self.model.eval()
        test_loss = 0.0
        criterion = nn.MSELoss()
        for inputs, labels in self._test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        aver_loss = test_loss / len(self._test_loader)
        print('epoch: {}, test loss: {:.6f}'.format(epoch, aver_loss))
        return aver_loss
    def _CheckPoint(self,**kwargs):
        epoch=kwargs["epoch"]
        dir_name=self.args.Save_Path+"/"+self.args.model+".pth"
        if os.path.exists(self.args.Save_Path):
            pass
        else:
            os.mkdir(self.args.Save_Path)

        torch.save(self.model.state_dict(), dir_name)
        print(f"save model at epoch {epoch}")
    def Do_Expr(self):
        self.Train_PDE()
        print("we have done the expr")





if __name__=="__main__":
    pass