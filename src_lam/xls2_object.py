import pandas as pd

# 定义通用类，用于将DataFrame的列转换为对象的属性
class GenericConfig:
    def __init__(self, df):
        for column in df.columns:
            # 特定于 'Residual' 列的转换
            if column == 'Residual':
                self.__dict__[column] = [True if x == 1.0 else False for x in df[column].dropna()]
            if column=='Scale_Coeff':
                # 获取该列的单元格值，假设它是第一行
                scale_coeff_string = df[column].dropna().iloc[0]

                # 分割字符串，转换为整数列表
                scale_coeff_list = [int(value.strip()) for value in scale_coeff_string.split(',')]
                # 更新对象的字典
                self.__dict__[column] = scale_coeff_list
            else:
                self.__dict__[column] = df[column].dropna().tolist()

    def __str__(self):
        return str(self.__dict__)
class Return_expr_dict():
    def __init__(self):
        pass

    @classmethod
    def sheet2dict(self,path):
        xlsx = pd.ExcelFile(path)
        # 创建一个字典来存储每个工作表的配置
        sheet_configs = {}

        for sheet in xlsx.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            sheet_configs[sheet] = GenericConfig(df)
        return sheet_configs

