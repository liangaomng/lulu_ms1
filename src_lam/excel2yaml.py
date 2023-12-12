import pandas as pd
import yaml
from pathlib import Path


class Excel2yaml():
    def __init__(self,ex_path):
        self.ex_path=ex_path
        # 构造新的保存路径，替换原始扩展名为.yaml
        self.save_path = str(Path(ex_path).with_suffix('.yaml'))
        print("save_path",self.save_path)

    def excel2yaml(self,exhit='LossRecord'):
        # 读取 Excel 文件中的所有 sheets
        xls = pd.ExcelFile(self.ex_path)
        # Get all sheet names and exclude 'LossRecord'
        sheet_names = [sheet_name for sheet_name in xls.sheet_names if sheet_name != exhit]

        # 初始化一个字典来存储所有 sheets 的数据，排除 'LossRecord' 工作表
        sheets_data = {}
        for sheet in sheet_names:
            if sheet != exhit:
                df = pd.read_excel(self.ex_path, sheet_name=sheet)

                # 处理空值：可以选择删除含空值的行，或填充默认值
                df = df.dropna()  # 删除含空值的行

                # 删除重复行
                df = df.drop_duplicates()

                # 转换为字典
                sheets_data[sheet] = df.to_dict(orient='records')
                # 将字典转换为 YAML 格式的字符串
        yaml_data = yaml.dump(sheets_data, allow_unicode=True)

        # 将 YAML 数据保存到文件
        with open(self.save_path, 'w', encoding='utf-8') as file:
            file.write(yaml_data)


if __name__=="__main__":
    file_path='../Expr1d/Expr_1.xlsx'
    Excel2yaml(file_path).excel2yaml()