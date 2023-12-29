
# import os
# os.environ['DDE_BACKEND'] = 'torch'

import argparse
from src_lam.Agent import Expr_Agent
from src_PDE.Heat import PDE_HeatData
import deepxde
print(deepxde.__path__[0])

import deepxde
# 其余的 DeepXDE 代码



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')#"Expr2d/Expr_9.xlsx"
    args = parser.parse_args()
    args.expr_set_path="EXPRS/Heat/Expr2d/Expr_1.xlsx"
    #expr
    HeatEquation=PDE_HeatData()
    data=HeatEquation.Get_Data()

    expr = Expr_Agent(
        pde_task="deepxde",
        solver=HeatEquation,
        Read_set_path=args.expr_set_path,
        compile_mode=False,
        record_interve=100)

    expr.Do_Expr()


