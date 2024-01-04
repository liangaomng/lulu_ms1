import argparse
from src_lam.Agent import Expr_Agent
from src_PDE.Heat import PDE_HeatData
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')#"Expr2d/Expr_9.xlsx"
    args = parser.parse_args()
    #expr
    np.random.seed(1234)
    HeatEquation=PDE_HeatData()
    data=HeatEquation.Get_Data()

    expr = Expr_Agent(
        pde_task="deepxde",
        solver=HeatEquation,
        Read_set_path=args.expr_set_path,
        compile_mode=False,)

    expr.Do_Expr()


