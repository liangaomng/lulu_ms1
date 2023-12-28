

import argparse
from src_lam.Agent import Expr_Agent
from src_PDE.Heat import PDE_HeatData
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')#"Expr2d/Expr_9.xlsx"
    args = parser.parse_args()
    read_set_path= ".EXPRS/Heat/Expr2d/Expr_1.xlsx"
    print("set_path",read_set_path)
    #expr
    HeatEquation=PDE_HeatData()

    expr = Expr_Agent(
        pde_task=True,
        solver=HeatEquation,
        Read_set_path=args.expr_set_path,
        Loss_Save_Path=read_set_path,
        compile_mode=False,
        record_interve=100)

    expr.Do_Expr()

    expr.Do_Expr()


