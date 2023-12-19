

import argparse
from src_lam.Agent import Expr_Agent

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')  # "Expr1d/Expr_1.xlsx"
    parser.add_argument('--compile_mode', default=False,type=bool,choices=[False,True], help='compile_mode')
    args = parser.parse_args()
    read_set_path = args.expr_set_path

    print("set_path",read_set_path)
    # expr
    expr = Expr_Agent(
                      pde_task=False,
                      args=args,
                      Read_set_path=read_set_path,
                      Loss_Save_Path=read_set_path,
                      compile_mode=args.compile_mode
                      )

    expr.Do_Expr()

