

import argparse
from src_lam.Agent import Expr_Agent

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')#"Expr2d/Expr_3.xlsx"
    args = parser.parse_args()
    read_set_path= args.expr_set_path
    print("path",read_set_path)
    # expr
    # expr = Expr_Agent(args=args,
    #                   Read_set_path=read_set_path,
    #                   Loss_Save_Path=read_set_path)
    #
    # expr.Do_Expr()


