import argparse
from src_lam.Agent import Expr_Agent
from src_PDE.Heat import PDE_HeatData
from src_PDE.Helmholtz import PDE_HelmholtzData

import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')#"Expr2d/Expr_9.xlsx"
    parser.add_argument('--pde_name', type=str, help='model_path')
    args = parser.parse_args()
    #expr
    np.random.seed(1234)
    print(args)
    if args.pde_name == "Helmholtz":
        
        solver=PDE_HelmholtzData()
        print("Helmholtz")
        
    elif args.pde_name == "Heat":
        
        solver=PDE_HeatData()
        print("Heat")
    
    print("PDE_Data:",solver)
    data=solver.Get_Data()

    expr = Expr_Agent(
        pde_task="deepxde",
        solver=solver,
        Read_set_path=args.expr_set_path,
        compile_mode=False)

    expr.Do_Expr()


