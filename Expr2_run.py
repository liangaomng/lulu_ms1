

import argparse
from src_lam.Agent import Expr_Agent
from src_lam.possion_holes import PoissonEquationWithHoles

domain_size = 1.0
circle_params = [(-0.6, -0.6, 0.3), (0.3, -0.3, 0.6), (0.6, 0.6, 0.3)]  # 圆的参数
ellipse_params = [(-0.5, 0.5, 1 / 4, 1 / 8)]  # 椭圆的参数
num_samples_domain = 2400  # 包括边界
num_samples_circles = [550, 1100, 550]  # 三个圆的采样数量
num_samples_ellipse = 400

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pytorch")
    parser.add_argument('--expr_set_path', type=str, help='expr_set_path')  # "Expr2d/Expr_8.xlsx"
    parser.add_argument('--compile_mode', default=False,type=bool,choices=[False,True], help='compile_mode')
    args = parser.parse_args()
    read_set_path = args.expr_set_path
    # poisson
    poisson = PoissonEquationWithHoles(domain_size,
                                       circle_params,
                                       ellipse_params,
                                       num_samples_domain,
                                       num_samples_circles,
                                       num_samples_ellipse)

    # expr
    expr = Expr_Agent(
                      pde_task=True,
                      solver=poisson,
                      Read_set_path=args.expr_set_path,
                      Loss_Save_Path=read_set_path,
                      compile_mode=args.compile_mode,
                      record_interve=100)

    expr.Do_Expr()

