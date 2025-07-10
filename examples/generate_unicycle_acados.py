#!/usr/bin/env python3
"""
Generate ACADOS C code for unicycle model
"""

import numpy as np
import os
import sys
from casadi import SX, vertcat, sin, cos, Function

acados_path = "/home/astomodynamics/acados"
sys.path.append(os.path.join(acados_path, "interfaces", "acados_template"))

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    print("ACADOS template imported successfully")
except ImportError as e:
    print(f"Error importing ACADOS template: {e}")
    print("Make sure ACADOS is properly installed and PYTHONPATH is set")
    sys.exit(1)


def export_unicycle_model():
    """Create and export unicycle model for ACADOS"""
    model = AcadosModel()
    model.name = 'unicycle'
    
    x = SX.sym('x')
    y = SX.sym('y') 
    theta = SX.sym('theta')
    states = vertcat(x, y, theta)
    model.x = states
    
    v = SX.sym('v')
    omega = SX.sym('omega')
    controls = vertcat(v, omega)
    model.u = controls
    
    xdot = SX.sym('xdot')
    ydot = SX.sym('ydot')
    thetadot = SX.sym('thetadot')
    xdot_val = vertcat(xdot, ydot, thetadot)
    model.xdot = xdot_val
    
    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        omega
    )
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot_val - f_expl
    
    return model


def generate_c_code():
    """Generate C code for the unicycle model"""
    print("Creating unicycle model...")
    model = export_unicycle_model()
    
    ocp = AcadosOcp()
    ocp.model = model
    
    nx = 3
    nu = 2
    N = 100
    
    ocp.dims.N = N
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    ny = nx + nu
    ny_e = nx
    
    ocp.cost.Vx = np.vstack([np.zeros((nu, nx)), np.eye(nx)])
    ocp.cost.Vu = np.vstack([np.eye(nu), np.zeros((nx, nu))])
    ocp.cost.Vx_e = np.eye(nx)
    
    ocp.cost.W = np.diag([0.05, 0.05, 0.01, 0.01, 0.0])
    ocp.cost.W_e = np.diag([100.0, 100.0, 100.0])
    
    ocp.cost.yref = np.array([0.0, 0.0, 3.0, 3.0, np.pi/2])
    ocp.cost.yref_e = np.array([3.0, 3.0, np.pi/2])
    
    ocp.constraints.lbu = np.array([-2.0, -np.pi])
    ocp.constraints.ubu = np.array([2.0, np.pi])
    ocp.constraints.idxbu = np.array([0, 1])
    
    ocp.constraints.x0 = np.array([0.0, 0.0, np.pi/2])
    
    cx1, cy1, r1 = 1.0, 1.0, 0.4
    cx2, cy2, r2 = 1.5, 2.5, 0.4
    
    h_expr = vertcat(
        (model.x[0] - cx1)**2 + (model.x[1] - cy1)**2 - r1**2,
        (model.x[0] - cx2)**2 + (model.x[1] - cy2)**2 - r2**2
    )
    model.con_h_expr = h_expr
    
    ocp.constraints.lh = np.array([0.0, 0.0])
    ocp.constraints.uh = np.array([1e6, 1e6])
    
    ocp.solver_options.tf = N * 0.03
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 100
    
    ocp.code_export_directory = 'c_generated_code'
    
    print(f"Generating C code in {ocp.code_export_directory}/...")
    
    from acados_template import AcadosOcpSolver
    
    solver = AcadosOcpSolver(ocp, json_file='unicycle_acados_ocp.json')
    
    print("C code generation completed!")
    print(f"Generated files are in: {os.path.abspath(ocp.code_export_directory)}")
    
    print("\nGenerated files:")
    for root, dirs, files in os.walk(ocp.code_export_directory):
        level = root.replace(ocp.code_export_directory, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            print(f"{subindent}{file}")
    
    return ocp.code_export_directory


def main():
    """Main function"""
    print("ACADOS C Code Generator for Unicycle Model")
    print("=" * 50)
    
    code_dir = generate_c_code()
    
    print("\nNext steps:")
    print("1. Review the generated code")
    print("2. Update unicycle_benchmark.cpp to use the generated solver")


if __name__ == "__main__":
    main()