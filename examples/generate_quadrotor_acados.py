#!/usr/bin/env python3
"""
Generate ACADOS C code for quadrotor optimal control problem
"""

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sin, cos, norm_2, diag, mtimes

def export_quadrotor_model():
    """Create quadrotor dynamics model for ACADOS"""
    model_name = 'quadrotor'
    
    # Constants
    g = 9.81
    mass = 1.2
    arm_length = 0.165
    Ixx = 7.782e-3
    Iyy = 7.782e-3  
    Izz = 1.439e-2
    
    # State: [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    x = SX.sym('x')
    y = SX.sym('y')
    z = SX.sym('z')
    qw = SX.sym('qw')
    qx = SX.sym('qx')
    qy = SX.sym('qy')
    qz = SX.sym('qz')
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')
    omega_x = SX.sym('omega_x')
    omega_y = SX.sym('omega_y')
    omega_z = SX.sym('omega_z')
    
    x_state = vertcat(x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z)
    
    # Control: [f1, f2, f3, f4] - motor forces
    f1 = SX.sym('f1')
    f2 = SX.sym('f2')
    f3 = SX.sym('f3')
    f4 = SX.sym('f4')
    u = vertcat(f1, f2, f3, f4)
    
    model = AcadosModel()
    model.name = model_name
    model.x = x_state
    model.u = u
    
    # Rotation matrix from quaternion
    R11 = 1 - 2*(qy**2 + qz**2)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)
    
    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx**2 + qz**2)
    R23 = 2*(qy*qz - qx*qw)
    
    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx**2 + qy**2)
    
    # Total thrust and thrust vector in world frame
    f_total = f1 + f2 + f3 + f4
    thrust_world_x = f_total * R13 / mass
    thrust_world_y = f_total * R23 / mass
    thrust_world_z = f_total * R33 / mass - g
    
    # Torques in body frame
    tau_x = arm_length * (f2 - f4)
    tau_y = arm_length * (f1 - f3)
    tau_z = 0.0
    
    # Angular acceleration
    omega_dot_x = tau_x / Ixx
    omega_dot_y = tau_y / Iyy
    omega_dot_z = tau_z / Izz
    
    # Quaternion derivative
    qw_dot = 0.5 * (-qx*omega_x - qy*omega_y - qz*omega_z)
    qx_dot = 0.5 * (qw*omega_x + qy*omega_z - qz*omega_y)
    qy_dot = 0.5 * (qw*omega_y - qx*omega_z + qz*omega_x)
    qz_dot = 0.5 * (qw*omega_z + qx*omega_y - qy*omega_x)
    
    # State derivatives
    x_dot = vertcat(
        vx, vy, vz,
        qw_dot, qx_dot, qy_dot, qz_dot,
        thrust_world_x, thrust_world_y, thrust_world_z,
        omega_dot_x, omega_dot_y, omega_dot_z
    )
    
    model.f_expl_expr = x_dot
    model.f_impl_expr = x_dot - x_dot
    model.con_h_expr = vertcat()
    model.cost_y_expr = vertcat(u, x_state)
    model.cost_y_expr_e = x_state
    
    return model

def create_ocp_solver():
    """Create ACADOS OCP solver for quadrotor"""
    model = export_quadrotor_model()
    
    ocp = AcadosOcp()
    ocp.model = model
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nu + nx
    ny_e = nx
    
    N = 400
    Tf = 8.0
    ocp.dims.N = N
    ocp.solver_options.tf = Tf
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    dt = Tf / N
    Q = dt * np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    R = dt * 0.01 * np.eye(nu)
    Qf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    W = np.block([[R, np.zeros((nu, nx))], [np.zeros((nx, nu)), Q]])
    ocp.cost.W = W
    
    Vx = np.zeros((ny, nx))
    Vx[nu:, :] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[:nu, :] = np.eye(nu)
    
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.W_e = Qf
    ocp.cost.Vx_e = np.eye(nx)
    
    yref = np.zeros(ny)
    yref_e = np.zeros(ny_e)
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e
    
    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([4.0, 4.0, 4.0, 4.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 500
    ocp.solver_options.nlp_solver_tol_stat = 1e-5
    ocp.solver_options.nlp_solver_tol_eq = 1e-5
    ocp.solver_options.nlp_solver_tol_ineq = 1e-5
    ocp.solver_options.nlp_solver_tol_comp = 1e-5
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.print_level = 0
    
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp_quadrotor.json')
    return solver

if __name__ == '__main__':
    print("Generating ACADOS code for quadrotor...")
    solver = create_ocp_solver()
    print("Done! Generated files in c_generated_code/")
    
    print("\nTesting solver...")
    x0 = np.zeros(13)
    x0[0] = 3.0
    x0[2] = 2.0
    x0[3] = 1.0
    
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)
    
    for i in range(solver.N):
        yref = np.zeros(17)
        yref[4:7] = x0[0:3]
        yref[7] = 1.0
        solver.set(i, "yref", yref)
    
    yref_e = np.zeros(13)
    yref_e[0] = 3.0
    yref_e[2] = 2.0
    yref_e[3] = 1.0
    solver.set(solver.N, "yref", yref_e)
    
    status = solver.solve()
    print(f"Solver status: {status}")
    
    if status == 0:
        x_sol = solver.get(0, "x")
        u_sol = solver.get(0, "u")
        print(f"Initial state: {x_sol}")
        print(f"Initial control: {u_sol}")
        print("Solver test successful!")
    else:
        print("Solver test failed!")