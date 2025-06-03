#include "dynamics_model/usv_3dof.hpp"
#include <cmath>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

// NOTE: Use with caution. TODO: Check if this is correct.
namespace cddp {

// Define convenient aliases for state/control indices
// using State = Usv3Dof::StateIndex; // Removed - incorrect
// using Control = Usv3Dof::ControlIndex; // Removed - incorrect

Usv3Dof::Usv3Dof(double timestep, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type)
{
    // --- Assign Generic USV Parameters ---
    m_ = 100.0;    // Mass (kg)
    Iz_ = 10.0;   // Yaw inertia (kg*m^2)

    // Added mass coefficients (using Fossen's notation convention where X_udot is negative)
    X_udot_ = -10.0; // kg
    Y_vdot_ = -50.0; // kg
    Y_rdot_ = -5.0;  // kg*m
    N_vdot_ = -5.0;  // kg*m (often symmetric Y_rdot = N_vdot)
    N_rdot_ = -5.0;  // kg*m^2

    // Linear damping coefficients (using convention where X_u is negative damping force coeff)
    X_u_ = -20.0;    // Ns/m
    Y_v_ = -100.0;   // Ns/m
    Y_r_ = 0.0;     // Nms/rad (simplified)
    N_v_ = 0.0;     // Nms/rad (simplified)
    N_r_ = -20.0;    // Nms/rad

    // --- Precompute Matrices ---
    // Mass matrix (Rigid Body + Added Mass)
    Eigen::Matrix3d M_rb = Eigen::Matrix3d::Zero();
    M_rb.diagonal() << m_, m_, Iz_;

    Eigen::Matrix3d M_a = Eigen::Matrix3d::Zero();
    M_a(0, 0) = -X_udot_;
    M_a(1, 1) = -Y_vdot_;
    M_a(1, 2) = -Y_rdot_;
    M_a(2, 1) = -N_vdot_;
    M_a(2, 2) = -N_rdot_;

    Eigen::Matrix3d M = M_rb + M_a;
    M_inv_ = M.inverse(); // Precompute inverse

    // Linear damping matrix (D_L in dx/dt = ... - D_L*nu)
    D_L_ = Eigen::Matrix3d::Zero();
    D_L_(0, 0) = -X_u_;
    D_L_(1, 1) = -Y_v_;
    D_L_(1, 2) = -Y_r_;
    D_L_(2, 1) = -N_v_;
    D_L_(2, 2) = -N_r_;
}

Eigen::VectorXd Usv3Dof::getContinuousDynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double time) const
{
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

    // Extract state components
    const double psi = state(Usv3Dof::STATE_PSI);
    const double u = state(Usv3Dof::STATE_U);
    const double v = state(Usv3Dof::STATE_V);
    const double r = state(Usv3Dof::STATE_R);
    Eigen::Vector3d nu(u, v, r); // Body velocity vector

    // Extract control inputs (forces/torque)
    const Eigen::Vector3d tau = control;

    // --- 1. Kinematics: eta_dot = J(psi) * nu ---
    const double c_psi = std::cos(psi);
    const double s_psi = std::sin(psi);
    state_dot(Usv3Dof::STATE_X) = c_psi * u - s_psi * v;
    state_dot(Usv3Dof::STATE_Y) = s_psi * u + c_psi * v;
    state_dot(Usv3Dof::STATE_PSI) = r;

    // --- 2. Dynamics: nu_dot = M_inv * (tau - C(nu)*nu - D_L*nu) ---
    // Coriolis matrix C(nu)
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    const double m_x = m_ - X_udot_; // Effective mass in surge
    const double m_y = m_ - Y_vdot_; // Effective mass in sway
    const double m_yr = -Y_rdot_;   // Effective coupling term (note sign from M_a)

    C(0, 2) = -m_y * v - m_yr * r;
    C(1, 2) =  m_x * u;
    C(2, 0) =  m_y * v + m_yr * r;
    C(2, 1) = -m_x * u;

    // Calculate accelerations
    Eigen::Vector3d nu_dot = M_inv_ * (tau - C * nu - D_L_ * nu);
    state_dot(Usv3Dof::STATE_U) = nu_dot(0);
    state_dot(Usv3Dof::STATE_V) = nu_dot(1);
    state_dot(Usv3Dof::STATE_R) = nu_dot(2);

    return state_dot;
}


VectorXdual2nd Usv3Dof::getContinuousDynamicsAutodiff(
        const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const
{
    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);
    using autodiff::dual2nd; // Use autodiff's dual number type

    // Extract state components
    const dual2nd psi = state(Usv3Dof::STATE_PSI);
    const dual2nd u = state(Usv3Dof::STATE_U);
    const dual2nd v = state(Usv3Dof::STATE_V);
    const dual2nd r = state(Usv3Dof::STATE_R);
    autodiff::Vector3dual2nd nu; // Added autodiff:: prefix
    nu << u, v, r; // Body velocity vector

    // Extract control inputs (forces/torque)
    const autodiff::Vector3dual2nd tau = control; // Added autodiff:: prefix

    // --- 1. Kinematics: eta_dot = J(psi) * nu ---
    // Use autodiff math functions (cos, sin)
    const dual2nd c_psi = cos(psi);
    const dual2nd s_psi = sin(psi);
    state_dot(Usv3Dof::STATE_X) = c_psi * u - s_psi * v;
    state_dot(Usv3Dof::STATE_Y) = s_psi * u + c_psi * v;
    state_dot(Usv3Dof::STATE_PSI) = r;

    // --- 2. Dynamics: nu_dot = M_inv * (tau - C(nu)*nu - D_L*nu) ---
    // Coriolis matrix C(nu) with dual numbers
    autodiff::Matrix3dual2nd C = autodiff::Matrix3dual2nd::Zero(); // Added autodiff:: prefixes
    const double m_x = m_ - X_udot_; // Constants remain double
    const double m_y = m_ - Y_vdot_;
    const double m_yr = -Y_rdot_;

    C(0, 2) = -m_y * v - m_yr * r;
    C(1, 2) =  m_x * u;
    C(2, 0) =  m_y * v + m_yr * r;
    C(2, 1) = -m_x * u;

    // Calculate accelerations using Eigen operations compatible with dual types
    // Need M_inv_ and D_L_ as dual matrices (or cast) for compatibility
    autodiff::Vector3dual2nd nu_dot = M_inv_.cast<dual2nd>() * (tau - C * nu - D_L_.cast<dual2nd>() * nu); // Added autodiff:: prefix
    state_dot(Usv3Dof::STATE_U) = nu_dot(0);
    state_dot(Usv3Dof::STATE_V) = nu_dot(1);
    state_dot(Usv3Dof::STATE_R) = nu_dot(2);

    return state_dot;
}


Eigen::MatrixXd Usv3Dof::getStateJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double time) const
{
    // Use analytical calculation for better accuracy/efficiency
     Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    // Extract state components needed for Jacobians
    const double psi = state(Usv3Dof::STATE_PSI);
    const double u = state(Usv3Dof::STATE_U);
    const double v = state(Usv3Dof::STATE_V);
    const double r = state(Usv3Dof::STATE_R);

    // --- Kinematic Part (Upper Left 3x6 Block) ---
    const double c_psi = std::cos(psi);
    const double s_psi = std::sin(psi);

    // d(eta_dot)/d(psi)
    A(Usv3Dof::STATE_X, Usv3Dof::STATE_PSI) = -s_psi * u - c_psi * v;
    A(Usv3Dof::STATE_Y, Usv3Dof::STATE_PSI) =  c_psi * u - s_psi * v;
    // A(Usv3Dof::STATE_PSI, Usv3Dof::STATE_PSI) = 0; // Already zero

    // d(eta_dot)/d(nu) - The rotation matrix J(psi)
    A(Usv3Dof::STATE_X, Usv3Dof::STATE_U) = c_psi;
    A(Usv3Dof::STATE_X, Usv3Dof::STATE_V) = -s_psi;
    A(Usv3Dof::STATE_Y, Usv3Dof::STATE_U) = s_psi;
    A(Usv3Dof::STATE_Y, Usv3Dof::STATE_V) = c_psi;
    A(Usv3Dof::STATE_PSI, Usv3Dof::STATE_R) = 1.0;

    // --- Dynamic Part (Lower Right 3x3 Block) ---
    // d(nu_dot)/d(nu) = M_inv * (-d(C*nu)/d(nu) - D_L)

    // Calculate d(C*nu)/d(nu)
    Eigen::Matrix3d dCnu_dnu = Eigen::Matrix3d::Zero();
    const double m_x = m_ - X_udot_;
    const double m_y = m_ - Y_vdot_;
    const double m_yr = -Y_rdot_; // Note sign change from M_a definition

    // Row 0: d/d(u,v,r) of [-m_y*v*r - m_yr*r*r]
    // dCnu_dnu(0, 0) = 0;
    dCnu_dnu(0, 1) = -m_y * r;
    dCnu_dnu(0, 2) = -m_y * v - 2 * m_yr * r;
    // Row 1: d/d(u,v,r) of [m_x*u*r]
    dCnu_dnu(1, 0) = m_x * r;
    // dCnu_dnu(1, 1) = 0;
    dCnu_dnu(1, 2) = m_x * u;
    // Row 2: d/d(u,v,r) of [ (m_y*v + m_yr*r)*u - m_x*u*v ] = [ (m_y - m_x)uv + m_yr*ru ]
    dCnu_dnu(2, 0) = (m_y - m_x) * v + m_yr * r;
    dCnu_dnu(2, 1) = (m_y - m_x) * u;
    dCnu_dnu(2, 2) = m_yr * u;


    // Lower right block of A
    Eigen::MatrixXd A_dyn = M_inv_ * (-dCnu_dnu - D_L_);
    A.block<3, 3>(Usv3Dof::STATE_U, Usv3Dof::STATE_U) = A_dyn;

    // d(nu_dot)/d(eta) = 0 (Lower Left 3x3 Block is zero)
    return A;
}

Eigen::MatrixXd Usv3Dof::getControlJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double time) const
{
    // Use analytical calculation
     Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);

    // d(eta_dot)/d(tau) = 0 (Upper 3x3 block is zero)

    // d(nu_dot)/d(tau) = M_inv * d(tau)/d(tau) = M_inv
    B.block<3, 3>(Usv3Dof::STATE_U, Usv3Dof::CONTROL_TAU_U) = M_inv_;

    return B;
}

std::vector<Eigen::MatrixXd> Usv3Dof::getStateHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double dt) const
{
     return DynamicalSystem::getStateHessian(state, control, dt);
}

std::vector<Eigen::MatrixXd> Usv3Dof::getControlHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double time) const
{
    // Dynamics are linear in control (nu_dot = M_inv * (tau - ...)),
    // so the second derivative d^2f / du^2 is zero.
    std::vector<Eigen::MatrixXd> hessian(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessian[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessian;
}


} // namespace cddp