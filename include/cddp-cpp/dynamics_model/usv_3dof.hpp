#ifndef CDDP_USV_3DOF_HPP
#define CDDP_USV_3DOF_HPP

#include "cddp_core/dynamical_system.hpp"
#include <Eigen/Dense>

namespace cddp {

/**
 * @brief 3-DOF Planar Unmanned Surface Vehicle (USV) dynamics model.
 *
 * State vector: [x, y, psi, u, v, r]
 *  - (x, y): Inertial position (m)
 *  - psi: Yaw angle (rad)
 *  - (u, v): Body-fixed linear velocities (surge, sway) (m/s)
 *  - r: Body-fixed yaw rate (rad/s)
 *
 * Control input: [tau_u, tau_v, tau_r]
 *  - tau_u: Surge force (N)
 *  - tau_v: Sway force (N)
 *  - tau_r: Yaw torque (Nm)
 *
 */
class Usv3Dof : public DynamicalSystem {
public:
    /**
     * @brief Constructor for the Usv3Dof model.
     * @param timestep Time step for discretization (s).
     * @param integration_type Integration method ("euler" or "rk4").
     */
    Usv3Dof(double timestep, std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics dx/dt = f(x, u).
     * @param state Current state vector [x, y, psi, u, v, r].
     * @param control Current control input [tau_u, tau_v, tau_r].
     * @return State derivative vector [dx, dy, dpsi, du, dv, dr].
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the discrete-time dynamics x_{k+1} = F(x_k, u_k).
     *        Uses the base class numerical integration.
     * @param state Current state vector x_k.
     * @param control Current control input u_k.
     * @return Next state vector x_{k+1}.
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control);
    }

    /**
     * @brief Computes the Jacobian of the dynamics wrt. state (A = df/dx).
     *        Currently uses numerical differentiation from the base class.
     * @param state Current state vector.
     * @param control Current control input.
     * @return State Jacobian matrix A (6x6).
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Jacobian of the dynamics wrt. control (B = df/du).
     *        Currently uses numerical differentiation from the base class.
     * @param state Current state vector.
     * @param control Current control input.
     * @return Control Jacobian matrix B (6x3).
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics wrt. state (d^2f/dx^2).
     *        Currently uses numerical differentiation from the base class.
     * @param state Current state vector.
     * @param control Current control input.
     * @return Vector of state Hessian matrices (one 6x6 matrix per state dimension).
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics wrt. control (d^2f/du^2).
     *        Currently uses numerical differentiation from the base class.
     * @param state Current state vector.
     * @param control Current control input.
     * @return Vector of control Hessian matrices (one 3x3 matrix per state dimension).
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control) const override;
    
    /**
     * @brief Auto-diff version of continuous dynamics 
     * @param state State vector with dual numbers.
     * @param control Control vector with dual numbers.
     * @return State derivative vector with dual numbers.
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(
        const VectorXdual2nd& state, const VectorXdual2nd& control) const override;


private:
    // State indices
    static constexpr int STATE_X = 0;
    static constexpr int STATE_Y = 1;
    static constexpr int STATE_PSI = 2;
    static constexpr int STATE_U = 3;
    static constexpr int STATE_V = 4;
    static constexpr int STATE_R = 5;
    static constexpr int STATE_DIM = 6;

    // Control indices
    static constexpr int CONTROL_TAU_U = 0;
    static constexpr int CONTROL_TAU_V = 1;
    static constexpr int CONTROL_TAU_R = 2;
    static constexpr int CONTROL_DIM = 3;

    // --- Model Parameters ---
    // Rigid body mass and inertia
    double m_;    // Mass (kg)
    double Iz_;   // Yaw inertia (kg*m^2)
    // Added mass coefficients
    double X_udot_; // Surge added mass
    double Y_vdot_; // Sway added mass
    double Y_rdot_; // Sway-Yaw added mass coupling
    double N_vdot_; // Yaw-Sway added mass coupling
    double N_rdot_; // Yaw added mass
    // Linear damping coefficients
    double X_u_;    // Linear surge damping
    double Y_v_;    // Linear sway damping
    double Y_r_;    // Linear sway-yaw damping coupling
    double N_v_;    // Linear yaw-sway damping coupling
    double N_r_;    // Linear yaw damping

    // Precomputed matrices
    Eigen::Matrix3d M_inv_; // Inverse of total mass matrix M = M_RB + M_A
    Eigen::Matrix3d D_L_;   // Linear damping matrix
};

} // namespace cddp

#endif // CDDP_USV_3DOF_HPP 