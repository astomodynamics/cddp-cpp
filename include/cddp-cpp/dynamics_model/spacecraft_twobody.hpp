#ifndef SPACECRAFT_TWOBODY_HPP
#define SPACECRAFT_TWOBODY_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief SpacecraftTwobody dynamics model
 *
 * This class models the dynamics of a spacecraft in a two-body problem.
 *
 * The state is represented by the position and velocity of the spacecraft in
 * inertial Cartesian coordinates. The control input is the 3D thrust vector.
 *
 * State:  [x, y, z, vx, vy, vz]
 * Control: [ux, uy, uz]
 */
class SpacecraftTwobody : public DynamicalSystem {
public:
  /**
   * @brief Constructor
   *
   * @param timestep Discretization time step [s]
   * @param mu Gravitational parameter of the central body [m^3/s^2]
   * @param mass Spacecraft mass [kg]
   */
  SpacecraftTwobody(double timestep, double mu, double mass);

  /**
   * @brief Computes the continuous-time dynamics
   *
   * @param state Current state vector
   * @param control Current control input
   * @return State derivative vector
   */
  Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &control) const override;

  /**
   * @brief Computes the discrete-time dynamics
   *
   * @param state Current state vector
   * @param control Current control input
   * @return Next state vector
   */
  Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control) const override {
    return DynamicalSystem::getDiscreteDynamics(state, control);
  }

  /**
   * @brief Computes the state Jacobian matrix (∂f/∂x)
   *
   * @param state Current state vector
   * @param control Current control input
   * @return State Jacobian matrix
   */
  Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control) const override;

  /**
   * @brief Computes the control Jacobian matrix (∂f/∂u)
   *
   * @param state Current state vector
   * @param control Current control input
   * @return Control Jacobian matrix
   */
  Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                      const Eigen::VectorXd &control) const override;

  /**
   * @brief Computes the state Hessian matrix (∂²f/∂x²)
   *
   * @param state Current state vector
   * @param control Current control input
   * @return State Hessian matrix
   */
  Eigen::MatrixXd getStateHessian(const Eigen::VectorXd &state,
                                   const Eigen::VectorXd &control) const override;

  /**
   * @brief Computes the control Hessian matrix (∂²f/∂u²)
   *
   * @param state Current state vector
   * @param control Current control input
   * @return Control Hessian matrix
   */
  Eigen::MatrixXd getControlHessian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control) const override;

private:
  // State indices
  static constexpr int STATE_X = 0;    ///< x-position index
  static constexpr int STATE_Y = 1;    ///< y-position index
  static constexpr int STATE_Z = 2;    ///< z-position index
  static constexpr int STATE_VX = 3;   ///< x-velocity index
  static constexpr int STATE_VY = 4;   ///< y-velocity index
  static constexpr int STATE_VZ = 5;   ///< z-velocity index
  static constexpr int STATE_DIM = 6; ///< State dimension

  // Control indices
  static constexpr int CONTROL_UX = 0; ///< x-thrust index
  static constexpr int CONTROL_UY = 1; ///< y-thrust index
  static constexpr int CONTROL_UZ = 2; ///< z-thrust index
  static constexpr int CONTROL_DIM = 3; ///< Control dimension

  // System parameters
  double mu_;   ///< Gravitational parameter [m^3/s^2]
  double mass_; ///< Spacecraft mass [kg]
};

} // namespace cddp

#endif // SPACECRAFT_TWOBODY_HPP