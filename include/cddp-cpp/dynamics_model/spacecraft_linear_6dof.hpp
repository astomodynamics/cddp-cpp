/*
 Copyright 2024 Tomo Sasaki (and Gemini Assistant)

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#ifndef CDDP_SPACECRAFT_LINEAR_6DOF_HPP
#define CDDP_SPACECRAFT_LINEAR_6DOF_HPP

#include "cddp_core/dynamical_system.hpp"
#include "cddp-cpp/dynamics_model/spacecraft_linear.hpp" // HCW model
#include "cddp-cpp/dynamics_model/mrp_attitude.hpp"      // MRP Attitude model
#include <autodiff/common/functions.hpp> // For autodiff::abs

namespace cddp {

class SpacecraftLinear6DOF : public DynamicalSystem {
public:
    // Combined state and control dimensions
    static constexpr int STATE_DIM = HCW::STATE_DIM + MrpAttitude::STATE_DIM;       // 6 + 6 = 12
    static constexpr int CONTROL_DIM = HCW::CONTROL_DIM + MrpAttitude::CONTROL_DIM; // 3 + 3 = 6 (Forces in Body Frame, Torques in Body Frame)

    // State indices for HCW part
    static constexpr int STATE_HCW_X = 0;
    static constexpr int STATE_HCW_Y = 1;
    static constexpr int STATE_HCW_Z = 2;
    static constexpr int STATE_HCW_VX = 3;
    static constexpr int STATE_HCW_VY = 4;
    static constexpr int STATE_HCW_VZ = 5;

    // State indices for MRP part (offset by HCW::STATE_DIM)
    static constexpr int STATE_MRP_X = HCW::STATE_DIM + MrpAttitude::STATE_MRP_X;
    static constexpr int STATE_MRP_Y = HCW::STATE_DIM + MrpAttitude::STATE_MRP_Y;
    static constexpr int STATE_MRP_Z = HCW::STATE_DIM + MrpAttitude::STATE_MRP_Z;
    static constexpr int STATE_OMEGA_X = HCW::STATE_DIM + MrpAttitude::STATE_OMEGA_X;
    static constexpr int STATE_OMEGA_Y = HCW::STATE_DIM + MrpAttitude::STATE_OMEGA_Y;
    static constexpr int STATE_OMEGA_Z = HCW::STATE_DIM + MrpAttitude::STATE_OMEGA_Z;

    // Control indices (Forces and Torques in Spacecraft Body Frame)
    static constexpr int CONTROL_BODY_FX = 0;    // Force along body X-axis
    static constexpr int CONTROL_BODY_FY = 1;    // Force along body Y-axis
    static constexpr int CONTROL_BODY_FZ = 2;    // Force along body Z-axis
    static constexpr int CONTROL_BODY_TAU_X = 3; // Torque around body X-axis
    static constexpr int CONTROL_BODY_TAU_Y = 4; // Torque around body Y-axis
    static constexpr int CONTROL_BODY_TAU_Z = 5; // Torque around body Z-axis


    /**
     * Constructor for the 6DOF Spacecraft model
     * @param timestep Time step for discretization
     * @param mean_motion Mean motion for HCW model
     * @param mass Mass for HCW model
     * @param inertia_matrix Inertia matrix for MRP Attitude model
     * @param integration_type Integration method (e.g., "euler", "rk4")
     */
    SpacecraftLinear6DOF(double timestep, double mean_motion, double mass,
                   const Eigen::Matrix3d& inertia_matrix,
                   std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the 6DOF model
     * State vector: [hcw_states, mrp_states]
     * Control vector: [hcw_controls, mrp_controls]
     * @param state Current combined state vector
     * @param control Current combined control input
     * @param time Current time
     * @return Combined state derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                         const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the discrete-time dynamics using the specified integration method
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control, time);
    }

    /**
     * Computes the Jacobian of the dynamics with respect to the state (delegates to base class using Autodiff)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getStateJacobian(state, control, time);
    }

    /**
     * Computes the Jacobian of the dynamics with respect to the control input (delegates to base class using Autodiff)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getControlJacobian(state, control, time);
    }

    /**
     * Computes the Hessian of the dynamics with respect to the state (delegates to base class using Autodiff)
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getStateHessian(state, control, time);
    }

    /**
     * Computes the Hessian of the dynamics with respect to the control (delegates to base class using Autodiff)
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getControlHessian(state, control, time);
    }

    /**
     * Computes the cross Hessian of the dynamics (delegates to base class using Autodiff)
     */
    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control, double time) const override {
         return DynamicalSystem::getCrossHessian(state, control, time);
    }


    /**
     * Computes the continuous-time dynamics of the 6DOF model using autodiff
     * @param state Current combined state vector (autodiff type)
     * @param control Current combined control input (autodiff type)
     * @param time Current time
     * @return Combined state derivative vector (autodiff type)
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd& state,
                                                const VectorXdual2nd& control, double time) const override;

private:
    HCW hcw_model_;               // Instance of the HCW model
    MrpAttitude mrp_attitude_model_; // Instance of the MRP Attitude model

    // Helper function for skew-symmetric matrix
    template <typename T>
    static Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& v) {
        Eigen::Matrix<T, 3, 3> S;
        S << T(0), -v(2), v(1),
             v(2),  T(0), -v(0),
            -v(1),  v(0),  T(0);
        return S;
    }

    // Helper function to get DCM from MRP (Body to LVLH/Nav Frame)
    // C_lvlh_body = C_N_B in Schaub & Junkins notation
    template <typename T>
    static Eigen::Matrix<T, 3, 3> mrpToDcmBodyToLVLH(const Eigen::Matrix<T, 3, 1>& mrp) {
        T mrp_sq_norm = mrp.squaredNorm();
        Eigen::Matrix<T, 3, 3> S = skew(mrp);
        Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();

        // Using autodiff::abs for compatibility with dual types
        if (autodiff::abs(T(1.0) + mrp_sq_norm) < T(1e-9)) { 
             // This case (1+s^2 approx 0) should not happen for real MRPs (s^2 >=0)
             // but as a safeguard for numerical stability with dual types if somehow 1+s^2 becomes very small.
             // For MRPs, 1+s^2 is always >= 1.
             // If it was (1-s^2) in denominator, this would be for Gibb's params.
             // Let's stick to the formula assuming (1+s^2) is not zero.
        }
        
        T den = (T(1.0) + mrp_sq_norm) * (T(1.0) + mrp_sq_norm);
        // Using autodiff::abs for compatibility with dual types
        if (autodiff::abs(den) < T(1e-12)) { // If denominator is zero
            // This implies 1+mrp_sq_norm is zero, which is impossible for real mrp_sq_norm >= 0.
            // Return identity or handle error appropriately. For now, assume valid MRPs.
            // For safety, return identity matrix in an unlikely singular case.
            return I;
        }
        
        // C_N_B = I + (-4*(1-sigma^2)*S + 8*S^2) / (1+sigma^2)^2
        // (Schaub & Junkins, Eq. 3.161, C_N_B(q) where q is MRP)
        return I + (T(-4.0) * (T(1.0) - mrp_sq_norm) * S + T(8.0) * S * S) / den;
    }
};

} // namespace cddp

#endif // CDDP_SPACECRAFT_LINEAR_6DOF_HPP 