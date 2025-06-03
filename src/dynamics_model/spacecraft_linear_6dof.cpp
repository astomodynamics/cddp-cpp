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

#include "cddp-cpp/dynamics_model/spacecraft_linear_6dof.hpp"
#include <autodiff/forward/dual.hpp>       // For dual types
#include <autodiff/forward/dual/eigen.hpp> // For Eigen support with dual types
#include <cmath> // For std::abs in helper potentially, though autodiff might have its own

namespace cddp {

SpacecraftLinear6DOF::SpacecraftLinear6DOF(
    double timestep, double mean_motion, double mass,
    const Eigen::Matrix3d& inertia_matrix,
    std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      hcw_model_(timestep, mean_motion, mass, integration_type), // Initialize HCW model
      mrp_attitude_model_(timestep, inertia_matrix, integration_type) // Initialize MRP model
{
    // Constructor body (if needed)
}

Eigen::VectorXd SpacecraftLinear6DOF::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const 
{
    // Extract MRP state (sigma)
    Eigen::Vector3d mrp_sigma = state.segment<3>(STATE_MRP_X);

    // Extract body-frame forces and torques from control input
    Eigen::Vector3d F_body = control.segment<3>(CONTROL_BODY_FX);
    Eigen::Vector3d tau_body = control.segment<3>(CONTROL_BODY_TAU_X);

    // Compute DCM from body to LVLH frame
    Eigen::Matrix3d C_lvlh_body = mrpToDcmBodyToLVLH<double>(mrp_sigma);

    // Transform body forces to LVLH forces
    Eigen::Vector3d F_lvlh = C_lvlh_body * F_body;

    // HCW model uses forces in LVLH frame
    Eigen::VectorXd hcw_control = F_lvlh;
    // MrpAttitude model uses torques in body frame
    Eigen::VectorXd mrp_control = tau_body;

    // Extract states for HCW model
    Eigen::VectorXd hcw_state = state.segment(0, HCW::STATE_DIM);
    // Extract states for MRP Attitude model
    Eigen::VectorXd mrp_attitude_state = state.segment(HCW::STATE_DIM, MrpAttitude::STATE_DIM);

    // Compute dynamics for each subsystem
    Eigen::VectorXd hcw_state_dot = hcw_model_.getContinuousDynamics(hcw_state, hcw_control, time);
    Eigen::VectorXd mrp_state_dot = mrp_attitude_model_.getContinuousDynamics(mrp_attitude_state, mrp_control, time);

    // Combine the state derivatives
    Eigen::VectorXd combined_state_dot(STATE_DIM);
    combined_state_dot.segment(0, HCW::STATE_DIM) = hcw_state_dot;
    combined_state_dot.segment(HCW::STATE_DIM, MrpAttitude::STATE_DIM) = mrp_state_dot;

    return combined_state_dot;
}

VectorXdual2nd SpacecraftLinear6DOF::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const 
{
    // Extract MRP state (sigma) as autodiff type
    autodiff::Vector3dual2nd mrp_sigma_ad = state.segment<3>(STATE_MRP_X);

    // Extract body-frame forces and torques from control input as autodiff type
    autodiff::Vector3dual2nd F_body_ad = control.segment<3>(CONTROL_BODY_FX);
    autodiff::Vector3dual2nd tau_body_ad = control.segment<3>(CONTROL_BODY_TAU_X);

    // Compute DCM from body to LVLH frame using autodiff types
    autodiff::Matrix3dual2nd C_lvlh_body_ad = mrpToDcmBodyToLVLH<autodiff::dual2nd>(mrp_sigma_ad);

    // Transform body forces to LVLH forces using autodiff types
    autodiff::Vector3dual2nd F_lvlh_ad = C_lvlh_body_ad * F_body_ad;

    // HCW model uses forces in LVLH frame (autodiff type)
    VectorXdual2nd hcw_control_ad = F_lvlh_ad;
    // MrpAttitude model uses torques in body frame (autodiff type)
    VectorXdual2nd mrp_control_ad = tau_body_ad;

    // Extract states for HCW model (autodiff types)
    VectorXdual2nd hcw_state_ad = state.segment(0, HCW::STATE_DIM);
    // Extract states for MRP Attitude model (autodiff types)
    VectorXdual2nd mrp_attitude_state_ad = state.segment(HCW::STATE_DIM, MrpAttitude::STATE_DIM);

    // Compute dynamics for each subsystem using autodiff
    VectorXdual2nd hcw_state_dot_ad = hcw_model_.getContinuousDynamicsAutodiff(hcw_state_ad, hcw_control_ad, time);
    VectorXdual2nd mrp_state_dot_ad = mrp_attitude_model_.getContinuousDynamicsAutodiff(mrp_attitude_state_ad, mrp_control_ad, time);

    // Combine the state derivatives (autodiff types)
    VectorXdual2nd combined_state_dot_ad(STATE_DIM);
    combined_state_dot_ad.segment(0, HCW::STATE_DIM) = hcw_state_dot_ad;
    combined_state_dot_ad.segment(HCW::STATE_DIM, MrpAttitude::STATE_DIM) = mrp_state_dot_ad;

    return combined_state_dot_ad;
}

} // namespace cddp 