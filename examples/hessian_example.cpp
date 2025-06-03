/*
 Copyright 2024 Tomo Sasaki

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

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <iomanip>

#include "dynamics_model/pendulum.hpp"
#include "dynamics_model/dubins_car.hpp"

using namespace cddp;

// Helper function to print a matrix with a label
void printMatrix(const std::string& label, const Eigen::MatrixXd& matrix) {
    std::cout << label << " (" << matrix.rows() << "x" << matrix.cols() << ")" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << matrix << std::endl << std::endl;
}

// Helper function to print the Hessian tensor
void printHessianTensor(const std::string& label, const std::vector<Eigen::MatrixXd>& hessian) {
    std::cout << label << " (Tensor with " << hessian.size() << " matrices)" << std::endl;
    
    for (size_t i = 0; i < hessian.size(); ++i) {
        std::cout << "Matrix for state dimension " << i << " (" 
                  << hessian[i].rows() << "x" << hessian[i].cols() << "):" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << hessian[i] << std::endl;
    }
    std::cout << std::endl;
}

// Function to demonstrate pendulum Hessian
void demonstratePendulumHessian() {
    std::cout << "\n========== PENDULUM MODEL EXAMPLE ==========" << std::endl;
    
    // Create a pendulum model
    double timestep = 0.01;
    double length = 1.0;     // Length of the pendulum [m]
    double mass = 1.0;       // Mass [kg]
    double damping = 0.1;    // Damping coefficient
    std::string integration = "rk4";
    
    Pendulum pendulum(timestep, length, mass, damping, integration);
    
    // Define a state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(2);
    state << M_PI / 4.0, 0.0;  // 45-degree angle, zero velocity
    
    Eigen::VectorXd control = Eigen::VectorXd::Zero(1);
    control << 1.0;  // Apply a torque of 1 Nm
    
    // Print system information
    std::cout << "Pendulum parameters:" << std::endl;
    std::cout << "Length: " << pendulum.getLength() << " m" << std::endl;
    std::cout << "Mass: " << pendulum.getMass() << " kg" << std::endl;
    std::cout << "Damping: " << pendulum.getDamping() << std::endl;
    std::cout << "Gravity: " << pendulum.getGravity() << " m/s²" << std::endl;
    std::cout << "Timestep: " << pendulum.getTimestep() << " s" << std::endl;
    std::cout << "Integration: " << pendulum.getIntegrationType() << std::endl << std::endl;
    
    // Print state and control
    std::cout << "State: [theta, theta_dot] = [" << state.transpose() << "]" << std::endl;
    std::cout << "Control: [torque] = [" << control.transpose() << "]" << std::endl << std::endl;
    
    // Compute dynamics
    Eigen::VectorXd xdot = pendulum.getContinuousDynamics(state, control, 0.0);
    std::cout << "Continuous Dynamics (xdot): [" << xdot.transpose() << "]" << std::endl;
    
    Eigen::VectorXd next_state = pendulum.getDiscreteDynamics(state, control, 0.0);
    std::cout << "Discrete Dynamics (next state): [" << next_state.transpose() << "]" << std::endl << std::endl;
    
    // Compute Jacobians
    Eigen::MatrixXd A = pendulum.getStateJacobian(state, control, 0.0);
    Eigen::MatrixXd B = pendulum.getControlJacobian(state, control, 0.0);
    
    printMatrix("State Jacobian (A)", A);
    printMatrix("Control Jacobian (B)", B);
    
    // Compute Hessians
    std::vector<Eigen::MatrixXd> state_hessian = pendulum.getStateHessian(state, control, 0.0);
    std::vector<Eigen::MatrixXd> control_hessian = pendulum.getControlHessian(state, control, 0.0);
    std::vector<Eigen::MatrixXd> cross_hessian = pendulum.getCrossHessian(state, control, 0.0);
    
    printHessianTensor("State Hessian (d²f/dx²)", state_hessian);
    printHessianTensor("Control Hessian (d²f/du²)", control_hessian);
    printHessianTensor("Cross Hessian (d²f/dudx)", cross_hessian);
    
    // Demonstrate how to access specific elements of the Hessian
    // For example, accessing d²theta_dot/dtheta² (second derivative of theta_dot with respect to theta)
    int state_idx = 1;  // theta_dot is state index 1
    int wrt_state_idx1 = 0;  // first derivative with respect to theta (index 0)
    int wrt_state_idx2 = 0;  // second derivative with respect to theta (index 0)
    
    double d2f_dx2 = state_hessian[state_idx](wrt_state_idx1, wrt_state_idx2);
    std::cout << "d²theta_dot/dtheta² = " << d2f_dx2 << std::endl;
}

// Function to demonstrate Dubins car Hessian
void demonstrateDubinsCarHessian() {
    std::cout << "\n========== DUBINS CAR MODEL EXAMPLE ==========" << std::endl;
    
    // Create a Dubins car model
    double speed = 1.0;      // Constant forward speed [m/s]
    double timestep = 0.01;  // Time step [s]
    std::string integration = "rk4";
    
    DubinsCar dubins_car(speed, timestep, integration);
    
    // Define a state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(3);
    state << 0.0, 0.0, M_PI / 4.0;  // (x, y, theta) = (0, 0, 45°)
    
    Eigen::VectorXd control = Eigen::VectorXd::Zero(1);
    control << 0.5;  // Turn rate of 0.5 rad/s
    
    // Print system information
    std::cout << "Dubins Car parameters:" << std::endl;
    std::cout << "Speed: " << speed << " m/s" << std::endl;
    std::cout << "Timestep: " << dubins_car.getTimestep() << " s" << std::endl;
    std::cout << "Integration: " << dubins_car.getIntegrationType() << std::endl << std::endl;
    
    // Print state and control
    std::cout << "State: [x, y, theta] = [" << state.transpose() << "]" << std::endl;
    std::cout << "Control: [omega] = [" << control.transpose() << "]" << std::endl << std::endl;
    
    // Compute dynamics
    Eigen::VectorXd xdot = dubins_car.getContinuousDynamics(state, control, 0.0);
    std::cout << "Continuous Dynamics (xdot): [" << xdot.transpose() << "]" << std::endl;
    
    Eigen::VectorXd next_state = dubins_car.getDiscreteDynamics(state, control, 0.0);
    std::cout << "Discrete Dynamics (next state): [" << next_state.transpose() << "]" << std::endl << std::endl;
    
    // Compute Jacobians
    Eigen::MatrixXd A = dubins_car.getStateJacobian(state, control, 0.0);
    Eigen::MatrixXd B = dubins_car.getControlJacobian(state, control, 0.0);
    
    printMatrix("State Jacobian (A)", A);
    printMatrix("Control Jacobian (B)", B);
    
    // Compute Hessians
    std::vector<Eigen::MatrixXd> state_hessian = dubins_car.getStateHessian(state, control, 0.0);
    std::vector<Eigen::MatrixXd> control_hessian = dubins_car.getControlHessian(state, control, 0.0);
    std::vector<Eigen::MatrixXd> cross_hessian = dubins_car.getCrossHessian(state, control, 0.0);
    
    printHessianTensor("State Hessian (d²f/dx²)", state_hessian);
    printHessianTensor("Control Hessian (d²f/du²)", control_hessian);
    printHessianTensor("Cross Hessian (d²f/dudx)", cross_hessian);
    
    // Demonstrate how to access specific elements of the Hessian
    // For example, accessing d²x/dtheta² (second derivative of x with respect to theta)
    int state_idx = 0;  // x is state index 0
    int wrt_state_idx1 = 2;  // first derivative with respect to theta (index 2)
    int wrt_state_idx2 = 2;  // second derivative with respect to theta (index 2)
    
    double d2f_dx2 = state_hessian[state_idx](wrt_state_idx1, wrt_state_idx2);
    std::cout << "d²x/dtheta² = " << d2f_dx2 << std::endl;
}

int main() {
    std::cout << "Hessian Examples for Dynamical Systems" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Demonstrate pendulum Hessian
    demonstratePendulumHessian();
    
    // Demonstrate Dubins car Hessian
    demonstrateDubinsCarHessian();
    return 0;
} 