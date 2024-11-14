#ifndef CDDP_BICYCLE_HPP
#define CDDP_BICYCLE_HPP

#include "Eigen/Dense"
#include <vector>
#include "cddp_core/DynamicalSystem.hpp"

namespace cddp {

class Bicycle : public cddp::DynamicalSystem {
public:
    int state_size_;     // State dimension (x, y, theta, v)
    int control_size_;   // Control dimension (a, delta)
    double timestep_;    // Time step
    double wheelbase_;   // Distance between front and rear axles
    int integration_type_;

    Bicycle(int state_dim, int control_dim, double timestep, double wheelbase, int integration_type) :
        DynamicalSystem(state_dim, control_dim, timestep, integration_type) {
            state_size_ = state_dim;      // Should be 4: [x, y, theta, v]
            control_size_ = control_dim;   // Should be 2: [acceleration, steering_angle]
            timestep_ = timestep;
            wheelbase_ = wheelbase;
            integration_type_ = integration_type;
        }

    Eigen::VectorXd dynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        // State: [x, y, theta, v]
        // Control: [acceleration, steering_angle]
        Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(state_size_);
        
        double v = state(3);          // velocity
        double theta = state(2);      // heading angle
        double delta = control(1);    // steering angle
        double a = control(0);        // acceleration

        // Kinematic bicycle model equations
        state_dot(0) = v * cos(theta);                    // dx/dt
        state_dot(1) = v * sin(theta);                    // dy/dt
        state_dot(2) = (v / wheelbase_) * tan(delta);     // dtheta/dt
        state_dot(3) = a;                                 // dv/dt

        return state_dot;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsJacobian(
            const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        // Initialize Jacobian matrices
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(state_size_, state_size_);   // df/dx
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_); // df/du

        double v = state(3);          // velocity
        double theta = state(2);      // heading angle
        double delta = control(1);    // steering angle

        // State Jacobian (A matrix)
        // df1/dx = d(dx/dt)/dx
        A(0, 2) = -v * sin(theta);    // df1/dtheta
        A(0, 3) = cos(theta);         // df1/dv

        // df2/dx = d(dy/dt)/dx
        A(1, 2) = v * cos(theta);     // df2/dtheta
        A(1, 3) = sin(theta);         // df2/dv

        // df3/dx = d(dtheta/dt)/dx
        A(2, 3) = tan(delta) / wheelbase_;  // df3/dv

        // Control Jacobian (B matrix)
        // df/du
        B(3, 0) = 1.0;   // df4/da (acceleration effect on velocity)
        B(2, 1) = v / (wheelbase_ * pow(cos(delta), 2));  // df3/ddelta

        return std::make_tuple(A, B);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsHessian(
            const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        // Initialize Hessian matrices
        Eigen::MatrixXd hxx = Eigen::MatrixXd::Zero(state_size_ * state_size_, state_size_);
        Eigen::MatrixXd hxu = Eigen::MatrixXd::Zero(state_size_ * control_size_, state_size_);
        Eigen::MatrixXd huu = Eigen::MatrixXd::Zero(state_size_ * control_size_, control_size_);

        double v = state(3);          // velocity
        double theta = state(2);      // heading angle
        double delta = control(1);    // steering angle

        // Fill in non-zero Hessian terms
        // Second derivatives with respect to states
        int idx;
        
        // d²(dx/dt)/dtheta²
        idx = 2 * state_size_ + 0;  // (theta, x) component
        hxx(idx, 2) = -v * cos(theta);

        // d²(dy/dt)/dtheta²
        idx = 2 * state_size_ + 1;  // (theta, y) component
        hxx(idx, 2) = -v * sin(theta);

        // Mixed derivatives (state-control)
        // d²(dtheta/dt)/dv/ddelta
        idx = 3 * control_size_ + 1;  // (v, delta) component
        hxu(idx, 2) = 1.0 / (wheelbase_ * pow(cos(delta), 2));

        // Second derivatives with respect to controls
        // d²(dtheta/dt)/ddelta²
        idx = 1 * control_size_ + 2;  // (delta, theta) component
        huu(idx, 1) = 2.0 * v * sin(delta) / (wheelbase_ * pow(cos(delta), 3));

        return std::make_tuple(hxx, hxu, huu);
    }
};

}  // namespace cddp

#endif // CDDP_BICYCLE_HPP