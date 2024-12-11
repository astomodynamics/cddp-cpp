/*
 * Example code demonstrating the car model with CDDP
 */
#include <iostream>
#include <vector>
#include <filesystem>
#include <random>
#include <cmath>
#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

namespace cddp {

class CarParkingObjective : public NonlinearObjective {
public:
    CarParkingObjective(const Eigen::VectorXd& goal_state, double timestep)
        : NonlinearObjective(timestep), reference_state_(goal_state) {
        // Control cost coefficients: cu = 1e-2*[1 .01]
        cu_ = Eigen::Vector2d(1e-2, 1e-4);

        // Final cost coefficients: cf = [.1 .1 1 .3]
        cf_ = Eigen::Vector4d(0.1, 0.1, 1.0, 0.3);

        // Smoothness scales for final cost: pf = [.01 .01 .01 1]
        pf_ = Eigen::Vector4d(0.01, 0.01, 0.01, 1.0);

        // Running cost coefficients: cx = 1e-3*[1 1]
        cx_ = Eigen::Vector2d(1e-3, 1e-3);

        // Smoothness scales for running cost: px = [.1 .1]
        px_ = Eigen::Vector2d(0.1, 0.1);
    }

    double running_cost(const Eigen::VectorXd& state, 
                       const Eigen::VectorXd& control, 
                       int index) const override {
        // Control cost: lu = cu*u.^2
        double lu = cu_.dot(control.array().square().matrix());

        // Running cost on distance from origin: lx = cx*sabs(x(1:2,:),px)
        Eigen::VectorXd xy_state = state.head(2);
        double lx = cx_.dot(sabs(xy_state, px_));

        return lu + lx;
    }

    double terminal_cost(const Eigen::VectorXd& final_state) const override {
        // Final state cost: llf = cf*sabs(x(:,final),pf);
        return cf_.dot(sabs(final_state, pf_)) + running_cost(final_state, Eigen::VectorXd::Zero(2), 0);
    }

    const Eigen::VectorXd& getReferenceState() const override { 
        return reference_state_; 
    }

private:
    // Helper function for smooth absolute value (pseudo-Huber)
    Eigen::VectorXd sabs(const Eigen::VectorXd& x, const Eigen::VectorXd& p) const {
        return ((x.array().square() / p.array().square() + 1.0).sqrt() * p.array() - p.array()).matrix();
    }

    Eigen::VectorXd reference_state_;
    Eigen::Vector2d cu_; // Control cost coefficients
    Eigen::Vector4d cf_; // Final cost coefficients
    Eigen::Vector4d pf_; // Smoothness scales for final cost 
    Eigen::Vector2d cx_; // Running cost coefficients
    Eigen::Vector2d px_; // Smoothness scales for running cost
};
} // namespace cddp

void plotCarBox(const Eigen::VectorXd& state, Eigen::VectorXd& control,
                double length, double width, 
                const std::string& color) {
    double x = state(0);
    double y = state(1);
    double theta = state(2);
    double steering = control(1);
    
    // Compute car corners
    std::vector<double> car_x(5), car_y(5);
    
    // Front right
    car_x[0] = x + length/2 * cos(theta) - width/2 * sin(theta);
    car_y[0] = y + length/2 * sin(theta) + width/2 * cos(theta);
    
    // Front left
    car_x[1] = x + length/2 * cos(theta) + width/2 * sin(theta);
    car_y[1] = y + length/2 * sin(theta) - width/2 * cos(theta);
    
    // Rear left
    car_x[2] = x - length/2 * cos(theta) + width/2 * sin(theta);
    car_y[2] = y - length/2 * sin(theta) - width/2 * cos(theta);
    
    // Rear right
    car_x[3] = x - length/2 * cos(theta) - width/2 * sin(theta);
    car_y[3] = y - length/2 * sin(theta) + width/2 * cos(theta);
    
    // Close the polygon
    car_x[4] = car_x[0];
    car_y[4] = car_y[0];
    
    // Plot car body
    std::map<std::string, std::string> keywords;
    keywords["color"] = color;
    plt::plot(car_x, car_y, keywords);

    // Plot base point (center of rear axle)
    std::vector<double> base_x = {x};
    std::vector<double> base_y = {y};
    keywords["color"] = "red";
    keywords["marker"] = "o";
    plt::plot(base_x, base_y, keywords);

    // Plot steering direction
    // Calculate front axle center point
    double front_x = x + length/2 * cos(theta);
    double front_y = y + length/2 * sin(theta);
    
    // Calculate steering indicator endpoint
    double steering_length = width/2;  // Length of steering indicator line
    double steering_angle = theta + steering;  // Global steering angle
    double steering_end_x = front_x + steering_length * cos(steering_angle);
    double steering_end_y = front_y + steering_length * sin(steering_angle);
    
    // Plot steering indicator
    std::vector<double> steer_x = {front_x, steering_end_x};
    std::vector<double> steer_y = {front_y, steering_end_y};
    keywords["color"] = "green";
    keywords.erase("marker");
    plt::plot(steer_x, steer_y, keywords);
}

int main() {
    // Problem parameters
    int state_dim = 4;     // [x y theta v]
    int control_dim = 2;   // [wheel_angle acceleration]
    int horizon = 500;     // Same as MATLAB example
    double timestep = 0.03;  // h = 0.03 from MATLAB
    std::string integration_type = "euler";

    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.1);

    // Create car instance
    double wheelbase = 2.0;  // d = 2.0 from MATLAB example
    std::unique_ptr<cddp::DynamicalSystem> system = 
        std::make_unique<cddp::Car>(timestep, wheelbase, integration_type);

    // Initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 1.0, 1.0, 1.5*M_PI, 0.0;  // [1; 1; pi*3/2; 0] from MATLAB

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0, 0.0, 0.0;  // NOT USED IN THIS EXAMPLE

    // Create the nonlinear objective
    auto objective = std::make_unique<cddp::CarParkingObjective>(goal_state, timestep);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -0.5, -2.0;  // [steering_angle, acceleration]
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 0.5, 2.0;

    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(
            control_lower_bound, control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 1;
    options.verbose = true;
    options.cost_tolerance = 1e-7;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "both";
    options.regularization_state = 1.0;
    options.regularization_control = 1.0;
    cddp_solver.setOptions(options);

    // Initialize with random controls
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Random initialization
    for(auto& u : U) {
        // u(0) = d(gen);  // Random steering
        // u(1) = d(gen);  // Random acceleration
        u(0) = 0.01;
        u(1) = 0.01;
    }

    X[0] = initial_state;

    double J = 0.0;

    // Simulate initial trajectory
    for(size_t t = 0; t < horizon; t++) {
        J += cddp_solver.getObjective().running_cost(X[t], U[t], t);
        X[t + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[t], U[t]);
    }
    J += cddp_solver.getObjective().terminal_cost(X.back());
    std::cout << "Initial cost: " << J << std::endl;
    std::cout << "Initial state: " << X[0].transpose() << std::endl;
    std::cout << "Final state: " << X.back().transpose() << std::endl;

    
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    // cddp::CDDPSolution solution = cddp_solver.solve();
    cddp::CDDPSolution solution = cddp_solver.solveCLDDP();
    // cddp::CDDPSolution solution = cddp_solver.solveLogCDDP();

    // Extract solution trajectories
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // // Create directory for plots
    // const std::string plotDirectory = "../results/tests";
    // if (!fs::exists(plotDirectory)) {
    //     fs::create_directories(plotDirectory);
    // }

    // // Prepare visualization data
    // std::vector<double> x_hist, y_hist, theta_hist, v_hist;
    // for(const auto& x : X_sol) {
    //     x_hist.push_back(x(0));
    //     y_hist.push_back(x(1));
    //     theta_hist.push_back(x(2));
    //     v_hist.push_back(x(3));
    // }

    // // Visualization matching MATLAB style
    // plt::figure_size(800, 800);
    
    // // Set plot limits and properties matching MATLAB
    // plt::xlim(-4, 4);
    // plt::ylim(-4, 4);
    // plt::grid(true);
    
    // // Plot goal configuration
    // double car_length = 2.1;  // From MATLAB body = [0.9 2.1 0.3]
    // double car_width = 0.9;   
    // Eigen::VectorXd empty_control = Eigen::VectorXd::Zero(control_dim);
    // plotCarBox(goal_state, empty_control, car_length, car_width, "r");
    
    // // Animation loop
    // for(size_t i = 0; i < X_sol.size(); i++) {
    //     if(i % 5 == 0) {
    //         plt::clf();
            
    //         // Plot full trajectory
    //         plt::plot(x_hist, y_hist, "b-");
            
    //         // Plot current car position
    //         plotCarBox(X_sol[i], U_sol[i], car_length, car_width, "k");
            
    //         // Plot settings
    //         plt::grid(true);
    //         plt::axis("equal");
    //         plt::xlim(-4, 4);
    //         plt::ylim(-4, 4);
            
    //         std::string filename = plotDirectory + "/car_frame_" + 
    //                              std::to_string(i) + ".png";
    //         plt::save(filename);
    //         plt::pause(0.01);
    //     }
    // }

    return 0;
}

// Create animation from frames using ImageMagick:
// convert -delay 50 ../results/tests/car_frame_*.png ../results/tests/car_motion.gif