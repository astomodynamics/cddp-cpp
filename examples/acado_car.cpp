/*
 * Copyright 2024 Tomo Sasaki
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <acado/acado_toolkit.hpp>
#include <acado/acado_optimal_control.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace ACADO;


int main() {
    // Problem parameters
    const int state_dim   = 4;    // [x, y, theta, v]
    const int control_dim = 2;    // [steering angle, acceleration]
    const int horizon     = 500;  // Discretization steps
    const double timestep = 0.03;
    const double T        = horizon * timestep; 
    const double wheelbase = 2.0;

    // Define differential states and controls
    DifferentialState x, y, theta, v;
    Control delta, a;

    // Define the dynamic system (differential equation)
    DifferentialEquation f(0.0, T);
    f << dot(x) == v * cos(theta);
    f << dot(y) == v * sin(theta);
    f << dot(theta) == v * tan(delta) / wheelbase;
    f << dot(v) == a;

    // Set up OCP
    OCP ocp(0.0, T, horizon);

    // Subject to the system dynamics
    ocp.subjectTo( f );

    // Initial conditions (at t=0)
    ocp.subjectTo( AT_START, x     == 1.0 );
    ocp.subjectTo( AT_START, y     == 1.0 );
    ocp.subjectTo( AT_START, theta == 1.5 * M_PI );
    ocp.subjectTo( AT_START, v     == 0.0 );

    // (Loose) state bounds
    ocp.subjectTo( -1e20 <= x <= 1e20 );
    ocp.subjectTo( -1e20 <= y <= 1e20 );
    ocp.subjectTo( -1e20 <= theta <= 1e20 );
    ocp.subjectTo( -1e20 <= v <= 1e20 );

    // Control bounds (steering angle and acceleration)
    ocp.subjectTo( -0.5 <= delta <= 0.5 );
    ocp.subjectTo( -2.0 <= a     <= 2.0 );

    // Define the cost (objective) terms.
    ocp.minimizeLagrangeTerm( 1e-2 * delta * delta + 1e-4 * a * a + 1e-3 * (x*x + y*y) );

    // Terminal cost
    ocp.minimizeMayerTerm( 0.1 * (x*x + y*y) + 1.0 * (theta*theta) + 0.3 * (v*v) );

    // Set up the optimization algorithm and options
    OptimizationAlgorithm algorithm(ocp);
    algorithm.set( MAX_NUM_ITERATIONS, 1000 );

    // Solve the OCP while measuring solve time
    auto start_time = std::chrono::high_resolution_clock::now();
    algorithm.solve();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Solve time: " << duration.count() << " microseconds" << std::endl;

    // Retrieve the solution: state and control trajectories
    VariablesGrid stateGrid, controlGrid;
    algorithm.getDifferentialStates(stateGrid);
    algorithm.getControls(controlGrid);

    std::cout << "Solution retrieved: " 
              << stateGrid.getNumPoints() << " states, " 
              << controlGrid.getNumPoints() << " controls." << std::endl;

    return 0;
}
