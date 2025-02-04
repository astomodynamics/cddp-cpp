/*
 * train_gp_dynamical_system.cpp
 *
 * This program trains a Gaussian Process (GP) dynamics model using data loaded from a CSV file.
 *
 * The CSV file is assumed to have a header row and each subsequent row contains:
 *    theta, theta_dot, control, label1, label2
 *
 * Here:
 *   - The input is formed by concatenating the state (theta, theta_dot) and the control.
 *   - The output (label) is either the continuous dynamics (derivative) or the discrete next state.
 *
 * Usage:
 *    $ ./train_gp_dynamical_system dataset.csv [continuous|discrete]
 *
 * If the second argument is not provided, the label type defaults to "continuous".
 *
 * After training, the model parameters (hyperparameters, training data, and precomputed kernel inverse)
 * are saved to "gp_model_parameters.txt".
 *
 * Copyright 2024 Tomo Sasaki
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include "cddp_core/gp_dynamical_system.hpp"  // Ensure that this header provides GaussianProcessDynamics

using namespace cddp;

int main(int argc, char* argv[]) {
    // Check and parse command-line arguments.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " dataset.csv [continuous|discrete]" << std::endl;
        return 1;
    }
    std::string csv_file = argv[1];
    std::string label_type = (argc >= 3) ? argv[2] : "continuous";
    bool is_continuous = (label_type == "continuous");

    // For this example we assume:
    const int state_dim = 2;     // e.g., [theta, theta_dot]
    const int control_dim = 1;   // e.g., torque
    const int output_dim = 2;    // same as state_dim (either derivative or next state)

    // --- Load CSV Data ---
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Could not open CSV file: " << csv_file << std::endl;
        return 1;
    }

    std::string line;
    // Skip header line.
    if (!std::getline(file, line)) {
        std::cerr << "CSV file appears to be empty!" << std::endl;
        return 1;
    }

    // Read data rows.
    std::vector< std::vector<double> > raw_data;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }
        // Expect at least 5 columns: state_dim + control_dim + output_dim = 2 + 1 + 2 = 5.
        if (row.size() >= (state_dim + control_dim + output_dim)) {
            raw_data.push_back(row);
        }
    }
    file.close();

    const int n_samples = raw_data.size();
    if (n_samples == 0) {
        std::cerr << "No valid data found in CSV file." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << n_samples << " samples from " << csv_file << std::endl;

    // Create Eigen matrices for training data.
    Eigen::MatrixXd X_train(n_samples, state_dim + control_dim);
    Eigen::MatrixXd Y_train(n_samples, output_dim);

    for (int i = 0; i < n_samples; ++i) {
        // First (state_dim + control_dim) columns form the input.
        for (int j = 0; j < state_dim + control_dim; ++j) {
            X_train(i, j) = raw_data[i][j];
        }
        // Next output_dim columns form the output (label).
        for (int j = 0; j < output_dim; ++j) {
            Y_train(i, j) = raw_data[i][state_dim + control_dim + j];
        }
    }

    // --- Set GP Hyperparameters ---
    double dt = 0.02;  // time step (used to compute discrete dynamics if needed)
    double length_scale = 1.0;
    double signal_variance = 1.0;
    double noise_variance = 1e-4;

    // Create the Gaussian Process dynamics model.
    // The integration method is set to "euler" here; it is used if discrete predictions are requested.
    GaussianProcessDynamics gp_model(state_dim, control_dim, dt, "euler",
                                     is_continuous, length_scale, signal_variance, noise_variance);

    // Train the GP model (i.e. store the training data and precompute the inverted kernel matrix).
    gp_model.train(X_train, Y_train);
    std::cout << "GP model training complete with " << n_samples << " samples." << std::endl;

    // --- Test the GP Model on the First Sample ---
    Eigen::VectorXd test_state = X_train.row(0).head(state_dim).transpose();
    Eigen::VectorXd test_control = X_train.row(0).segment(state_dim, control_dim).transpose();

    Eigen::VectorXd predicted_continuous = gp_model.getContinuousDynamics(test_state, test_control);
    Eigen::VectorXd predicted_discrete = gp_model.getDiscreteDynamics(test_state, test_control);

    std::cout << "Test Sample:" << std::endl;
    std::cout << "  State:   " << test_state.transpose() << std::endl;
    std::cout << "  Control: " << test_control.transpose() << std::endl;
    if (is_continuous) {
        std::cout << "Predicted derivative (continuous dynamics): " << predicted_continuous.transpose() << std::endl;
        std::cout << "Predicted next state (via Euler integration): " << predicted_discrete.transpose() << std::endl;
    } else {
        std::cout << "Predicted next state (discrete dynamics): " << predicted_discrete.transpose() << std::endl;
        std::cout << "Approximate derivative: " << predicted_continuous.transpose() << std::endl;
    }

    // --- Save Model Parameters to File ---
    // Here we save the hyperparameters, the training data, and the precomputed kernel inverse.
    std::ofstream model_out("gp_model_parameters.txt");
    if (!model_out.is_open()) {
        std::cerr << "Failed to open file for saving model parameters." << std::endl;
        return 1;
    }
    
    // Save hyperparameters.
    model_out << "is_continuous " << is_continuous << "\n";
    model_out << "dt " << dt << "\n";
    model_out << "integration_type euler\n";
    model_out << "length_scale " << length_scale << "\n";
    model_out << "signal_variance " << signal_variance << "\n";
    model_out << "noise_variance " << noise_variance << "\n";
    
    // Save training data dimensions.
    model_out << "X_train " << n_samples << " " << (state_dim + control_dim) << "\n";
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < state_dim + control_dim; ++j) {
            model_out << X_train(i, j) << " ";
        }
        model_out << "\n";
    }
    model_out << "Y_train " << n_samples << " " << output_dim << "\n";
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            model_out << Y_train(i, j) << " ";
        }
        model_out << "\n";
    }
    
    // Save the precomputed kernel inverse.
    // This assumes that your GaussianProcessDynamics class provides a method getKInv()
    // which returns a const reference to the Eigen::MatrixXd containing the inverse.
    const Eigen::MatrixXd& K_inv = gp_model.getKInv();
    model_out << "K_inv " << K_inv.rows() << " " << K_inv.cols() << "\n";
    for (int i = 0; i < K_inv.rows(); ++i) {
        for (int j = 0; j < K_inv.cols(); ++j) {
            model_out << K_inv(i, j) << " ";
        }
        model_out << "\n";
    }
    
    model_out.close();
    std::cout << "Model parameters saved to gp_model_parameters.txt" << std::endl;

    return 0;
}
