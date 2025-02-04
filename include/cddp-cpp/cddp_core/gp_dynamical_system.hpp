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

#ifndef CDDP_GP_DYNAMICAL_SYSTEM_HPP
#define CDDP_GP_DYNAMICAL_SYSTEM_HPP

#include "cddp_core/dynamical_system.hpp"
#include <Eigen/Dense>
#include <string>

namespace cddp {

class GaussianProcessDynamics : public DynamicalSystem {
public:
    /**
     * Constructor.
     *
     * @param state_dim       Dimension of state.
     * @param control_dim     Dimension of control.
     * @param timestep        Time step (dt).
     * @param integration_type String indicating the integration method (e.g., "euler").
     * @param is_continuous   True if the GP is trained to model continuous dynamics (derivatives);
     *                        false if the GP models discrete dynamics.
     * @param length_scale    Length scale for the squared-exponential kernel.
     * @param signal_variance Signal variance for the kernel.
     * @param noise_variance  Noise variance (likelihood variance).
     */
    GaussianProcessDynamics(int state_dim, int control_dim, double timestep, std::string integration_type,
                            bool is_continuous = true,
                            double length_scale = 1.0,
                            double signal_variance = 1.0,
                            double noise_variance = 1e-4);
    
    virtual ~GaussianProcessDynamics();

    /**
     * Train the GP by storing training data and precomputing the kernel matrix inverse.
     * @param X_train  Training inputs: Each row is a concatenated [state; control] vector.
     * @param Y_train  Training outputs: Each row is the corresponding label (either next state or derivative).
     */
    void train(const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& Y_train);

    /**
     * Predict the dynamics for a given input x = [state; control].
     *
     * @param x Input vector (state and control concatenated).
     * @return  Predicted output vector.
     */
    Eigen::VectorXd predict(const Eigen::VectorXd& x) const;

    virtual Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control) const override;
    
    virtual Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& control) const override;

    virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                             const Eigen::VectorXd& control) const override;
    
    virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control) const override;
    
    virtual Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state,
                                            const Eigen::VectorXd& control) const override;
    
    virtual Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state,
                                              const Eigen::VectorXd& control) const override;
    
    virtual Eigen::MatrixXd getCrossHessian(const Eigen::VectorXd& state,
                                            const Eigen::VectorXd& control) const override;
                                            
    const Eigen::MatrixXd& getKInv() const { return K_inv_; }

private:
    // Squared-exponential kernel function.
    double kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const;

    // Compute the kernel vector between input x and each training input.
    Eigen::VectorXd computeKernelVector(const Eigen::VectorXd& x) const;

    bool is_continuous_;       // True if GP outputs continuous dynamics (derivatives).
    double length_scale_;      // Length scale hyperparameter.
    double signal_variance_;   // Signal variance hyperparameter.
    double noise_variance_;    // Noise variance.

    // Training data: rows correspond to data points.
    Eigen::MatrixXd X_train_;
    Eigen::MatrixXd Y_train_;
    // Precomputed inverse of (K + noise_variance * I).
    Eigen::MatrixXd K_inv_;

    bool trained_; // Flag indicating whether training data has been set.
};

} // namespace cddp

#endif // CDDP_GP_DYNAMICAL_SYSTEM_HPP
