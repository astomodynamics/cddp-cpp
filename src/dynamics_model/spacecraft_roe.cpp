/**
 * @file    cddp_spacecraft_roe.cpp
 * @author  ...
 * @date    2024
 * @brief   Implementation of SpacecraftROE for QNS-ROE dynamics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "dynamics_model/spacecraft_roe.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace cddp
{

    SpacecraftROE::SpacecraftROE(
        double timestep,
        const std::string &integration_type,
        double a,
        double u0)
        : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
          a_(a),
          u0_(u0)
    {

        n_ref_ = std::sqrt(mu_ / (a_ * a_ * a_)); // Mean motion of the reference orbit
    }

    //-----------------------------------------------------------------------------
    Eigen::VectorXd SpacecraftROE::getContinuousDynamics(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &control) const
    {
        /**
         * Based on the linear QNSROE model:
         *   ẋ = A x + B u
         *
         *   x = [ da, dlambda, dex, dey, dix, diy ]
         *   u = [ ur, ut, un ]
         *
         * A = [  0           0  0  0  0  0
         *       -3/2 * n_ref_ 0  0  0  0  0
         *        0           0  0  0  0  0
         *        0           0  0  0  0  0
         *        0           0  0  0  0  0
         *        0           0  0  0  0  0 ]
         *
         * B(t) = 1/(n_ref_*a_) *
         *        [  0   2     0
         *         -2   0     0
         *          su  2cu   0
         *         -cu  2su   0
         *          0    0    cu
         *          0    0    su ]
         */

        // Create a zero derivative vector
        Eigen::VectorXd xdot = Eigen::VectorXd::Zero(STATE_DIM);

        // A*x portion:
        double da = state(STATE_DA);
        xdot(STATE_DLAMBDA) = -1.5 * n_ref_ * da;

        // B*u portion:
        double ur = control(CONTROL_UR);
        double ut = control(CONTROL_UT);
        double un = control(CONTROL_UN);

        // Factor for scaling control
        const double factor = 1.0 / (n_ref_ * a_);

        // Calculate su and cu based on initial argument of latitude:
        double su = std::sin(u0_);
        double cu = std::cos(u0_);

        // Combine:
        xdot(STATE_DA) += 2.0 * factor * ut;
        xdot(STATE_DLAMBDA) += -2.0 * factor * ur;
        xdot(STATE_DEX) = factor * (su * ur + 2.0 * cu * ut);
        xdot(STATE_DEY) = factor * (-cu * ur + 2.0 * su * ut);
        xdot(STATE_DIX) = factor * (cu * un);
        xdot(STATE_DIY) = factor * (su * un);

        return xdot;
    }

    //-----------------------------------------------------------------------------
    Eigen::MatrixXd SpacecraftROE::getStateJacobian(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &control) const
    {
        // Return the state Jacobian matrix A
        Eigen::MatrixXd A(STATE_DIM, STATE_DIM);
        A.setZero();

        A(STATE_DA, STATE_DLAMBDA) = -1.5 * n_ref_ * state(STATE_DA);

        return A;
    }

    //-----------------------------------------------------------------------------
    Eigen::MatrixXd SpacecraftROE::getControlJacobian(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &control) const
    {
        // Return the control Jacobian matrix B
        Eigen::MatrixXd B(STATE_DIM, CONTROL_DIM);
        B.setZero();
        double factor = 1.0 / (n_ref_ * a_);
        double su = std::sin(u0_);
        double cu = std::cos(u0_);

        B(STATE_DA, CONTROL_UT) = 2.0 * factor;
        B(STATE_DLAMBDA, CONTROL_UR) = -2.0 * factor;
        B(STATE_DEX, CONTROL_UR) = factor * (su * control(CONTROL_UR) + 2.0 * cu * control(CONTROL_UT));
        B(STATE_DEY, CONTROL_UR) = factor * (-cu * control(CONTROL_UR) + 2.0 * su * control(CONTROL_UT));
        B(STATE_DIX, CONTROL_UN) = factor * cu;
        B(STATE_DIY, CONTROL_UN) = factor * su;

        return B;
    }

    //-----------------------------------------------------------------------------
    std::vector<Eigen::MatrixXd> SpacecraftROE::getStateHessian(
        const Eigen::VectorXd & /*state*/,
        const Eigen::VectorXd & /*control*/) const
    {
        // For this linear(ish) model, second derivatives wrt state are zero.
        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
        for (int i = 0; i < STATE_DIM; ++i) {
            hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
        }
        return hessians;
    }

    //-----------------------------------------------------------------------------
    std::vector<Eigen::MatrixXd> SpacecraftROE::getControlHessian(
        const Eigen::VectorXd & /*state*/,
        const Eigen::VectorXd & /*control*/) const
    {
        // Similarly, second derivatives wrt control are zero for a linear system.
        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
        for (int i = 0; i < STATE_DIM; ++i) {
            hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
        }
        return hessians;
    }

    //-----------------------------------------------------------------------------
    Eigen::VectorXd SpacecraftROE::transformROEToHCW(const Eigen::VectorXd &roe, double t) const
    {
        // Check dimension
        if (roe.size() != STATE_DIM)
        {
            throw std::runtime_error(
                "transformROEToHCW: input ROE vector must be 6D [da, dlambda, dex, dey, dix, diy].");
        }

        // 1) Compute the angle for reference orbit at time t
        double phi = n_ref_ * t + u0_;
        double cn = std::cos(phi);
        double sn = std::sin(phi);

        Eigen::Matrix<double, 6, 6> T;
        T.setZero();

        // Fill row by row (0-based indexing).
        // Position components
        // Row 0: [1, 0, -cn,   -sn,   0,    0] * a_
        T(0, 0) = 1.0;
        T(0, 1) = 0.0;
        T(0, 2) = -cn;
        T(0, 3) = -sn;
        T(0, 4) = 0.0;
        T(0, 5) = 0.0;

        // Row 1: [0, 1, 2sn,  -2cn,  0,    0] * a_
        T(1, 0) = 0.0;
        T(1, 1) = 1.0;
        T(1, 2) = 2.0 * sn;
        T(1, 3) = -2.0 * cn;
        T(1, 4) = 0.0;
        T(1, 5) = 0.0;

        // Row 2: [0, 0, 0,     0,     sn,   -cn] * a_
        T(2, 0) = 0.0;
        T(2, 1) = 0.0;
        T(2, 2) = 0.0;
        T(2, 3) = 0.0;
        T(2, 4) = sn;
        T(2, 5) = -cn;

        // Velocity components
        // Row 3: [0, 0, n*sn,  -n*cn,  0,    0]
        T(3, 0) = 0.0;
        T(3, 1) = 0.0;
        T(3, 2) = n_ref_ * sn;
        T(3, 3) = -n_ref_ * cn;
        T(3, 4) = 0.0;
        T(3, 5) = 0.0;

        // Row 4: [-3n/2, 0, 2n*cn,  2n*sn,  0,    0]
        T(4, 0) = -1.5 * n_ref_;
        T(4, 1) = 0.0;
        T(4, 2) = 2.0 * n_ref_ * cn;
        T(4, 3) = 2.0 * n_ref_ * sn;
        T(4, 4) = 0.0;
        T(4, 5) = 0.0;

        // Row 5: [0, 0, 0,     0,     n*cn, n*sn]
        T(5, 0) = 0.0;
        T(5, 1) = 0.0;
        T(5, 2) = 0.0;
        T(5, 3) = 0.0;
        T(5, 4) = n_ref_ * cn;
        T(5, 5) = n_ref_ * sn;

        // Multiply entire matrix by 'a_'
        T *= a_;

        // 3) Multiply T by the input ROE vector
        Eigen::VectorXd hcw = T * roe; // hcw is [x, y, z, xdot, ydot, zdot]

        return hcw;
    }

    Eigen::VectorXd SpacecraftROE::transformHCWToROE(const Eigen::VectorXd &hcw, double t) const
    {
        // Check dimension
        if (hcw.size() != STATE_DIM)
        {
            throw std::runtime_error(
                "transformHCWToROE: input HCW state vector must be 6D [x, y, z, xdot, ydot, zdot].");
        }

        // 1) Compute the orbital phase for the reference orbit at time t
        double phi = n_ref_ * t + u0_;
        double cn = std::cos(phi);
        double sn = std::sin(phi);

        // 2) Build the inverse transformation matrix Tinv (6x6)
        //
        //      Tinv = [4   0    0     0     2/n   0
        //              0   1    0    -2/n   0     0
        //              3cn 0    0     sn/n  2cn/n 0
        //              3sn 0    0    -cn/n  2sn/n 0
        //              0   0   sn     0     0     cn/n
        //              0   0  -cn     0     0     sn/n ] / a
        //

        Eigen::Matrix<double, 6, 6> Tinv;
        Tinv.setZero();

        // row 0
        Tinv(0, 0) = 4.0;
        Tinv(0, 1) = 0.0;
        Tinv(0, 2) = 0.0;
        Tinv(0, 3) = 0.0;
        Tinv(0, 4) = 2.0 / n_ref_;
        Tinv(0, 5) = 0.0;

        // row 1
        Tinv(1, 0) = 0.0;
        Tinv(1, 1) = 1.0;
        Tinv(1, 2) = 0.0;
        Tinv(1, 3) = -2.0 / n_ref_;
        Tinv(1, 4) = 0.0;
        Tinv(1, 5) = 0.0;

        // row 2
        Tinv(2, 0) = 3.0 * cn;
        Tinv(2, 1) = 0.0;
        Tinv(2, 2) = 0.0;
        Tinv(2, 3) = sn / n_ref_;
        Tinv(2, 4) = 2.0 * cn / n_ref_;
        Tinv(2, 5) = 0.0;

        // row 3
        Tinv(3, 0) = 3.0 * sn;
        Tinv(3, 1) = 0.0;
        Tinv(3, 2) = 0.0;
        Tinv(3, 3) = -cn / n_ref_;
        Tinv(3, 4) = 2.0 * sn / n_ref_;
        Tinv(3, 5) = 0.0;

        // row 4
        Tinv(4, 0) = 0.0;
        Tinv(4, 1) = 0.0;
        Tinv(4, 2) = sn;
        Tinv(4, 3) = 0.0;
        Tinv(4, 4) = 0.0;
        Tinv(4, 5) = cn / n_ref_;

        // row 5
        Tinv(5, 0) = 0.0;
        Tinv(5, 1) = 0.0;
        Tinv(5, 2) = -cn;
        Tinv(5, 3) = 0.0;
        Tinv(5, 4) = 0.0;
        Tinv(5, 5) = sn / n_ref_;

        // Divide entire matrix by a_
        Tinv /= a_;

        // 3) Multiply Tinv by the input HCW vector => QNS-ROE
        Eigen::VectorXd roeVec = Tinv * hcw;

        return roeVec;
    }

} // namespace cddp
