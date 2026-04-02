#include <gtest/gtest.h>
#include "cddp-cpp/dynamics_model/mrp_attitude.hpp"
#include "cddp-cpp/dynamics_model/quaternion_attitude.hpp"
#include "cddp-cpp/dynamics_model/euler_attitude.hpp"
#include "cddp_core/helper.hpp" // For attitude conversions
#include <Eigen/Dense>
#include <vector>

namespace cddp
{
    namespace tests
    {

        // Common setup for attitude tests
        class AttitudeDynamicsTest : public ::testing::Test
        {
        protected:
            double dt = 0.01;
            Eigen::Matrix3d inertia = Eigen::Vector3d(0.1, 0.2, 0.3).asDiagonal(); // Slightly non-uniform inertia
            Eigen::VectorXd state_mrp;
            Eigen::VectorXd control_mrp = Eigen::VectorXd::Zero(MrpAttitude::CONTROL_DIM);

            Eigen::VectorXd state_quat;
            Eigen::VectorXd control_quat = Eigen::VectorXd::Zero(QuaternionAttitude::CONTROL_DIM);

            Eigen::VectorXd state_euler;
            Eigen::VectorXd control_euler = Eigen::VectorXd::Zero(EulerAttitude::CONTROL_DIM);

            // Initial conditions
            Eigen::Vector3d initial_euler_angles = Eigen::Vector3d(0.1, -0.2, 0.3); // Yaw, Pitch, Roll (radians)
            Eigen::Vector3d initial_omega = Eigen::Vector3d(0.5, -0.3, 0.8);        // rad/s

            void SetUp() override
            {
                // Convert initial Euler to MRP and Quaternion for consistent start
                Eigen::Matrix3d R_init = cddp::helper::eulerZYXToRotationMatrix(initial_euler_angles);
                Eigen::Vector3d mrp_init = cddp::helper::rotationMatrixToMRP(R_init);
                Eigen::Vector4d quat_init = cddp::helper::rotationMatrixToQuat(R_init);

                state_mrp.resize(MrpAttitude::STATE_DIM);
                state_mrp << mrp_init, initial_omega;

                state_quat.resize(QuaternionAttitude::STATE_DIM);
                state_quat << quat_init, initial_omega;

                state_euler.resize(EulerAttitude::STATE_DIM);
                state_euler << initial_euler_angles, initial_omega;
            }
        };

        // --- MRP Attitude Tests ---
        TEST_F(AttitudeDynamicsTest, MrpConstruction)
        {
            EXPECT_NO_THROW(MrpAttitude mrp_model(dt, inertia));
            MrpAttitude mrp_model(dt, inertia);
            EXPECT_EQ(mrp_model.getStateDim(), MrpAttitude::STATE_DIM);
            EXPECT_EQ(mrp_model.getControlDim(), MrpAttitude::CONTROL_DIM);
        }

        TEST_F(AttitudeDynamicsTest, MrpDynamicsDimension)
        {
            MrpAttitude mrp_model(dt, inertia);
            Eigen::VectorXd x_dot = mrp_model.getContinuousDynamics(state_mrp, control_mrp, 0.0);
            EXPECT_EQ(x_dot.size(), MrpAttitude::STATE_DIM);
        }

        TEST_F(AttitudeDynamicsTest, MrpJacobianDimensions)
        {
            MrpAttitude mrp_model(dt, inertia);
            Eigen::MatrixXd A = mrp_model.getStateJacobian(state_mrp, control_mrp, 0.0);
            Eigen::MatrixXd B = mrp_model.getControlJacobian(state_mrp, control_mrp, 0.0);
            EXPECT_EQ(A.rows(), MrpAttitude::STATE_DIM);
            EXPECT_EQ(A.cols(), MrpAttitude::STATE_DIM);
            EXPECT_EQ(B.rows(), MrpAttitude::STATE_DIM);
            EXPECT_EQ(B.cols(), MrpAttitude::CONTROL_DIM);
        }

        TEST_F(AttitudeDynamicsTest, MrpHessianDimensions)
        {
            MrpAttitude mrp_model(dt, inertia);
            auto state_hess = mrp_model.getStateHessian(state_mrp, control_mrp, 0.0);
            auto control_hess = mrp_model.getControlHessian(state_mrp, control_mrp, 0.0);
            auto cross_hess = mrp_model.getCrossHessian(state_mrp, control_mrp, 0.0);

            EXPECT_EQ(state_hess.size(), MrpAttitude::STATE_DIM);
            EXPECT_EQ(control_hess.size(), MrpAttitude::STATE_DIM);
            EXPECT_EQ(cross_hess.size(), MrpAttitude::STATE_DIM);

            if (state_hess.size() == MrpAttitude::STATE_DIM)
            { // Avoid crashing if sizes are wrong
                for (int i = 0; i < MrpAttitude::STATE_DIM; ++i)
                {
                    EXPECT_EQ(state_hess[i].rows(), MrpAttitude::STATE_DIM);
                    EXPECT_EQ(state_hess[i].cols(), MrpAttitude::STATE_DIM);
                    EXPECT_EQ(control_hess[i].rows(), MrpAttitude::CONTROL_DIM);
                    EXPECT_EQ(control_hess[i].cols(), MrpAttitude::CONTROL_DIM);
                    EXPECT_EQ(cross_hess[i].rows(), MrpAttitude::CONTROL_DIM);
                    EXPECT_EQ(cross_hess[i].cols(), MrpAttitude::STATE_DIM);
                }
            }
        }

        // --- Quaternion Attitude Tests ---
        TEST_F(AttitudeDynamicsTest, QuaternionConstruction)
        {
            EXPECT_NO_THROW(QuaternionAttitude quat_model(dt, inertia));
            QuaternionAttitude quat_model(dt, inertia);
            EXPECT_EQ(quat_model.getStateDim(), QuaternionAttitude::STATE_DIM);
            EXPECT_EQ(quat_model.getControlDim(), QuaternionAttitude::CONTROL_DIM);
        }

        TEST_F(AttitudeDynamicsTest, QuaternionDynamicsDimension)
        {
            QuaternionAttitude quat_model(dt, inertia);
            Eigen::VectorXd x_dot = quat_model.getContinuousDynamics(state_quat, control_quat, 0.0);
            EXPECT_EQ(x_dot.size(), QuaternionAttitude::STATE_DIM);
        }

        TEST_F(AttitudeDynamicsTest, QuaternionJacobianDimensions)
        {
            QuaternionAttitude quat_model(dt, inertia);
            Eigen::MatrixXd A = quat_model.getStateJacobian(state_quat, control_quat, 0.0);
            Eigen::MatrixXd B = quat_model.getControlJacobian(state_quat, control_quat, 0.0);
            EXPECT_EQ(A.rows(), QuaternionAttitude::STATE_DIM);
            EXPECT_EQ(A.cols(), QuaternionAttitude::STATE_DIM);
            EXPECT_EQ(B.rows(), QuaternionAttitude::STATE_DIM);
            EXPECT_EQ(B.cols(), QuaternionAttitude::CONTROL_DIM);
        }

        TEST_F(AttitudeDynamicsTest, QuaternionHessianDimensions)
        {
            QuaternionAttitude quat_model(dt, inertia);
            auto state_hess = quat_model.getStateHessian(state_quat, control_quat, 0.0);
            auto control_hess = quat_model.getControlHessian(state_quat, control_quat, 0.0);
            auto cross_hess = quat_model.getCrossHessian(state_quat, control_quat, 0.0);

            EXPECT_EQ(state_hess.size(), QuaternionAttitude::STATE_DIM);
            EXPECT_EQ(control_hess.size(), QuaternionAttitude::STATE_DIM);
            EXPECT_EQ(cross_hess.size(), QuaternionAttitude::STATE_DIM);

            if (state_hess.size() == QuaternionAttitude::STATE_DIM)
            { // Avoid crashing
                for (int i = 0; i < QuaternionAttitude::STATE_DIM; ++i)
                {
                    EXPECT_EQ(state_hess[i].rows(), QuaternionAttitude::STATE_DIM);
                    EXPECT_EQ(state_hess[i].cols(), QuaternionAttitude::STATE_DIM);
                    EXPECT_EQ(control_hess[i].rows(), QuaternionAttitude::CONTROL_DIM);
                    EXPECT_EQ(control_hess[i].cols(), QuaternionAttitude::CONTROL_DIM);
                    EXPECT_EQ(cross_hess[i].rows(), QuaternionAttitude::STATE_DIM);
                    EXPECT_EQ(cross_hess[i].cols(), QuaternionAttitude::CONTROL_DIM);
                }
            }
        }

        // --- Euler Attitude Tests ---
        TEST_F(AttitudeDynamicsTest, EulerConstruction)
        {
            EXPECT_NO_THROW(EulerAttitude euler_model(dt, inertia));
            EulerAttitude euler_model(dt, inertia);
            EXPECT_EQ(euler_model.getStateDim(), EulerAttitude::STATE_DIM);
            EXPECT_EQ(euler_model.getControlDim(), EulerAttitude::CONTROL_DIM);
        }

        TEST_F(AttitudeDynamicsTest, EulerDynamicsDimension)
        {
            EulerAttitude euler_model(dt, inertia);
            Eigen::VectorXd x_dot = euler_model.getContinuousDynamics(state_euler, control_euler, 0.0);
            EXPECT_EQ(x_dot.size(), EulerAttitude::STATE_DIM);
        }

        TEST_F(AttitudeDynamicsTest, EulerJacobianDimensions)
        {
            EulerAttitude euler_model(dt, inertia);
            // Test near non-singular point
            Eigen::VectorXd non_singular_state = state_euler;
            non_singular_state(EulerAttitude::STATE_EULER_Y) = 0.1; // Ensure pitch is not pi/2
            Eigen::MatrixXd A = euler_model.getStateJacobian(non_singular_state, control_euler, 0.0);
            Eigen::MatrixXd B = euler_model.getControlJacobian(non_singular_state, control_euler, 0.0);
            EXPECT_EQ(A.rows(), EulerAttitude::STATE_DIM);
            EXPECT_EQ(A.cols(), EulerAttitude::STATE_DIM);
            EXPECT_EQ(B.rows(), EulerAttitude::STATE_DIM);
            EXPECT_EQ(B.cols(), EulerAttitude::CONTROL_DIM);
        }

        TEST_F(AttitudeDynamicsTest, EulerHessianDimensions)
        {
            EulerAttitude euler_model(dt, inertia);
            // Test near non-singular point
            Eigen::VectorXd non_singular_state = state_euler;
            non_singular_state(EulerAttitude::STATE_EULER_Y) = 0.1; // Ensure pitch is not pi/2
            auto state_hess = euler_model.getStateHessian(non_singular_state, control_euler, 0.0);
            auto control_hess = euler_model.getControlHessian(non_singular_state, control_euler, 0.0);
            auto cross_hess = euler_model.getCrossHessian(non_singular_state, control_euler, 0.0);

            EXPECT_EQ(state_hess.size(), EulerAttitude::STATE_DIM);
            EXPECT_EQ(control_hess.size(), EulerAttitude::STATE_DIM);
            EXPECT_EQ(cross_hess.size(), EulerAttitude::STATE_DIM);

            if (state_hess.size() == EulerAttitude::STATE_DIM)
            { // Avoid crashing
                for (int i = 0; i < EulerAttitude::STATE_DIM; ++i)
                {
                    EXPECT_EQ(state_hess[i].rows(), EulerAttitude::STATE_DIM);
                    EXPECT_EQ(state_hess[i].cols(), EulerAttitude::STATE_DIM);
                    EXPECT_EQ(control_hess[i].rows(), EulerAttitude::CONTROL_DIM);
                    EXPECT_EQ(control_hess[i].cols(), EulerAttitude::CONTROL_DIM);
                    EXPECT_EQ(cross_hess[i].rows(), EulerAttitude::STATE_DIM);
                    EXPECT_EQ(cross_hess[i].cols(), EulerAttitude::CONTROL_DIM);
                }
            }
        }

        // --- Trajectory Comparison ---
        TEST_F(AttitudeDynamicsTest, TrajectoryComparison)
        {
            int num_steps = 500;
            double sim_time = num_steps * dt;

            // Models (using RK4 integration for better accuracy)
            MrpAttitude mrp_model(dt, inertia, "rk4");
            QuaternionAttitude quat_model(dt, inertia, "rk4");
            EulerAttitude euler_model(dt, inertia, "rk4");

            // Trajectory storage
            std::vector<double> t_eval(num_steps + 1);
            std::vector<Eigen::Vector3d> euler_traj_mrp(num_steps + 1);
            std::vector<Eigen::Vector3d> euler_traj_euler(num_steps + 1);
            std::vector<Eigen::Vector3d> omega_traj_mrp(num_steps + 1);
            std::vector<Eigen::Vector4d> quat_traj_quat(num_steps + 1);
            std::vector<Eigen::Vector3d> omega_traj_euler(num_steps + 1);
            std::vector<Eigen::Vector3d> omega_traj_quat_sim(num_steps + 1);
            std::vector<Eigen::Vector4d> quat_traj_mrp(num_steps + 1);

            // Initial states already set in SetUp()
            Eigen::VectorXd current_state_mrp = state_mrp;
            Eigen::VectorXd current_state_quat = state_quat;
            Eigen::VectorXd current_state_euler = state_euler;

            // Store initial state
            t_eval[0] = 0.0;
            euler_traj_mrp[0] = cddp::helper::mrpToEulerZYX(current_state_mrp.head<3>());
            euler_traj_euler[0] = current_state_euler.head<3>();
            omega_traj_mrp[0] = current_state_mrp.tail<3>();
            omega_traj_euler[0] = current_state_euler.tail<3>();
            quat_traj_quat[0] = current_state_quat.head<4>();
            quat_traj_mrp[0] = cddp::helper::mrpToQuat(current_state_mrp.head<3>());
            omega_traj_quat_sim[0] = current_state_quat.tail<3>();

            // Simulation loop (using zero control)
            for (int k = 0; k < num_steps; ++k)
            {
                current_state_mrp = mrp_model.getDiscreteDynamics(current_state_mrp, control_mrp, 0.0);
                current_state_quat = quat_model.getDiscreteDynamics(current_state_quat, control_quat, 0.0);
                current_state_euler = euler_model.getDiscreteDynamics(current_state_euler, control_euler, 0.0);

                // Store results
                t_eval[k + 1] = (k + 1) * dt;
                euler_traj_mrp[k + 1] = cddp::helper::mrpToEulerZYX(current_state_mrp.head<3>());
                euler_traj_euler[k + 1] = current_state_euler.head<3>(); // Already Euler angles

                omega_traj_mrp[k + 1] = current_state_mrp.tail<3>();
                omega_traj_euler[k + 1] = current_state_euler.tail<3>();
                quat_traj_quat[k + 1] = current_state_quat.head<4>();
                quat_traj_mrp[k + 1] = cddp::helper::mrpToQuat(current_state_mrp.head<3>());
                omega_traj_quat_sim[k + 1] = current_state_quat.tail<3>();
            }

            ASSERT_EQ(t_eval.size(), static_cast<size_t>(num_steps + 1));
            ASSERT_EQ(euler_traj_mrp.size(), static_cast<size_t>(num_steps + 1));
            ASSERT_EQ(quat_traj_quat.size(), static_cast<size_t>(num_steps + 1));
            ASSERT_EQ(quat_traj_mrp.size(), static_cast<size_t>(num_steps + 1));
            EXPECT_TRUE(current_state_mrp.allFinite());
            EXPECT_TRUE(current_state_quat.allFinite());
            EXPECT_TRUE(current_state_euler.allFinite());
            EXPECT_TRUE(euler_traj_mrp.back().allFinite());
            EXPECT_TRUE(euler_traj_euler.back().allFinite());
            EXPECT_TRUE(quat_traj_quat.back().allFinite());
            EXPECT_TRUE(quat_traj_mrp.back().allFinite());
            EXPECT_TRUE(omega_traj_mrp.back().isApprox(omega_traj_quat_sim.back(), 1e-6));
            EXPECT_TRUE(omega_traj_mrp.back().isApprox(omega_traj_euler.back(), 1e-6));
        }

    } // namespace tests
} // namespace cddp
