#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cddp_core/dynamical_system.hpp"
#include "dynamics_model/acrobot.hpp"
#include "dynamics_model/bicycle.hpp"
#include "dynamics_model/car.hpp"
#include "dynamics_model/cartpole.hpp"
#include "dynamics_model/dreyfus_rocket.hpp"
#include "dynamics_model/dubins_car.hpp"
#include "dynamics_model/euler_attitude.hpp"
#include "dynamics_model/forklift.hpp"
#include "dynamics_model/lti_system.hpp"
#include "dynamics_model/manipulator.hpp"
#include "dynamics_model/mrp_attitude.hpp"
#include "dynamics_model/pendulum.hpp"
#include "dynamics_model/quadrotor.hpp"
#include "dynamics_model/quadrotor_rate.hpp"
#include "dynamics_model/quaternion_attitude.hpp"
#include "dynamics_model/spacecraft_landing2d.hpp"
#include "dynamics_model/spacecraft_linear.hpp"
#include "dynamics_model/spacecraft_linear_fuel.hpp"
#include "dynamics_model/spacecraft_nonlinear.hpp"
#include "dynamics_model/spacecraft_twobody.hpp"
#include "dynamics_model/unicycle.hpp"
#include "dynamics_model/usv_3dof.hpp"

namespace py = pybind11;

class PyDynamicalSystem : public cddp::DynamicalSystem {
public:
    using cddp::DynamicalSystem::DynamicalSystem;

    cddp::VectorXdual2nd
    getContinuousDynamicsAutodiff(const cddp::VectorXdual2nd &state,
                                  const cddp::VectorXdual2nd &control,
                                  double time) const override {
        throw std::runtime_error(
            "Python-defined DynamicalSystem objects do not support "
            "getContinuousDynamicsAutodiff. Override get_state_jacobian, "
            "get_control_jacobian, and any needed Hessian methods in Python, "
            "or use a built-in C++ dynamics model.");
    }

    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd &state,
                                          const Eigen::VectorXd &control,
                                          double time) const override {
        // Map C++ virtual dispatch to the snake_case Python API exported below.
        PYBIND11_OVERRIDE_NAME(Eigen::VectorXd, cddp::DynamicalSystem,
                               "get_continuous_dynamics",
                               getContinuousDynamics, state, control, time);
    }

    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &control,
                                        double time) const override {
        PYBIND11_OVERRIDE_NAME(Eigen::VectorXd, cddp::DynamicalSystem,
                               "get_discrete_dynamics", getDiscreteDynamics,
                               state, control, time);
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control,
                                     double time) const override {
        PYBIND11_OVERRIDE_NAME(Eigen::MatrixXd, cddp::DynamicalSystem,
                               "get_state_jacobian", getStateJacobian, state,
                               control, time);
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control,
                                       double time) const override {
        PYBIND11_OVERRIDE_NAME(Eigen::MatrixXd, cddp::DynamicalSystem,
                               "get_control_jacobian", getControlJacobian,
                               state, control, time);
    }

    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 double time) const override {
        PYBIND11_OVERRIDE_NAME(std::vector<Eigen::MatrixXd>,
                               cddp::DynamicalSystem, "get_state_hessian",
                               getStateHessian, state, control, time);
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control,
                      double time) const override {
        PYBIND11_OVERRIDE_NAME(std::vector<Eigen::MatrixXd>,
                               cddp::DynamicalSystem, "get_control_hessian",
                               getControlHessian, state, control, time);
    }

    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 double time) const override {
        PYBIND11_OVERRIDE_NAME(std::vector<Eigen::MatrixXd>,
                               cddp::DynamicalSystem, "get_cross_hessian",
                               getCrossHessian, state, control, time);
    }
};

void bind_dynamics(py::module_ &m) {
    py::class_<cddp::DynamicalSystem, PyDynamicalSystem>(m, "DynamicalSystem")
        .def(py::init<int, int, double, std::string>(), py::arg("state_dim"),
             py::arg("control_dim"), py::arg("timestep"),
             py::arg("integration_type") = "euler")
        .def("get_continuous_dynamics",
             &cddp::DynamicalSystem::getContinuousDynamics, py::arg("state"),
             py::arg("control"), py::arg("time") = 0.0)
        .def("get_discrete_dynamics", &cddp::DynamicalSystem::getDiscreteDynamics,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
        .def("get_state_jacobian", &cddp::DynamicalSystem::getStateJacobian,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
        .def("get_control_jacobian", &cddp::DynamicalSystem::getControlJacobian,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
        .def("get_state_hessian", &cddp::DynamicalSystem::getStateHessian,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
        .def("get_control_hessian", &cddp::DynamicalSystem::getControlHessian,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
        .def("get_cross_hessian", &cddp::DynamicalSystem::getCrossHessian,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
        .def_property_readonly("state_dim", &cddp::DynamicalSystem::getStateDim)
        .def_property_readonly("control_dim",
                               &cddp::DynamicalSystem::getControlDim)
        .def_property_readonly("timestep",
                               &cddp::DynamicalSystem::getTimestep)
        .def_property_readonly("integration_type",
                               &cddp::DynamicalSystem::getIntegrationType);

    py::class_<cddp::Pendulum, cddp::DynamicalSystem>(m, "Pendulum")
        .def(py::init<double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("length") = 1.0,
             py::arg("mass") = 1.0, py::arg("damping") = 0.0,
             py::arg("integration_type") = "euler");

    py::class_<cddp::Unicycle, cddp::DynamicalSystem>(m, "Unicycle")
        .def(py::init<double, std::string>(), py::arg("timestep"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::Bicycle, cddp::DynamicalSystem>(m, "Bicycle")
        .def(py::init<double, double, std::string>(), py::arg("timestep"),
             py::arg("wheelbase"), py::arg("integration_type") = "euler");

    py::class_<cddp::Car, cddp::DynamicalSystem>(m, "Car")
        .def(py::init<double, double, std::string>(),
             py::arg("timestep") = 0.03, py::arg("wheelbase") = 2.0,
             py::arg("integration_type") = "euler");

    py::class_<cddp::CartPole, cddp::DynamicalSystem>(m, "CartPole")
        .def(py::init<double, std::string, double, double, double, double,
                      double>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4",
             py::arg("cart_mass") = 1.0, py::arg("pole_mass") = 0.2,
             py::arg("pole_length") = 0.5, py::arg("gravity") = 9.81,
             py::arg("damping") = 0.0);

    py::class_<cddp::DubinsCar, cddp::DynamicalSystem>(m, "DubinsCar")
        .def(py::init<double, double, std::string>(), py::arg("speed"),
             py::arg("timestep"), py::arg("integration_type") = "euler");

    py::class_<cddp::Forklift, cddp::DynamicalSystem>(m, "Forklift")
        .def(py::init<double, double, std::string, bool, double>(),
             py::arg("timestep") = 0.01, py::arg("wheelbase") = 2.0,
             py::arg("integration_type") = "euler", py::arg("rear_steer") = true,
             py::arg("max_steering_angle") = 0.785398);

    py::class_<cddp::Acrobot, cddp::DynamicalSystem>(m, "Acrobot")
        .def(py::init<double, double, double, double, double, double, double,
                      std::string>(),
             py::arg("timestep"), py::arg("l1") = 1.0, py::arg("l2") = 1.0,
             py::arg("m1") = 1.0, py::arg("m2") = 1.0, py::arg("J1") = 1.0,
             py::arg("J2") = 1.0, py::arg("integration_type") = "euler");

    py::class_<cddp::Quadrotor, cddp::DynamicalSystem>(m, "Quadrotor")
        .def(py::init<double, double, const Eigen::Matrix3d &, double,
                      std::string>(),
             py::arg("timestep"), py::arg("mass"),
             py::arg("inertia_matrix"), py::arg("arm_length"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::QuadrotorRate, cddp::DynamicalSystem>(m, "QuadrotorRate")
        .def(py::init<double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("mass"), py::arg("max_thrust"),
             py::arg("max_rate"), py::arg("integration_type") = "euler");

    py::class_<cddp::Manipulator, cddp::DynamicalSystem>(m, "Manipulator")
        .def(py::init<double, std::string>(), py::arg("timestep"),
             py::arg("integration_type") = "rk4");

    py::class_<cddp::HCW, cddp::DynamicalSystem>(m, "HCW")
        .def(py::init<double, double, double, std::string>(),
             py::arg("timestep"), py::arg("mean_motion"), py::arg("mass"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::SpacecraftLinearFuel, cddp::DynamicalSystem>(
        m, "SpacecraftLinearFuel")
        .def(py::init<double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("mean_motion"), py::arg("isp"),
             py::arg("g0") = 9.80665,
             py::arg("integration_type") = "euler");

    py::class_<cddp::SpacecraftNonlinear, cddp::DynamicalSystem>(
        m, "SpacecraftNonlinear")
        .def(py::init<double, std::string, double, double, double, double>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4",
             py::arg("mass") = 1.0, py::arg("r_scale") = 1.0,
             py::arg("v_scale") = 1.0, py::arg("mu") = 1.0);

    py::class_<cddp::DreyfusRocket, cddp::DynamicalSystem>(m, "DreyfusRocket")
        .def(py::init<double, std::string, double, double>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4",
             py::arg("thrust_acceleration") = 64.0,
             py::arg("gravity_acceleration") = 32.0);

    py::class_<cddp::SpacecraftLanding2D, cddp::DynamicalSystem>(
        m, "SpacecraftLanding2D")
        .def(py::init<double, std::string, double, double, double, double,
                      double, double>(),
             py::arg("timestep") = 0.1, py::arg("integration_type") = "rk4",
             py::arg("mass") = 100000.0, py::arg("length") = 50.0,
             py::arg("width") = 10.0, py::arg("min_thrust") = 880000.0,
             py::arg("max_thrust") = 2210000.0,
             py::arg("max_gimble") = 0.349066);

    py::class_<cddp::SpacecraftTwobody, cddp::DynamicalSystem>(
        m, "SpacecraftTwobody")
        .def(py::init<double, double, double>(), py::arg("timestep"),
             py::arg("mu"), py::arg("mass"));

    py::class_<cddp::LTISystem, cddp::DynamicalSystem>(m, "LTISystem")
        .def(py::init<const Eigen::MatrixXd &, const Eigen::MatrixXd &, double,
                      std::string>(),
             py::arg("A"), py::arg("B"), py::arg("timestep"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::Usv3Dof, cddp::DynamicalSystem>(m, "Usv3Dof")
        .def(py::init<double, std::string>(), py::arg("timestep"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::EulerAttitude, cddp::DynamicalSystem>(m, "EulerAttitude")
        .def(py::init<double, const Eigen::Matrix3d &, std::string>(),
             py::arg("timestep"), py::arg("inertia_matrix"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::QuaternionAttitude, cddp::DynamicalSystem>(
        m, "QuaternionAttitude")
        .def(py::init<double, const Eigen::Matrix3d &, std::string>(),
             py::arg("timestep"), py::arg("inertia_matrix"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::MrpAttitude, cddp::DynamicalSystem>(m, "MrpAttitude")
        .def(py::init<double, const Eigen::Matrix3d &, std::string>(),
             py::arg("timestep"), py::arg("inertia_matrix"),
             py::arg("integration_type") = "euler");
}
