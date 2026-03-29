#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cddp_core/dynamical_system.hpp"
#include "dynamics_model/pendulum.hpp"
#include "dynamics_model/unicycle.hpp"
#include "dynamics_model/bicycle.hpp"
#include "dynamics_model/car.hpp"
#include "dynamics_model/cartpole.hpp"
#include "dynamics_model/dubins_car.hpp"
#include "dynamics_model/forklift.hpp"
#include "dynamics_model/acrobot.hpp"
#include "dynamics_model/quadrotor.hpp"
#include "dynamics_model/quadrotor_rate.hpp"
#include "dynamics_model/manipulator.hpp"
#include "dynamics_model/spacecraft_linear.hpp"
#include "dynamics_model/spacecraft_linear_fuel.hpp"
#include "dynamics_model/spacecraft_nonlinear.hpp"
#include "dynamics_model/dreyfus_rocket.hpp"
#include "dynamics_model/spacecraft_landing2d.hpp"
#include "dynamics_model/spacecraft_roe.hpp"
#include "dynamics_model/spacecraft_twobody.hpp"
#include "dynamics_model/lti_system.hpp"
#include "dynamics_model/usv_3dof.hpp"
#include "dynamics_model/euler_attitude.hpp"
#include "dynamics_model/quaternion_attitude.hpp"
#include "dynamics_model/mrp_attitude.hpp"

namespace py = pybind11;

// Use py::nodelete so Python doesn't free the object - CDDP takes ownership via unique_ptr
template <typename T>
using nodeleter = std::unique_ptr<T, py::nodelete>;

// Trampoline class for Python subclassing
class PyDynamicalSystem : public cddp::DynamicalSystem {
public:
    using cddp::DynamicalSystem::DynamicalSystem;

    Eigen::VectorXd getContinuousDynamics(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(Eigen::VectorXd, cddp::DynamicalSystem,
                          getContinuousDynamics, state, control, time);
    }

    Eigen::VectorXd getDiscreteDynamics(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(Eigen::VectorXd, cddp::DynamicalSystem,
                          getDiscreteDynamics, state, control, time);
    }

    Eigen::MatrixXd getStateJacobian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(Eigen::MatrixXd, cddp::DynamicalSystem,
                          getStateJacobian, state, control, time);
    }

    Eigen::MatrixXd getControlJacobian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(Eigen::MatrixXd, cddp::DynamicalSystem,
                          getControlJacobian, state, control, time);
    }

    std::vector<Eigen::MatrixXd> getStateHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, cddp::DynamicalSystem,
                          getStateHessian, state, control, time);
    }

    std::vector<Eigen::MatrixXd> getControlHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, cddp::DynamicalSystem,
                          getControlHessian, state, control, time);
    }

    std::vector<Eigen::MatrixXd> getCrossHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double time) const override {
        PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, cddp::DynamicalSystem,
                          getCrossHessian, state, control, time);
    }
};

void bind_dynamics(py::module_& m) {
    // Base class with trampoline - py::nodelete so CDDP can take ownership
    py::class_<cddp::DynamicalSystem, PyDynamicalSystem, nodeleter<cddp::DynamicalSystem>>(m, "DynamicalSystem")
        .def(py::init<int, int, double, std::string>(),
             py::arg("state_dim"), py::arg("control_dim"),
             py::arg("timestep"), py::arg("integration_type") = "euler")
        .def("get_continuous_dynamics", &cddp::DynamicalSystem::getContinuousDynamics,
             py::arg("state"), py::arg("control"), py::arg("time") = 0.0)
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
        .def_property_readonly("control_dim", &cddp::DynamicalSystem::getControlDim)
        .def_property_readonly("timestep", &cddp::DynamicalSystem::getTimestep)
        .def_property_readonly("integration_type", &cddp::DynamicalSystem::getIntegrationType);

    // --- Concrete dynamics models ---
    // All use nodeleter so CDDP solver takes ownership

    py::class_<cddp::Pendulum, cddp::DynamicalSystem, nodeleter<cddp::Pendulum>>(m, "Pendulum")
        .def(py::init<double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("length") = 1.0,
             py::arg("mass") = 1.0, py::arg("damping") = 0.0,
             py::arg("integration_type") = "euler");

    py::class_<cddp::Unicycle, cddp::DynamicalSystem, nodeleter<cddp::Unicycle>>(m, "Unicycle")
        .def(py::init<double, std::string>(),
             py::arg("timestep"), py::arg("integration_type") = "euler");

    py::class_<cddp::Bicycle, cddp::DynamicalSystem, nodeleter<cddp::Bicycle>>(m, "Bicycle")
        .def(py::init<double, double, std::string>(),
             py::arg("timestep"), py::arg("wheelbase"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::Car, cddp::DynamicalSystem, nodeleter<cddp::Car>>(m, "Car")
        .def(py::init<double, double, std::string>(),
             py::arg("timestep") = 0.03, py::arg("wheelbase") = 2.0,
             py::arg("integration_type") = "euler");

    py::class_<cddp::CartPole, cddp::DynamicalSystem, nodeleter<cddp::CartPole>>(m, "CartPole")
        .def(py::init<double, std::string, double, double, double, double, double>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4",
             py::arg("cart_mass") = 1.0, py::arg("pole_mass") = 0.2,
             py::arg("pole_length") = 0.5, py::arg("gravity") = 9.81,
             py::arg("damping") = 0.0);

    py::class_<cddp::DubinsCar, cddp::DynamicalSystem, nodeleter<cddp::DubinsCar>>(m, "DubinsCar")
        .def(py::init<double, double, std::string>(),
             py::arg("speed"), py::arg("timestep"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::Forklift, cddp::DynamicalSystem, nodeleter<cddp::Forklift>>(m, "Forklift")
        .def(py::init<double, double, std::string, bool, double>(),
             py::arg("timestep") = 0.01, py::arg("wheelbase") = 2.0,
             py::arg("integration_type") = "euler",
             py::arg("rear_steer") = true,
             py::arg("max_steering_angle") = 0.785398);

    py::class_<cddp::Acrobot, cddp::DynamicalSystem, nodeleter<cddp::Acrobot>>(m, "Acrobot")
        .def(py::init<double, double, double, double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("l1") = 1.0, py::arg("l2") = 1.0,
             py::arg("m1") = 1.0, py::arg("m2") = 1.0,
             py::arg("J1") = 1.0, py::arg("J2") = 1.0,
             py::arg("integration_type") = "euler");

    py::class_<cddp::Quadrotor, cddp::DynamicalSystem, nodeleter<cddp::Quadrotor>>(m, "Quadrotor")
        .def(py::init<double, double, const Eigen::Matrix3d&, double, std::string>(),
             py::arg("timestep"), py::arg("mass"),
             py::arg("inertia_matrix"), py::arg("arm_length"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::QuadrotorRate, cddp::DynamicalSystem, nodeleter<cddp::QuadrotorRate>>(m, "QuadrotorRate")
        .def(py::init<double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("mass"),
             py::arg("max_thrust"), py::arg("max_rate"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::Manipulator, cddp::DynamicalSystem, nodeleter<cddp::Manipulator>>(m, "Manipulator")
        .def(py::init<double, std::string>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4");

    py::class_<cddp::HCW, cddp::DynamicalSystem, nodeleter<cddp::HCW>>(m, "HCW")
        .def(py::init<double, double, double, std::string>(),
             py::arg("timestep"), py::arg("mean_motion"),
             py::arg("mass"), py::arg("integration_type") = "euler");

    py::class_<cddp::SpacecraftLinearFuel, cddp::DynamicalSystem, nodeleter<cddp::SpacecraftLinearFuel>>(m, "SpacecraftLinearFuel")
        .def(py::init<double, double, double, double, std::string>(),
             py::arg("timestep"), py::arg("mean_motion"),
             py::arg("isp"), py::arg("g0") = 9.80665,
             py::arg("integration_type") = "euler");

    py::class_<cddp::SpacecraftNonlinear, cddp::DynamicalSystem, nodeleter<cddp::SpacecraftNonlinear>>(m, "SpacecraftNonlinear")
        .def(py::init<double, std::string, double, double, double, double>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4",
             py::arg("mass") = 1.0, py::arg("r_scale") = 1.0,
             py::arg("v_scale") = 1.0, py::arg("mu") = 1.0);

    py::class_<cddp::DreyfusRocket, cddp::DynamicalSystem, nodeleter<cddp::DreyfusRocket>>(m, "DreyfusRocket")
        .def(py::init<double, std::string, double, double>(),
             py::arg("timestep"), py::arg("integration_type") = "rk4",
             py::arg("thrust_acceleration") = 64.0,
             py::arg("gravity_acceleration") = 32.0);

    py::class_<cddp::SpacecraftLanding2D, cddp::DynamicalSystem, nodeleter<cddp::SpacecraftLanding2D>>(m, "SpacecraftLanding2D")
        .def(py::init<double, std::string, double, double, double, double, double, double>(),
             py::arg("timestep") = 0.1, py::arg("integration_type") = "rk4",
             py::arg("mass") = 100000.0, py::arg("length") = 50.0,
             py::arg("width") = 10.0, py::arg("min_thrust") = 880000.0,
             py::arg("max_thrust") = 2210000.0, py::arg("max_gimble") = 0.349066);

    py::class_<cddp::SpacecraftROE, cddp::DynamicalSystem, nodeleter<cddp::SpacecraftROE>>(m, "SpacecraftROE")
        .def(py::init<double, const std::string&, double, double, double>(),
             py::arg("timestep"), py::arg("integration_type"),
             py::arg("a"), py::arg("u0") = 0.0, py::arg("mass_kg") = 1.0);

    py::class_<cddp::SpacecraftTwobody, cddp::DynamicalSystem, nodeleter<cddp::SpacecraftTwobody>>(m, "SpacecraftTwobody")
        .def(py::init<double, double, double>(),
             py::arg("timestep"), py::arg("mu"), py::arg("mass"));

    py::class_<cddp::LTISystem, cddp::DynamicalSystem, nodeleter<cddp::LTISystem>>(m, "LTISystem")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, double, std::string>(),
             py::arg("A"), py::arg("B"), py::arg("timestep"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::Usv3Dof, cddp::DynamicalSystem, nodeleter<cddp::Usv3Dof>>(m, "Usv3Dof")
        .def(py::init<double, std::string>(),
             py::arg("timestep"), py::arg("integration_type") = "euler");

    py::class_<cddp::EulerAttitude, cddp::DynamicalSystem, nodeleter<cddp::EulerAttitude>>(m, "EulerAttitude")
        .def(py::init<double, const Eigen::Matrix3d&, std::string>(),
             py::arg("timestep"), py::arg("inertia_matrix"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::QuaternionAttitude, cddp::DynamicalSystem, nodeleter<cddp::QuaternionAttitude>>(m, "QuaternionAttitude")
        .def(py::init<double, const Eigen::Matrix3d&, std::string>(),
             py::arg("timestep"), py::arg("inertia_matrix"),
             py::arg("integration_type") = "euler");

    py::class_<cddp::MrpAttitude, cddp::DynamicalSystem, nodeleter<cddp::MrpAttitude>>(m, "MrpAttitude")
        .def(py::init<double, const Eigen::Matrix3d&, std::string>(),
             py::arg("timestep"), py::arg("inertia_matrix"),
             py::arg("integration_type") = "euler");
}
