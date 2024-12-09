#ifndef CDDP_CONSTRAINT_HPP
#define CDDP_CONSTRAINT_HPP

#include <Eigen/Dense>

namespace cddp {

class Constraint {
public:
    // Constructor
    Constraint(const std::string& name) : name_(name) {}

    // Get the name of the constraint
    const std::string& getName() const { return name_; }

    // Evaluate the constraint function: g(x, u)
    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Get the lower bound of the constraint
    virtual Eigen::VectorXd getLowerBound() const = 0;

    // Get the upper bound of the constraint
    virtual Eigen::VectorXd getUpperBound() const = 0;

    // Get the Jacobian of the constraint w.r.t the state: dg/dx
    virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Get the Jacobian of the constraint w.r.t the control: dg/du
    virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Get both Jacobians: dg/dx, dg/du
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getJacobians(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
        return {getStateJacobian(state, control), getControlJacobian(state, control)};
    }
private:
    std::string name_; // Name of the constraint
};


class ControlBoxConstraint : public Constraint {
public:
    ControlBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) 
        : Constraint("ControlBoxConstraint"), lower_bound_(lower_bound), upper_bound_(upper_bound) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return control; // The constraint is directly on the control
    }

    Eigen::VectorXd getLowerBound() const override {
        return lower_bound_;
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Zero(control.size(), state.size()); 
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Identity(control.size(), control.size()); 
    }

private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

class StateBoxConstraint : public Constraint {
public:
    StateBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) 
        : Constraint("StateBoxConstraint"), lower_bound_(lower_bound), upper_bound_(upper_bound) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return state; // The constraint is directly on the state
    }

    Eigen::VectorXd getLowerBound() const override {
        return lower_bound_;
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Identity(state.size(), state.size()); 
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Zero(state.size(), control.size()); 
    }

private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

// CircleConstraint (assuming the circle is centered at the origin)
class CircleConstraint : public Constraint {
public:
    CircleConstraint(double radius) : Constraint("CircleConstraint"), radius_(radius) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        Eigen::Vector2d position(state(0), state(1)); // Assuming the first two elements of the state are x and y position
        return Eigen::VectorXd::Constant(1, position.squaredNorm()); 
    }

    Eigen::VectorXd getLowerBound() const override {
        return Eigen::VectorXd::Constant(1, 0.0); 
    }

    Eigen::VectorXd getUpperBound() const override {
        return Eigen::VectorXd::Constant(1, radius_ * radius_); 
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        Eigen::MatrixXd jacobian(1, state.size());
        jacobian << 2 * state(0), 2 * state(1), Eigen::RowVectorXd::Zero(state.size() - 2); // Assuming x and y are the first two state elements
        return jacobian;
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Zero(1, control.size()); 
    }

private:
    double radius_;
};

/**
* @brief Log barrier function for constrained optimization
* 
* Implements a log barrier interior point method to handle inequality constraints.
*/
class LogBarrier {
public:
   /**
    * @brief Types of bounds to consider in barrier function
    */
   enum BoundType {
       UPPER_BOUND,  ///< Only consider upper bounds
       LOWER_BOUND,  ///< Only consider lower bounds  
       BOTH_BOUNDS   ///< Consider both upper and lower bounds
   };

   /**
    * @brief Construct a log barrier with given parameters
    * @param barrier_coeff Coefficient controlling barrier steepness
    * @param relaxation_coeff Relaxation factor for numerical stability 
    * @param barrier_order Order of barrier polynomial (1=linear, 2=quadratic)
    * @param bound_type Which bounds to include in barrier
    */
   LogBarrier(double barrier_coeff = 1e-2, double relaxation_coeff = 1.0,
              int barrier_order = 2, BoundType bound_type = BOTH_BOUNDS);

   /**
    * @brief Evaluate barrier function for given constraint
    * @param constraint Constraint being enforced
    * @param state Current state
    * @param control Current control input
    * @return Barrier function value
    */
   double evaluate(const Constraint& constraint, 
                  const Eigen::VectorXd& state,
                  const Eigen::VectorXd& control) const;

   /**
    * @brief Get state and control gradients of barrier
    * @param constraint Constraint being enforced
    * @param state Current state
    * @param control Current control input 
    * @param is_relaxed Whether constraint is in relaxed region
    * @return Tuple of state and control gradients
    */
   std::tuple<Eigen::VectorXd, Eigen::VectorXd> getGradients(
           const Constraint& constraint, 
           const Eigen::VectorXd& state,
           const Eigen::VectorXd& control,
           bool is_relaxed) const;

   /**
    * @brief Get Hessians of barrier function
    * @param constraint Constraint being enforced  
    * @param state Current state
    * @param control Current control input
    * @param is_relaxed Whether constraint is in relaxed region
    * @return Tuple of state, control and cross Hessians
    */
   std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getHessians(
           const Constraint& constraint,
           const Eigen::VectorXd& state, 
           const Eigen::VectorXd& control,
           bool is_relaxed) const;

   /**
    * @brief Get current barrier coefficient
    * @return Current barrier coefficient
    */
   double getBarrierCoeff() {
        return barrier_coeff_;
   }

   /**
    * @brief Set barrier coefficient to new value
    * @param barrier_coeff New barrier coefficient value
    */ 
   void setBarrierCoeff(double barrier_coeff) {
        barrier_coeff_ = barrier_coeff;
   }

    /**
     * @brief Get current relaxation coefficient
     * @return Current relaxation coefficient
     */
    double getRelaxationCoeff() {
        return relaxation_coeff_;
    }

    /**
     * @brief Set relaxation coefficient to new value
     * @param relaxation_coeff New relaxation coefficient value
     */
    void setRelaxationCoeff(double relaxation_coeff) {
        relaxation_coeff_ = relaxation_coeff;
    }

    /**
     * @brief Get current barrier order
     * @return Current barrier order
     */
    BoundType getBoundType() {
        return bound_type_;
    }

    /**
     * @brief Set barrier order to new value
     * @param bound_type New bound type
     */
    void setBoundType(BoundType bound_type) {
        bound_type_ = bound_type;
    }


private:
   double barrier_coeff_;    ///< Coefficient controlling barrier steepness
   double relaxation_coeff_; ///< Relaxation factor for numerical stability  
   int barrier_order_;       ///< Order of barrier polynomial
   BoundType bound_type_;     ///< Which bounds to include
};

} // namespace cddp

#endif // CDDP_CONSTRAINT_HPP