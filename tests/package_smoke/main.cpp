#include <cddp-cpp/cddp.hpp>
#include <cddp-cpp/cddp_core/helper.hpp>

int main() {
  const auto finite = cddp::finite_difference_jacobian(
      [](const Eigen::VectorXd &x) { return x; }, Eigen::VectorXd::Zero(1));
  return finite.rows() == 1 ? 0 : 1;
}
