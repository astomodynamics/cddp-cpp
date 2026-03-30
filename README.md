# Constrained Differential Dynamic Programming (CDDP) solver in C++
<img src="docs/assets/cddp_in_cpp.png" width="800" alt="CDDP IN CPP">

**This library is under active development.**  Star the repo :star: to stay updated on its progress and eventual release. I welcome any suggestions as I'm building this library to gain practical C++ experience.

## Overview
This is an optimal control solver library using constrained differential dynamic programming (CDDP) written in C++. This library is particularly useful for mobile robot trajectory optimization and model predictive control (MPC).

The CDDP library solves problems in the form:

$$
\min_{\mathbf{U}} J(\mathbf{x}_0, \mathbf{U}) = \phi(\mathbf{x}_N) + \sum \ell(\mathbf{x}_k,\mathbf{u}_k)
$$

$$
\mathrm{s.t.~}  \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k,\mathbf{u}_k) 
$$

$$
\quad \mathbf{g}(\mathbf{x}_k,\mathbf{u})_k\leq 0
$$

$$
\quad {\mathbf{x}}_{0} = \mathbf{x}{(t_0)} 
$$

$$
\quad k = 0,1,\cdots N-1
$$

## Examples
The default C++ build currently includes a barrier-strategy comparison example:

```bash
./examples/test_barrier_strategies
```

Legacy visualization-heavy C++ examples remain in the repository as reference
material, but they are no longer part of the default build. Python bindings are
the supported plotting and notebook workflow.

## Installation
### Dependencies
* [CMake](https://cmake.org/) (Build System)
```bash
sudo apt-get install cmake # For Ubuntu
brew install cmake # For macOS
```

* [Eigen](https://formulae.brew.sh/formula/eigen) (Linear Algebra Library in CPP)
```bash
sudo apt-get install libeigen3-dev # For Ubuntu
brew install eigen # For macOS
```

### Building
```bash
git clone https://github.com/astomodynamics/cddp-cpp
cd cddp-cpp
mkdir build && cd build
cmake ..
make -j4
make test
```

## ROS 
If you want to use this library for ROS2 MPC node, please refer [CDDP MPC Package](https://github.com/astomodynamics/cddp_mpc). You do not need to install this library to use the package. MPC package will automatically install this library as a dependency.

## References
* Y. Tassa, N. Mansard and E. Todorov, "Control-limited differential dynamic programming," 2014 IEEE International Conference on Robotics and Automation (ICRA), 2014, pp. 1168-1175, doi: <10.1109/ICRA.2014.6907001>.
* Pavlov, A., Shames, I., and Manzie, C., “Interior Point Differential Dynamic Programming,” IEEE Transactions on Control Systems Technology, Vol. 29, No. 6, 2021, pp. 2720–2727.
* Yuval Tassa's iLQG/DDP trajectory optimization: <https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization>
* Andrei Pavlov's GitHub repository: <https://github.com/xapavlov/ipddp>

## Third Party Libraries

This library uses the following open-source libraries as core dependencies:

* [autodiff](https://github.com/autodiff/autodiff) (MIT License)

This library also uses the following open-source libraries for optional features:

* [Ipopt](https://github.com/coin-or/Ipopt) (EPL License)
* [CasADi](https://web.casadi.org/) (GPL License)


## Citing
If you use this work in an academic context, please cite this repository.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Collaboration
Contributions to this CDDP solver library are welcome! Whether you're interested in fixing bugs, adding new features, improving documentation, or suggesting new example applications, your input is valuable.

If you'd like to contribute:
1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bug-fix`.
3.  **Make your changes** and ensure they are well-tested.
4.  **Commit your changes:** `git commit -m 'Add some amazing feature'`.
5.  **Push to the branch:** `git push origin feature/your-feature-name`.
6.  **Open a Pull Request** against the `main` (or `master`) branch.

If you have ideas for collaboration, want to discuss potential research applications, or have any questions, please feel free to open an issue on GitHub or reach out to the @astomodynamics. We are particularly interested in exploring its use in novel robotic systems and complex motion planning scenarios.

## TODO
* improve python binding ergonomics
* improve parallelization
* add simulation examples and Python visualizations
  * Quadruped robot
  * Manipulator
  * Humanoid
