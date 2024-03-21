# Constrained Differential Dynamic Programming (CDDP) solver in C++
**This library is actively under construction.**
## Overview
This is an optimal control solver library using constrained differential dynamic programming (CDDP) written in C++ based on Xie's [paper](https://zhaomingxie.github.io/projects/CDDP/CDDP.pdf). This library is particularly useful for mobile robot trajectory optimization and model predictive control (MPC).

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

**Author: Tomohiro Sasaki** 

## Installation
### Dependencies
* [Eigen](https://formulae.brew.sh/formula/eigen) (Linear Algebra Library in CPP)
    
```bash
sudo apt-get install libeigen3-dev # For Ubuntu
brew install eigen # For macOS
```

Although the library automatically finds the following dependencies and installs if you do not have ones, here is how you can install on your own.

* [OSQP](https://osqp.org/) (QP solver) and [osqp-cpp](https://github.com/google/osqp-cpp) (C++ Wrapper for OSQP)
```bash
conda install -c conda-forge osqp
```

### Building
```bash
git clone https://github.com/astomodynamics/CDDP-cpp
cd CDDP-cpp
mkdir build
cd build
cmake ..
make
make test
```

## Basic Usage


## ROS 
If you want to use this library for ROS2 MPC node, please refer [CDDP MPC Package](https://github.com/astomodynamics/cddp_mpc_pkg)

## References
* Zhaoming Xie, C. Karen Liu, and Kris Hauser, "Differential Dynamic Programming with Nonlinear Constraints," 2017 IEEE International Conference on Robotics and Automation (ICRA), 2017.


## Citing
If you use this work in an academic context, please cite this repository.

## TODO
* add state constraint handling
* add regularization process
* add initial guess functionality based on path planner solution
* add automatic differentiation
* add nonlinear objective and constraint handling
