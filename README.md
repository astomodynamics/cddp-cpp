# Constrained Differential Dynamic Programming (CDDP) solver in C++

## Overview
This is an optimal control solver library using constrained differential dynamic programming (CDDP) written in C++ based on Xie's [paper](https://zhaomingxie.github.io/projects/CDDP/CDDP.pdf) and [implementation](https://github.com/ZhaomingXie/CDDP). This library is particularly useful for mobile robot trajectory optimization and model predictive control (MPC).

This library is still under construction. 

**Author: Tomohiro Sasaki** 

## Installation
### Dependencies
* [Eigen](https://formulae.brew.sh/formula/eigen) (Linear Algebra Library in CPP)
    
```bash
sudo apt-get install libeigen3-dev # For Ubuntu
brew install eigen # For macOS
```

Although the library automatically finds the following dependencies and installs if you do not have ones, here is how you can install on your own.

* [OSQP](https://osqp.org/) (QP solver) and [OSQP-Eigen](https://robotology.github.io/osqp-eigen/) (C++ OSQP Wrapper)
```bash
conda install -c conda-forge osqp
conda install -c conda-forge osqp-eigen
```

### Building
```bash
git clone https://github.com/astomodynamics/CDDP-cpp 
mkdir build
cd build
cmake ..
make
# make install
```

## Basic Usage


## References
* Zhaoming Xie, C. Karen Liu, and Kris Hauser, "Differential Dynamic Programming with Nonlinear Constraints," 2017 IEEE International Conference on Robotics and Automation (ICRA), 2017.


## Citing
If you use this work in an academic context, please cite this repository.

## TODO
