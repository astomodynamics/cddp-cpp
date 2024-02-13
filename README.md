# DDP-cpp
Implementation of iLQR (Iterative Linear Quadratic Regulator) algorithm for trajectory optimization, based on [Yuval Tassa's Matlab implementation](https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization).


## Usage

* Install eigen or download into `include` (https://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
* `mkdir build; cd build`
* `cmake ..` 
* `make`
* Define a dynamics and cost model based on Model (see double_integrator example), or run with `./run_iLQR acrobot`

## References

* Tassa, Yuval, Nicolas Mansard, and Emo Todorov. "Control-limited differential dynamic programming." Robotics and Automation (ICRA), 2014 IEEE International Conference on. IEEE, 2014.
* Li, Weiwei, and Emanuel Todorov. "Iterative linear quadratic regulator design for nonlinear biological movement systems." ICINCO (1). 2004.
* https://github.com/kazuotani14/iLQR?tab=readme-ov-file