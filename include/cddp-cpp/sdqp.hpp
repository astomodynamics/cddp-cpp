/*
    MIT License

    Copyright (c) 2022 Zhepei Wang (wangzhepei@live.com)

    Modified by Tomo Sasaki (tomo.sasaki.hiro@gmail.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef CDDP_SDQP_HPP
#define CDDP_SDQP_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <random>

namespace sdqp
{
    const double eps = 1.0e-12;

    enum
    {
        MINIMUM = 0,
        INFEASIBLE,
    };

    inline void set_zero(double *x, const int d) {
        for (int i = 0; i < d; ++i)
        {
            x[i] = 0.0;
        }
    }

    inline double dot(const double *x,
                      const double *y, 
                      const int d){
        double s = 0.0;
        for (int i = 0; i < d; ++i)
        {
            s += x[i] * y[i];
        }
        return s;
    }

    inline double sqr_norm(const double *x, const int d)
    {
        double s = 0.0;
        for (int i = 0; i < d; ++i)
        {
            s += x[i] * x[i];
        }
        return s;
    }

    inline void mul(const double *x,
                    const double s,
                    double *y,
                    const int d) {
        for (int i = 0; i < d; ++i)
        {
            y[i] = x[i] * s;
        }
        return;
    }

    inline int max_abs(const double *x, const int d)
    {
        int id = 0;
        double mag = std::fabs(x[0]);
        for (int i = 1; i < d; ++i)
        {
            const double s = std::fabs(x[i]);
            if (s > mag)
            {
                id = i;
                mag = s;
            }
        }
        return id;
    }

    inline void cpy(const double *x,
                    double *y, 
                    const int d) {
        for (int i = 0; i < d; ++i)
        {
            y[i] = x[i];
        }
        return;
    }

    inline int move_to_front(const int i,
                             int *next,
                             int *prev)
    {
        if (i == 0 || i == next[0])
        {
            return i;
        }
        const int previ = prev[i];
        next[prev[i]] = next[i];
        prev[next[i]] = prev[i];
        next[i] = next[0];
        prev[i] = 0;
        prev[next[i]] = i;
        next[0] = i;
        return previ;
    }

    int min_norm(const double *halves,
                const int n,
                const int m,
                double *opt,
                double *work,
                int *next,
                int *prev,
                const int d) {
        
        if (d == 1)
        {
            opt[0] = 0.0;
            bool l = false;
            bool r = false;

            for (int i = 0; i != m; i = next[i])
            {
                const double a = halves[2 * i];
                const double b = halves[2 * i + 1];
                if (a * opt[0] + b > 2.0 * eps)
                {
                    if (std::fabs(a) < 2.0 * eps)
                    {
                        return INFEASIBLE;
                    }

                    l = l || a < 0.0;
                    r = r || a > 0.0;

                    if (l && r)
                    {
                        return INFEASIBLE;
                    }

                    opt[0] = -b / a;
                }
            }

            return MINIMUM;
        } else {    
            int status = MINIMUM;
            set_zero(opt, d);

            if (m <= 0)
            {
                return status;
            }

            double *reflx = work;
            double *new_opt = reflx + d;
            double *new_halves = new_opt + (d - 1);
            double *new_work = new_halves + n * d;

            for (int i = 0; i != m; i = next[i])
            {
                const double *plane_i = halves + (d + 1) * i;

                if (dot(opt, plane_i, d) + plane_i[d] > (d + 1) * eps)
                {
                    const double s = sqr_norm(plane_i, d);

                    if (s < (d + 1) * eps * eps)
                    {
                        return INFEASIBLE;
                    }

                    mul(plane_i, -plane_i[d] / s, opt, d);

                    if (i == 0)
                    {
                        continue;
                    }

                    // stable Householder reflection with pivoting
                    const int id = max_abs(opt, d);
                    const double xnorm = std::sqrt(sqr_norm(opt, d));
                    cpy(opt, reflx, d);
                    reflx[id] += opt[id] < 0.0 ? -xnorm : xnorm;
                    const double h = -2.0 / sqr_norm(reflx, d);

                    for (int j = 0; j != i; j = next[j])
                    {
                        double *new_plane = new_halves + d * j;
                        const double *old_plane = halves + (d + 1) * j;
                        const double coeff = h * dot(old_plane, reflx, d);
                        for (int k = 0; k < d; ++k)
                        {
                            const int l = k < id ? k : k - 1;
                            new_plane[l] = k != id ? old_plane[k] + reflx[k] * coeff : new_plane[l];
                        }
                        new_plane[d - 1] = dot(opt, old_plane, d) + old_plane[d];
                    }

                    status = min_norm(new_halves, n, i, new_opt, new_work, next, prev, d - 1);

                    if (status == INFEASIBLE)
                    {
                        return INFEASIBLE;
                    }

                    double coeff = 0.0;
                    for (int j = 0; j < d; ++j)
                    {
                        const int k = j < id ? j : j - 1;
                        coeff += j != id ? reflx[j] * new_opt[k] : 0.0;
                    }
                    coeff *= h;
                    for (int j = 0; j < d; ++j)
                    {
                        const int k = j < id ? j : j - 1;
                        opt[j] += j != id ? new_opt[k] + reflx[j] * coeff : reflx[j] * coeff;
                    }

                    i = move_to_front(i, next, prev);
                }
            }


            return status;

        }
        return MINIMUM;
    }

    inline void rand_permutation(const int n,
                                 int *p)
    {
        typedef std::uniform_int_distribution<int> rand_int;
        typedef rand_int::param_type rand_range;
        static std::mt19937_64 gen;
        static rand_int rdi(0, 1);
        int j, k;
        for (int i = 0; i < n; ++i)
        {
            p[i] = i;
        }
        for (int i = 0; i < n; ++i)
        {
            rdi.param(rand_range(0, n - i - 1));
            j = rdi(gen) + i;
            k = p[j];
            p[j] = p[i];
            p[i] = k;
        }
    }

    inline double sdmn(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, const int d)
    {
        x.setZero();
        const int n = b.size();
        if (n < 1)
        {
            return 0.0;
        }

         // Debugging output
        std::cout << "Initial x: " << x.transpose() << std::endl;
        std::cout << "Matrix A: \n" << A << std::endl;
        std::cout << "Vector b: " << b.transpose() << std::endl;

        Eigen::VectorXi perm(n - 1);
        Eigen::VectorXi next(n);
        Eigen::VectorXi prev(n + 1);
        if (n > 1) {
            // Random permutation
            rand_permutation(n - 1, perm.data());

            prev(0) = 0;
            next(0) = perm(0) + 1;
            prev(perm(0) + 1) = 0;
            for (int i = 0; i < n - 2; ++i)
            {
                next(perm(i) + 1) = perm(i + 1) + 1;
                prev(perm(i + 1) + 1) = perm(i) + 1;
            }
            next(perm(n - 2) + 1) = n;
        } else {
            prev(0) = 0;
            next(0) = 1;
            next(1) = 1;
        }

        // Debugging output
        std::cout << "perm: " << perm.transpose() << std::endl;
        std::cout << "next: " << next.transpose() << std::endl;
        std::cout << "prev: " << prev.transpose() << std::endl;


        Eigen::MatrixXd halves(A.cols() + 1, n);
        Eigen::VectorXd work((n + 2) * (d + 2) * (d - 1) / 2 + 1 - d);

        // Scaling A's rows
        const Eigen::VectorXd scale = A.rowwise().norm();
        halves.topRows(A.cols()) = (A.array().colwise() / scale.array()).transpose();
        halves.bottomRows(1) = (-b.array() / scale.array()).transpose();

        // Debugging output
        std::cout << "Scaling factors (Code 1): " << scale.transpose() << std::endl;
        std::cout << "Halves matrix (Code 1): \n" << halves << std::endl;

        const int status = min_norm(halves.data(), n, n,
                                   x.data(), work.data(),
                                   next.data(), prev.data(), d);

        // Debugging output
        std::cout << "Status after min_norm: " << status << std::endl;
        std::cout << "x after min_norm: " << x.transpose() << std::endl;

        double minimum = INFINITY;
        if (status != INFEASIBLE)
        {
            minimum = x.norm();
        }
        std::cout << "Minimum value: " << minimum << std::endl;
        return minimum;
    }

    inline double sdqp(const Eigen::MatrixXd &Q, const Eigen::VectorXd &c, const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::VectorXd &x)
    {
        Eigen::LLT<Eigen::MatrixXd> llt(Q);
        if (llt.info() != Eigen::Success)
        {
            return INFINITY;
        }

        const Eigen::MatrixXd As = llt.matrixU().solve<Eigen::OnTheRight>(A); 
        const Eigen::VectorXd v = llt.solve(c);
        const Eigen::VectorXd bs = A * v + b;

        double minimum = sdmn(As, bs, x, Q.cols());
        std::cout << "minimum: " << minimum << std::endl;
        if (!std::isinf(minimum))
        {
            llt.matrixU().solveInPlace(x);
            x -= v;
            minimum = 0.5 * x.dot(Q * x) + c.dot(x);
        }

        return minimum;
    }

} // namespace sdqp

#endif // CDDP_SDQP_HPP
