import numpy as np
from numba import cuda
import sys
from bvp_problem import _abvp_f, _abvp_g, _abvp_Df, _abvp_Dg, _abvp_r
import matrix_operation_cuda
import matrix_factorization_cuda
from math import *


TPB = 32


def compute_residual_parallel(N, stages, size_y, size_z, size_p, t_span, y0, z0, p, alpha0, rk, tol):
    d_x_next, d_lte = dae_integration_parallel(N, stages, size_y, size_z, size_p, t_span, y0, z0, p, alpha0, rk, tol)
    d_g = compute_dae_parallel(N, size_y, size_z, size_p, y0, z0, p, alpha0)
    d_b = np.zeros((N - 1, size_z + size_y))
    f_s = np.zeros((N - 1, size_z + size_y))
    # compute the residual at each time node
    for i in range(N - 1):
        for j in range(size_z):
            d_b[i, j] = -d_g[i, j]
            f_s[i, j] = -d_b[i, j]
        for j in range(size_y):
            d_b[i, size_z + j] = -(d_x_next[i, j] - y0[i + 1, j])
            f_s[i, size_z + j] = -d_b[i, size_z + j]
    # compute the boundary conditions
    r_bc = np.zeros((size_y + size_p), dtype=np.float64)
    # this boundary function is currently on CPU
    _abvp_r(y0[0, 0: size_y], y0[N - 1, 0: size_y], p, r_bc)
    b_n = np.concatenate([-d_g[N - 1, 0: size_z], -r_bc])
    # form the residual vector of the system
    f_s = np.ravel(f_s.reshape(-1, 1))
    f_s = np.concatenate([f_s, -b_n])
    return f_s, d_b, b_n, d_lte


def dae_integration_parallel(N, stages, size_y, size_z, size_p, t_span, y0, z0, p, alpha0, rk, tol):
    # warp dimension for CUDA kernel
    grid_dims = (N + TPB - 1) // TPB
    # transfer the memory from CPU to GPU
    d_t_span = cuda.to_device(t_span)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_p = cuda.to_device(p)
    # create holders for intermediate variables
    d_M = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_Df = cuda.device_array((N * size_y, (size_y + size_z + size_p)), dtype=np.float64)
    d_Dg = cuda.device_array((N * size_z, (size_y + size_z + size_p)), dtype=np.float64)
    d_Wj = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_v = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_x_stage = cuda.device_array((N * (size_y + size_z), stages), dtype=np.float64)
    d_k = cuda.device_array((N * (size_y + size_z), stages), dtype=np.float64)
    d_f = cuda.device_array((N * (size_y + size_z), stages), dtype=np.float64)
    d_gamma = cuda.to_device(rk.gamma)
    d_alpha = cuda.to_device(rk.alpha)
    d_b = cuda.to_device(rk.b)
    d_e = cuda.to_device(rk.e)
    d_M_gamma_W = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_cpy = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_P = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_L = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_U = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    d_f_v = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_P_f_v = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_y_f_v = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_x_f_v = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_sum1 = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_sum2 = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_sum3 = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_sum4 = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_W_sum2 = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_error_sum = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_sigma = cuda.device_array((N, 1), dtype=np.float64)
    d_fac = cuda.device_array((N, 1), dtype=np.float64)
    # holder for the output
    d_x_next = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_lte = cuda.device_array((N, 1), dtype=np.float64)
    dae_integration_row_kernel[grid_dims, TPB](N, stages, size_y, size_z, size_p, alpha0,
                                               d_t_span, d_y0, d_z0, d_p, tol,
                                               d_M, d_Df, d_Dg, d_Wj, d_v, d_x_stage, d_k, d_f,
                                               d_gamma, d_alpha, d_b, d_e,
                                               d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps,
                                               d_f_v, d_P_f_v, d_y_f_v, d_x_f_v,
                                               d_sum1, d_sum2, d_sum3, d_sum4, d_W_sum2, d_error_sum,
                                               d_sigma, d_fac,
                                               d_x_next, d_lte)
    return d_x_next.copy_to_host(), d_lte.copy_to_host()


@cuda.jit
def dae_integration_row_kernel(N, stages, size_y, size_z, size_p, alpha0, d_t_span, d_y0, d_z0, d_p, tol,
                               d_M, d_Df, d_Dg, d_Wj, d_v, d_x_stage, d_k, d_f,
                               d_gamma, d_alpha, d_b, d_e,
                               d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps,
                               d_f_v, d_P_f_v, d_y_f_v, d_x_f_v,
                               d_sum1, d_sum2, d_sum3, d_sum4, d_W_sum2, d_error_sum,
                               d_sigma, d_fac,
                               d_x_next, d_lte):
    # cuda thread index
    i = cuda.grid(1)
    if i < (N - 1):
        dae_integration_row(stages, size_y, size_z, size_p, alpha0, d_t_span[i + 1] - d_t_span[i],
                            d_y0[i, 0: size_y], d_z0[i, 0: size_z], d_p, tol,
                            d_M[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            d_Df[i * size_y: (i + 1) * size_y, 0: size_y + size_z + size_p],
                            d_Dg[i * size_z: (i + 1) * size_z, 0: size_y + size_z + size_p],
                            d_Wj[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            d_v[i, 0: size_y + size_z],
                            d_x_stage[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: stages],
                            d_k[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: stages],
                            d_f[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: stages],
                            d_gamma, d_alpha, d_b, d_e,
                            d_M_gamma_W[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            d_cpy[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            d_P[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            d_L[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            d_U[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                            eps,
                            d_f_v[i, 0: size_y + size_z], d_P_f_v[i, 0: size_y + size_z],
                            d_y_f_v[i, 0: size_y + size_z], d_x_f_v[i, 0: size_y + size_z],
                            d_sum1[i, 0: size_y + size_z], d_sum2[i, 0: size_y + size_z],
                            d_sum3[i, 0: size_y + size_z], d_sum4[i, 0: size_y + size_z],
                            d_W_sum2[i, 0: size_y + size_z],
                            d_error_sum[i, 0: size_y + size_z],
                            d_sigma[i], d_fac[i],
                            d_x_next[i, 0: size_y + size_z], d_lte[i])
    elif i == N - 1:
        for j in range(size_y + size_z):
            d_x_next[i, j] = 0.0
        d_lte[i] = 0.0
    return


@cuda.jit(device=True)
def dae_integration_row(stages, size_y, size_z, size_p, alpha, delta, d_y0, d_z0, d_p, tol,
                        d_M, d_Df, d_Dg, d_Wj, d_v, d_x_stage, d_k, d_f,
                        d_gamma, d_alpha, d_b, d_e,
                        d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps,
                        d_f_v, d_P_f_v, d_y_f_v, d_x_f_v,
                        d_sum1, d_sum2, d_sum3, d_sum4, d_W_sum2, d_error_sum,
                        d_sigma, d_fac,
                        d_x_next, d_lte):
    """
    :param stages: number of stages of the Runge-Kutta method
    :param size_y: number of ODE variables
    :param size_z: number of DAE variables
    :param size_p: number of parameter variables
    :param alpha: continuation parameter
    :param delta: length of the time interval
    :param d_y0: shape: size_y
                 values of the ODE variables at the initial time
    :param d_z0: shape: size_z
                 values of the ODE variables at the initial time
    :param d_p: shape: size_p
                values of the parameter variables
    :param tol: convergence tolerance of the algorithm
    :param d_M: shape: (size_y + size_z) x (size_y + size_z)
            holder for mass matrix of the DAEs
    :param d_Df: shape: size_y x (size_y + size_z + size_p)
                 holder for derivatives of ODEs w.r.t its variables
    :param d_Dg: shape: size_z x (size_y + size_z + size_p)
                 holder for derivatives of DAEs w.r.t its variables
    :param d_Wj: shape: (size_y + size_z) x (size_y + size_z)
                holder for Jacobian of the DAE at the DAE integration
    :param d_v: shape: (size_y + size_z)
                residual in solving the DAEs
    :param d_x_stage: shape: (size_y + size_z) x stages
                      holder for ODE and DAE variables at each stage
    :param d_k: shape: (size_y + size_z) x stages
                       holder for derivatives at each stage
    :param d_f: shape: (size_y + size_z) x stages
                       intermediate results of DAE derivatives at each stage
    :param d_gamma: shape: stages x stages
    :param d_alpha: shape: stages x stages
    :param d_b: shape: stages
    :param d_e: shape: stages
    :param d_M_gamma_W: shape: (size_y + size_z) x (size_y + size_z)
                        temporary holder
    :param d_cpy: shape: (size_y + size_z) x (size_y + size_z)
                  holders for the copy of the matrix while performing LU decomposition
    :param d_P: shape: (size_y + size_z) x (size_y + size_z)
                holder for the permutation matrix in LU decomposition
    :param d_L: shape: (size_y + size_z) x (size_y + size_z)
                holder for the lower triangular matrix in LU decomposition
    :param d_U: shape: (size_y + size_z) x (size_y + size_z)
                holder for the upper triangular matrix in LU decomposition
    :param eps: machine epsilon
    :param d_f_v: shape: (size_y + size_z)
    :param d_P_f_v: shape: (size_y + size_z)
    :param d_y_f_v: shape: (size_y + size_z)
    :param d_x_f_v: shape: (size_y + size_z)
    :param d_sum1: shape: (size_y + size_z)
    :param d_sum2: shape: (size_y + size_z)
    :param d_sum3: shape: (size_y + size_z)
    :param d_sum4: shape: (size_y + size_z)
    :param d_W_sum2: shape: (size_y + size_z)
    :param d_error_sum: shape: (size_y + size_z)
    :param d_sigma: should be an one-element array
    :param d_fac: should be an one-element array
    :return:
    :param d_x_next: shape: (size_y + size_z)
                     integrated values of ODE and DAE variable at the next time node
    :param d_lte: should be an one-element array which records the local truncation error of the integration
    """
    # single step integration of the DAEs
    # create the Mass matrix for DAEs
    for i in range(size_y):
        for j in range(size_y + size_z):
            if i == j:
                d_M[i, j] = 1.0
            else:
                d_M[i, j] = 0.0
    for i in range(size_z):
        for j in range(size_y + size_z):
            d_M[i + size_y, j] = 0.0
    # zero initialize d_Df and d_Dg
    for j in range(size_y + size_z + size_p):
        for i in range(size_y):
            d_Df[i, j] = 0.0
        for i in range(size_z):
            d_Dg[i, j] = 0.0
    # compute the Jacobian of ODEs
    _abvp_Df(d_y0, d_z0, d_p, d_Df)
    # compute the Jacobian of DAEs
    _abvp_Dg(d_y0, d_z0, d_p, alpha, d_Dg)
    # create the Jacobian for DAEs
    for i in range(size_y):
        for j in range(size_y + size_z):
            d_Wj[i, j] = d_Df[i, j]
    for i in range(size_z):
        for j in range(size_y + size_z):
            d_Wj[i + size_y, j] = d_Dg[i, j]
    for i in range(size_y + size_z):
        for j in range(size_y + size_z):
            d_M_gamma_W[i, j] = d_M[i, j] - d_gamma[0, 0] * delta * d_Wj[i, j]
    # lu decompose the Jacobian matrix
    matrix_factorization_cuda.lu(d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps)
    # compute the residual for DAE to be consistent
    for i in range(size_y):
        d_v[i] = 0.0
    # v_j(ny + 1 : end) = DAE_g(y0, z0, p, alpha0)
    _abvp_g(d_y0, d_z0, d_p, alpha, d_v[size_y: size_y + size_z])
    # x_stage(:, 0) = x0
    for i in range(size_y):
        d_x_stage[i, 0] = d_y0[i]
    for i in range(size_z):
        d_x_stage[size_y + i, 0] = d_z0[i]
    # f(:, 1) = [ODE_h(y0, z0, p, alpha0) DAE_g(y0, z0, p, alpha0)]
    _abvp_f(d_y0, d_z0, d_p, d_f[0: size_y, 0])
    _abvp_g(d_y0, d_z0, d_p, alpha, d_f[size_y: size_y + size_z, 0])
    # k(:, 0) = U \ (L \ (P * (f(:, 0) - v_j)))
    for i in range(size_y + size_z):
        d_f_v[i] = d_f[i, 0] - d_v[i]
    # compute P * (f(:, 0) - v_j)) first, the result of the product is saved in d_P_f_v
    matrix_operation_cuda.mat_vec_mul(d_P, d_f_v, d_P_f_v)
    # first, forward solve the linear system L * (U * X) = P * (f(:, 0) - v_j)), and the result is saved in d_y_f_v
    matrix_factorization_cuda.forward_solve_vec(d_L, d_P_f_v, d_y_f_v, eps)
    # then, backward solve the linear system U * X = Y, and the result is saved in d_x_f_v
    matrix_factorization_cuda.backward_solve_vec(d_U, d_y_f_v, d_x_f_v, eps)
    # save the result to d_k
    for i in range(size_y + size_z):
        d_k[i, 0] = d_x_f_v[i]
    # start integration on each stage
    for i in range(1, stages):
        # zero-initialize the temporary variables
        for j in range(size_y + size_z):
            d_sum1[j] = 0.0
            d_sum2[j] = 0.0
        for l in range(i):
            # sum1 = sum1 + coefficients.alpha(i, l) * k(:, l)
            # sum2 = sum2 + coefficients.gamma(i, l) * k(:, l)
            for j in range(size_y + size_z):
                d_sum1[j] += d_alpha[i, l] * d_k[j, l]
                d_sum2[j] += d_gamma[i, l] * d_k[j, l]
        # x_stage(:, i) = x0 + delta * sum1
        for j in range(size_y):
            d_x_stage[j, i] = d_y0[j] + delta * d_sum1[j]
        for j in range(size_z):
            d_x_stage[size_y + j, i] = d_z0[j] + delta * d_sum1[size_y + j]
        # f(:, i) = [ODE_h(y_i, z_i, p, alpha0) DAE_g(y_i, z_i, p, alpha0)]
        _abvp_f(d_x_stage[0: size_y, i], d_x_stage[size_y: size_y + size_z, i], d_p, d_f[0: size_y, i])
        _abvp_g(d_x_stage[0: size_y, i], d_x_stage[size_y: size_y + size_z, i], d_p, alpha,
                d_f[size_y: size_y + size_z, i])
        # k(:, i) = U \ (L \ (f(:, i) + delta * W_j * sum2 - v_j))
        # compute W_j * sum2 first
        matrix_operation_cuda.mat_vec_mul(d_Wj, d_sum2, d_W_sum2)
        for j in range(size_y + size_z):
            d_f_v[j] = d_f[j, i] + delta * d_W_sum2[j] - d_v[j]
        # compute P * (f(:, i) + delta * W_j * sum2 - v_j) first, the result of the product is saved in d_P_f_v
        matrix_operation_cuda.mat_vec_mul(d_P, d_f_v, d_P_f_v)
        # first, forward solve the linear system L * (U * X) = P * (f(:, i) + delta * W_j * sum2 - v_j),
        # and the result is saved in d_y_f_v
        matrix_factorization_cuda.forward_solve_vec(d_L, d_P_f_v, d_y_f_v, eps)
        # then, backward solve the linear system U * X = Y, and the result is saved in d_x_f_v
        matrix_factorization_cuda.backward_solve_vec(d_U, d_y_f_v, d_x_f_v, eps)
        # save the result to d_k
        for j in range(size_y + size_z):
            d_k[j, i] = d_x_f_v[j]
    # zero-initialize the temporary variable
    for i in range(size_y + size_z):
        d_sum3[i] = 0.0
    for i in range(stages):
        for j in range(size_y + size_z):
            d_sum3[j] += d_b[i] * d_k[j, i]
    # compute the result of the integration x_next
    # x_next = x0 + delta * sum3
    for i in range(size_y):
        d_x_next[i] = d_y0[i] + delta * d_sum3[i]
    for i in range(size_z):
        d_x_next[i + size_y] = d_z0[i] + delta * d_sum3[i + size_y]

    # compute the local truncation error of the integration result
    d_sigma[0] = 0.0
    for i in range(size_y + size_z):
        d_sum4[i] = 0.0
    for i in range(stages):
        for j in range(size_y + size_z):
            d_sum4[j] += d_e[i] * d_k[j, i]
    for i in range(size_y + size_z):
        d_error_sum[i] = delta * d_sum4[i]
    for i in range(size_y + size_z):
        d_fac[0] = abs(d_error_sum[i]) / (tol * (1.0 + abs(d_x_next[i])))
        d_sigma[0] += d_fac[0] * d_fac[0]
    d_lte[0] = sqrt(d_sigma[0] / (size_y + size_z))
    return


def compute_dae_parallel(N, size_y, size_z, size_p, y0, z0, p, alpha0):
    # warp dimension for CUDA kernel
    grid_dims = (N + TPB - 1) // TPB
    # transfer memory from CPU to GPU
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_p = cuda.to_device(p)
    # create holder for output
    d_g = cuda.device_array((N, size_z), dtype=np.float64)
    compute_dae_kernel[grid_dims, TPB](N, size_y, size_z, size_p, d_y0, d_z0, d_p, alpha0, d_g)
    return d_g.copy_to_host()


@cuda.jit
def compute_dae_kernel(N, size_y, size_z, size_p, d_y0, d_z0, d_p, alpha0, d_g):
    # cuda thread index
    i = cuda.grid(1)
    if i < N:
        _abvp_g(d_y0[i, 0: size_y], d_z0[i, 0: size_z], d_p, alpha0,
                d_g[i, 0: size_z])
