import numpy as np
from bvp_problem import _abvp_f, _abvp_g, _abvp_Df, _abvp_Dg, _abvp_r, _abvp_Dr
import sys
from numba import cuda
import matrix_operation_cuda
import matrix_factorization_cuda


TPB = 32


def construct_jacobian_parallel(N, stages, size_y, size_z, size_p, t_span, y0, z0, p, alpha0, rk):
    # warp dimension for CUDA kernel
    grid_dims = ((N - 1) + TPB - 1) // TPB

    _, d_X_next = sensitivity_integration_parallel(N, stages, size_y, size_z, size_p, t_span, y0, z0, p, alpha0, rk)

    d_Dg, d_Dg_N = compute_dae_jacobian(N, size_y, size_z, size_p, t_span, y0, z0, p, alpha0)

    # holder for output variables
    d_A = cuda.device_array(((N - 1) * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_C = cuda.device_array(((N - 1) * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_H = cuda.device_array(((N - 1) * (size_y + size_z), size_p), dtype=np.float64)

    construct_jacobian_kernel[grid_dims, TPB](N, size_y, size_z, size_p, d_X_next, d_Dg, d_A, d_C, d_H)

    # compute the boundary conditions
    r_bc = np.zeros((size_y + size_p), dtype=np.float64)
    # this boundary function is currently on CPU
    _abvp_r(y0[0, 0: size_y], y0[N - 1, 0: size_y], p, r_bc)

    # compute the derivative of the boundary conditions
    d_r = np.zeros(((size_y + size_p), (size_y + size_y + size_p)), dtype=np.float64)
    _abvp_Dr(y0[0, 0: size_y], y0[N - 1, 0: size_y], p, d_r)
    # B_1 in the paper
    B_0 = np.zeros(((size_y + size_z + size_p), (size_y + size_z)), dtype=np.float64)
    B_0[size_z: size_y + size_z + size_p, 0: size_y] = d_r[0: size_y + size_p, 0: size_y]
    # transfer the jacobian of the final node back to CPU
    d_Dg_N = d_Dg_N.copy_to_host()
    # B_N in the paper
    B_n = np.zeros(((size_y + size_z + size_p), (size_y + size_z)), dtype=np.float64)
    B_n[0: size_z, 0: size_y + size_z] = d_Dg_N[-size_z:, 0: size_y + size_z]
    B_n[size_z: size_y + size_z + size_p, 0: size_y] = d_r[0: size_y + size_p, size_y: size_y + size_y]
    # H_N in the paper
    H_n = np.zeros(((size_y + size_z + size_p), size_p), dtype=np.float64)
    H_n[0: size_z, 0: size_p] = d_Dg_N[-size_z:, -size_p:]
    H_n[size_z: size_z + size_y + size_p, 0: size_p] = \
        d_r[0: size_y + size_p, size_y + size_y: size_y + size_y + size_p]

    return d_A.copy_to_host(), d_C.copy_to_host(), d_H.copy_to_host(), B_0, B_n, H_n


@cuda.jit
def construct_jacobian_kernel(N, size_y, size_z, size_p, d_X_next, d_Dg, d_A, d_C, d_H):
    # cuda thread index
    i = cuda.grid(1)
    if i < (N - 1):
        construct_jacobian_single_node(size_y, size_z, size_p,
                                       d_X_next[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (
                                               size_y + size_z + size_p)],
                                       d_Dg[i * size_z: (i + 1) * size_z, 0: (
                                               size_y + size_z + size_p)],
                                       d_A[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (size_y + size_z)],
                                       d_C[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (size_y + size_z)],
                                       d_H[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_p])
    return


@cuda.jit(device=True)
def construct_jacobian_single_node(size_y, size_z, size_p, d_X_next, d_Dg, d_A, d_C, d_H):
    # construct block A
    for i in range(size_z):
        for j in range(size_y + size_z):
            d_A[i, j] = d_Dg[i, j]
    for i in range(size_y):
        for j in range(size_y + size_z):
            d_A[i + size_z, j] = d_X_next[i, j]
    # construct block H
    for i in range(size_z):
        for j in range(size_p):
            d_H[i, j] = d_Dg[i, size_y + size_z + j]
    for i in range(size_y):
        for j in range(size_p):
            d_H[i + size_z, j] = d_X_next[i, size_y + size_z + j]
    # construct block C
    for i in range(size_z):
        for j in range(size_y + size_z):
            d_C[i, j] = 0.0
    for i in range(size_y):
        for j in range(size_y + size_z):
            if i == j:
                d_C[size_z + i, j] = -1.0
            else:
                d_C[size_z + i, j] = 0.0
    return


def sensitivity_integration_parallel(N, stages, size_y, size_z, size_p, t_span, y0, z0, p, alpha0, rk):
    # warp dimension for CUDA kernel
    grid_dims = ((N - 1) + TPB - 1) // TPB
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
    d_W_sum2 = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_X0 = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_V = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_X_stage = cuda.device_array((N * (size_y + size_z), stages * (size_y + size_z + size_p)), dtype=np.float64)
    d_W = cuda.device_array((N * (size_y + size_z), (size_y + size_z)), dtype=np.float64)
    d_K = cuda.device_array((N * (size_y + size_z), stages * (size_y + size_z + size_p)), dtype=np.float64)
    d_F_p = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_W_X = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_W_X_F_V = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_P_W_X_F_V = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_Y_W_X_F_V = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_X_W_X_F_V = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_Sum1 = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_Sum2 = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_Sum3 = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    d_W_Sum2 = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    # return values
    d_x_next = cuda.device_array((N, (size_y + size_z)), dtype=np.float64)
    d_X_next = cuda.device_array((N * (size_y + size_z), (size_y + size_z + size_p)), dtype=np.float64)
    sensitivity_integration_row_kernel[grid_dims, TPB](N, stages, size_y, size_z, size_p, alpha0,
                                                       d_t_span, d_y0, d_z0, d_p,
                                                       d_M, d_Df, d_Dg, d_Wj, d_v, d_x_stage, d_k, d_f,
                                                       d_gamma, d_alpha, d_b,
                                                       d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps,
                                                       d_f_v, d_P_f_v, d_y_f_v, d_x_f_v,
                                                       d_sum1, d_sum2, d_sum3, d_W_sum2,
                                                       d_X0, d_V, d_X_stage, d_W, d_K, d_F_p,
                                                       d_W_X, d_W_X_F_V, d_P_W_X_F_V, d_Y_W_X_F_V, d_X_W_X_F_V,
                                                       d_Sum1, d_Sum2, d_Sum3, d_W_Sum2,
                                                       d_x_next, d_X_next)
    return d_x_next, d_X_next


@cuda.jit
def sensitivity_integration_row_kernel(N, stages, size_y, size_z, size_p, alpha, d_t_span, d_y0, d_z0, d_p,
                                       d_M, d_Df, d_Dg, d_Wj, d_v, d_x_stage, d_k, d_f,
                                       d_gamma, d_alpha, d_b,
                                       d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps,
                                       d_f_v, d_P_f_v, d_y_f_v, d_x_f_v,
                                       d_sum1, d_sum2, d_sum3, d_W_sum2,
                                       d_X0, d_V, d_X_stage, d_W, d_K, d_F_p,
                                       d_W_X, d_W_X_F_V, d_P_W_X_F_V, d_Y_W_X_F_V, d_X_W_X_F_V,
                                       d_Sum1, d_Sum2, d_Sum3, d_W_Sum2,
                                       d_x_next, d_X_next):
    """
    :param N: number of nodes in the mesh
    :param stages: number of stages of teh Runge-Kutta method
    :param size_y: number of ODE variables
    :param size_z: number of DAE variables
    :param size_p: number of parameter variables
    :param alpha: continuation parameter
    :param d_t_span: mesh of the time span of the problem
    :param d_y0: shape: N x size_y
                 values of the ODE variables; row dominated order where row i is the value of ODE variables
                 at time node i
    :param d_z0: shape: N x size_z
                 values of the DAE variables; row dominated order where row i is the value of DAE variables
                 at time node i
    :param d_p: shape: size_p
                values of the parameter variables
    :param d_M: shape: N * (size_y + size_z) x (size_y + size_z)
                holder for mass matrix of DAEs; row dominated order where row i * (size_y + size_z) to
                row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_Df: shape: N * size_y x (size_y + size_z + size_p)
                 holder for derivatives of ODEs; row dominated order where row i * size_y to
                 row (i + 1) * size_y corresponds to time node i
    :param d_Dg: shape: N * size_z x (size_y + size_z + size_p)
                 holder for derivatives of DAEs; row dominated order where row i * size_z to
                 row (i + 1) * size_z corresponds to time node i
    :param d_Wj: shape: N * (size_y + size_z) x (size_y + size_z)
                holder for Jacobian of whole DAEs; row dominated order where row i * (size_y + size_z) to
                row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_v: shape: N x (size_y + size_z)
                holder for residuals when solving DAEs to make the initial consistent conditions; row dominated order
                where row i is the value of the residual at time node i
    :param d_x_stage: shape N * (size_y + size_z) x stages
                      holder for ODE and DAE variables at each stage; row dominated order where row i * (size_y + size_z)
                      to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_k: shape N * (size_y + size_z) x stages
                holder for ODE and DAE derivatives at each stage; row dominated order where row i * (size_y + size_z)
                to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_f: shape N * (size_y + size_z) x stages
                holder for results of derivatives at each stage; row dominated order where row i * (size_y + size_z)
                to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_gamma: coefficients from RK
    :param d_alpha: coefficients from RK
    :param d_b: coefficients from RK
    :param d_M_gamma_W: shape: N * (size_y + size_z) x (size_y + size_z)
                        temporary holder; row dominated order where row i * (size_y + size_z)
                        to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_cpy: shape: N * (size_y + size_z) x (size_y + size_z)
                  holders for the copy of the matrix while performing LU decomposition; row dominated order where row
                  i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_P: shape: N * (size_y + size_z) x (size_y + size_z)
                holder for the permutation matrix in LU decomposition; row dominated order where row
                  i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_L: shape: N * (size_y + size_z) x (size_y + size_z)
                holder for the lower triangular matrix in LU decomposition; row dominated order where row
                  i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_U: shape: N * (size_y + size_z) x (size_y + size_z)
                holder for the upper triangular matrix in LU decomposition; row dominated order where row
                  i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param eps: machine epsilon
    :param d_f_v: shape: N x (size_y + size_z)
                  temporary holder; row dominated
    :param d_P_f_v: shape: N x (size_y + size_z)
    :param d_y_f_v: shape: N x (size_y + size_z)
    :param d_x_f_v: shape: N x (size_y + size_z)
    :param d_sum1: shape: N x (size_y + size_z)
    :param d_sum2: shape: N x (size_y + size_z)
    :param d_sum3: shape: N x (size_y + size_z)
    :param d_W_sum2: shape: N x (size_y + size_z)
    :param d_X0: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                 initial value of the DAE sensitivities; row dominated order where row
                 i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_V: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                residual in solving DAE sensitivities; row dominated order where row
                i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_X_stage: shape: N * (size_y + size_z) x stages * (size_y + size_z + size_p)
                      intermediate results of DAE sensitivities at each stage; row dominated order where row
                      i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_W: shape: N * (size_y + size_z) x (size_y + size_z)
                holder for Jacobian of the DAE at the DAE sensitivity integration; row dominated order where row
                i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_K: shape: N * (size_y + size_z) x stages * (size_y + size_z + size_p)
                intermediate results of DAE sensitivity derivatives at each stage; row dominated order where row
                i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_F_p: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                  sensitivities w.r.t. parameter variables; row dominated order where row
                  i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_W_X: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                  holder for the product of W and X_stage; row dominated order where row
                  i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_W_X_F_V: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                      holder for the residual in the linear equation of DAE sensitivities; row dominated order where row
                      i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_P_W_X_F_V: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                        holder for the product of P and the residual in the linear equation; row dominated order where row
                        i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_Y_W_X_F_V: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                        holder for the intermediate solution in LU solve; row dominated order where row
                        i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_X_W_X_F_V: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                        holder for the final solution in LU solve; row dominated order where row
                        i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_Sum1: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                   holder for temporary sum during computing DAE sensitivities; row dominated order where row
                   i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_Sum2: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                   holder for temporary sum during computing DAE sensitivities; row dominated order where row
                   i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_Sum3: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                   holder for temporary sum during computing DAE sensitivities; row dominated order where row
                   i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :param d_W_Sum2: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                     holder for temporary product of W and Sum2; row dominated order where row
                     i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    :return:
    :param d_x_next: shape: N x (size_y + size_z)
                     integrated values of ODE and DAE variable at the next time node; row dominated order where row i
                     is the value of ODE variables at time node i
    :param d_X_next: shape: N * (size_y + size_z) x (size_y + size_z + size_p)
                     integrated values of DAE sensitivities at the next time node; row dominated order where row
                     i * (size_y + size_z) to row (i + 1) * (size_y + size_z) corresponds to time node i
    """
    # cuda thread index
    i = cuda.grid(1)
    if i < (N - 1):
        sensitivity_integration_row(stages, size_y, size_z, size_p, alpha,
                                    d_t_span[i + 1] - d_t_span[i], d_y0[i, 0: size_y], d_z0[i, 0: size_z], d_p,
                                    d_M[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    d_Df[i * size_y: (i + 1) * size_y, 0: size_y + size_z + size_p],
                                    d_Dg[i * size_z: (i + 1) * size_z, 0: size_y + size_z + size_p],
                                    d_Wj[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    d_v[i, 0: size_y + size_z],
                                    d_x_stage[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: stages],
                                    d_k[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: stages],
                                    d_f[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: stages],
                                    d_gamma, d_alpha, d_b,
                                    d_M_gamma_W[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    d_cpy[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    d_P[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    d_L[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    d_U[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: size_y + size_z],
                                    eps,
                                    d_f_v[i, 0: size_y + size_z], d_P_f_v[i, 0: size_y + size_z],
                                    d_y_f_v[i, 0: size_y + size_z], d_x_f_v[i, 0: size_y + size_z],
                                    d_sum1[i, 0: size_y + size_z], d_sum2[i, 0: size_y + size_z],
                                    d_sum3[i, 0: size_y + size_z],
                                    d_W_sum2[i, 0: size_y + size_z],
                                    d_X0[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_V[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_X_stage[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: stages * (size_y + size_z + size_p)],
                                    d_W[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (size_y + size_z)],
                                    d_K[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: stages * (size_y + size_z + size_p)],
                                    d_F_p[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_W_X[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_W_X_F_V[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_P_W_X_F_V[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_Y_W_X_F_V[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_X_W_X_F_V[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_Sum1[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_Sum2[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_Sum3[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_W_Sum2[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p],
                                    d_x_next[i, 0: size_y + size_z],
                                    d_X_next[i * (size_y + size_z): (i + 1) * (size_y + size_z),
                                    0: size_y + size_z + size_p])
    elif i == N - 1:
        for j in range(size_y + size_z):
            d_x_next[i, j] = 0.0
            for k in range(size_y + size_z + size_p):
                d_X_next[i * (size_y + size_z) + j, k] = 0.0
        return


@cuda.jit(device=True)
def sensitivity_integration_row(stages, size_y, size_z, size_p, alpha, delta, d_y0, d_z0, d_p,
                                d_M, d_Df, d_Dg, d_Wj, d_v, d_x_stage, d_k, d_f,
                                d_gamma, d_alpha, d_b,
                                d_M_gamma_W, d_cpy, d_P, d_L, d_U, eps,
                                d_f_v, d_P_f_v, d_y_f_v, d_x_f_v,
                                d_sum1, d_sum2, d_sum3, d_W_sum2,
                                d_X0, d_V, d_X_stage, d_W, d_K, d_F_p,
                                d_W_X, d_W_X_F_V, d_P_W_X_F_V, d_Y_W_X_F_V, d_X_W_X_F_V,
                                d_Sum1, d_Sum2, d_Sum3, d_W_Sum2,
                                d_x_next, d_X_next):
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
    :param d_W_sum2: shape: (size_y + size_z)
    :param d_X0: shape: (size_y + size_z) x (size_y + size_z + size_p)
                 initial value of the DAE sensitivities
    :param d_V: shape: (size_y + size_z) x (size_y + size_z + size_p)
                residual in solving DAE sensitivities
    :param d_X_stage: shape: (size_y + size_z) x stages * (size_y + size_z + size_p)
                      intermediate results of DAE sensitivities at each stage
    :param d_W: shape: (size_y + size_z) x (size_y + size_z)
                holder for Jacobian of the DAE at the DAE sensitivity integration
    :param d_K: shape: shape: (size_y + size_z) x stages * (size_y + size_z + size_p)
                       intermediate results of DAE sensitivity derivatives at each stage
    :param d_F_p: shape: (size_y + size_z) x (size_y + size_z + size_p)
                  sensitivities w.r.t. parameter variables
    :param d_W_X: shape: (size_y + size_z) x (size_y + size_z + size_p)
                  holder for the product of W and X_stage
    :param d_W_X_F_V: shape: (size_y + size_z) x (size_y + size_z + size_p)
                      holder for the residual in the linear equation of DAE sensitivities
    :param d_P_W_X_F_V: shape: (size_y + size_z) x (size_y + size_z + size_p)
                        holder for the product of P and the residual in the linear equation
    :param d_Y_W_X_F_V: shape: (size_y + size_z) x (size_y + size_z + size_p)
                        holder for the intermediate solution in LU solve
    :param d_X_W_X_F_V: shape: (size_y + size_z) x (size_y + size_z + size_p)
                        holder for the final solution in LU solve
    :param d_Sum1: shape: (size_y + size_z) x (size_y + size_z + size_p)
                   holder for temporary sum during computing DAE sensitivities
    :param d_Sum2: shape: (size_y + size_z) x (size_y + size_z + size_p)
                   holder for temporary sum during computing DAE sensitivities
    :param d_Sum3: shape: (size_y + size_z) x (size_y + size_z + size_p)
                   holder for temporary sum during computing DAE sensitivities
    :param d_W_Sum2: shape: (size_y + size_z) x (size_y + size_z + size_p)
                     holder for temporary product of W and Sum2
    :return:
    :param d_x_next: shape: (size_y + size_z)
                     integrated values of ODE and DAE variable at the next time node
    :param d_X_next: shape: (size_y + size_z) x (size_y + size_z + size_p)
                     integrated values of DAE sensitivities at the next time node
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

    ##########
    # single step integration for the sensitivities of DAEs
    ##########
    # V_j = zeros(ny + nz, ny + nz + np)
    for i in range(size_y):
        for j in range(size_y + size_z + size_p):
            d_V[i, j] = 0.0
    # V_j(1 + ny : end, :) = [g_y g_z g_p]
    # compute the Jacobian of DAEs
    _abvp_Dg(d_y0, d_z0, d_p, alpha, d_Dg)
    for i in range(size_z):
        for j in range(size_y + size_z + size_p):
            d_V[i + size_y, j] = d_Dg[i, j]
    # X0 = zeros(ny + nz, ny + nz + np)
    # X0(1  : ny, 1 : ny) = eye(ny)
    for i in range(size_y + size_z):
        for j in range(size_y + size_z + size_p):
            if i == j:
                d_X0[i, j] = 1.0
            else:
                d_X0[i, j] = 0.0
    # X_stage(:, 1 : (ny + nz + np)) = X0
    for i in range(size_y + size_z):
        for j in range(size_y + size_z + size_p):
            d_X_stage[i, j] = d_X0[i, j]
    # F_p(:, 1 + ny + nz : end) = [h_p g_p]
    # zero initialize F_p
    for i in range(size_y):
        for j in range(size_y + size_z):
            d_F_p[i, j] = 0.0
        for j in range(size_p):
            d_F_p[i, size_y + size_z + j] = d_Df[i, size_y + size_z + j]
    for i in range(size_z):
        for j in range(size_y + size_z):
            d_F_p[size_y + i, j] = 0.0
        for j in range(size_p):
            d_F_p[size_y + i, size_y + size_z + j] = d_Dg[i, size_y + size_z + j]
    # W = jacobian_DAE(tspan, x_stage(:, 1), p, alpha0)
    # compute the Jacobian of ODEsd_W
    _abvp_Df(d_x_stage[0: size_y, 0], d_x_stage[size_y: size_y + size_z, 0], d_p, d_Df)
    # compute the Jacobian of DAEs
    _abvp_Dg(d_x_stage[0: size_y, 0], d_x_stage[size_y: size_y + size_z, 0], d_p, alpha, d_Dg)
    # create the Jacobian for DAEs
    for i in range(size_y):
        for j in range(size_y + size_z):
            d_W[i, j] = d_Df[i, j]
    for i in range(size_z):
        for j in range(size_y + size_z):
            d_W[i + size_y, j] = d_Dg[i, j]
    # K(:, 1 : (ny + nz + np)) = U \ (L \ (W * X_stage(:, 1 : (ny + nz + np)) + F_p - V_j))
    # compute W * X_stage
    matrix_operation_cuda.mat_mul(d_W,
                                  d_X_stage[:, 0: (size_y + size_z + size_p)],
                                  d_W_X)
    # compute the residual in the linear equation
    for i in range(size_y + size_z):
        for j in range(size_y + size_z + size_p):
            d_W_X_F_V[i, j] = d_W_X[i, j] + d_F_p[i, j] - d_V[i, j]
    # compute P * (W * X_stage(:, 1 : (ny + nz + np)) + F_p - V_j) first,
    # the result of the product is saved in d_P_W_X_F_V
    matrix_operation_cuda.mat_mul(d_P, d_W_X_F_V, d_P_W_X_F_V)
    # first, forward solve the linear system L * (U * X) = P * (W * X_stage(:, 1 : (ny + nz + np)) + F_p - V_j),
    # and the result is saved in d_Y_W_X_F_V
    matrix_factorization_cuda.forward_solve_mat(d_L, d_P_W_X_F_V, d_Y_W_X_F_V, eps)
    # then, backward solve the linear system U * X = Y, and the result is saved in d_X_W_X_F_V
    matrix_factorization_cuda.backward_solve_mat(d_U, d_Y_W_X_F_V, d_X_W_X_F_V, eps)
    # copy the result
    for i in range(size_y + size_z):
        for j in range(size_y + size_z + size_p):
            d_K[i, j] = d_X_W_X_F_V[i, j]
    for i in range(1, stages):
        for j in range(size_y + size_z):
            for k in range(size_y + size_z + size_p):
                d_Sum1[j, k] = 0.0
                d_Sum2[j, k] = 0.0
        # for l = 1:i - 1
        for l in range(i):
            # Sum1 = Sum1 + coefficients.alpha(i, l) * K(:, 1 + (l - 1) * (ny + nz + np): l * (ny + nz + np))
            # Sum2 = Sum2 + coefficients.gamma(i, l) * K(:, 1 + (l - 1) * (ny + nz + np): l * (ny + nz + np))
            for j in range(size_y + size_z):
                for k in range(size_y + size_z + size_p):
                    d_Sum1[j, k] += d_alpha[i, l] * d_K[j, l * (size_y + size_z + size_p) + k]
                    d_Sum2[j, k] += d_gamma[i, l] * d_K[j, l * (size_y + size_z + size_p) + k]
        # X_stage(:, 1 + (i - 1) * (ny + nz + np): i * (ny + nz + np)) = X0 + delta * Sum1
        for j in range(size_y + size_z):
            for k in range(size_y + size_z + size_p):
                d_X_stage[j, i * (size_y + size_z + size_p) + k] = d_X0[j, k] + delta * d_Sum1[j, k]
        # [h_y, h_z, h_p, g_y, g_z, g_p] = difference_DAE(x_stage(:, i), p, alpha0)
        # compute the Jacobian of ODEs
        _abvp_Df(d_x_stage[0: size_y, i], d_x_stage[size_y: size_y + size_z, i], d_p, d_Df)
        # compute the Jacobian of DAEs
        _abvp_Dg(d_x_stage[0: size_y, i], d_x_stage[size_y: size_y + size_z, i], d_p, alpha, d_Dg)
        # F_p(:, 1 + ny + nz: end) = [h_p g_p]
        for j in range(size_y):
            for k in range(size_p):
                d_F_p[j, size_y + size_z + k] = d_Df[j, size_y + size_z + k]
        for j in range(size_z):
            for k in range(size_p):
                d_F_p[size_y + j, size_y + size_z + k] = d_Dg[j, size_y + size_z + k]
        # W = jacobian_DAE(tspan, x_stage(:, i), p, alpha0)
        # create the Jacobian for DAEs
        for j in range(size_y):
            for k in range(size_y + size_z):
                d_W[j, k] = d_Df[j, k]
        for j in range(size_z):
            for k in range(size_y + size_z):
                d_W[size_y + j, k] = d_Dg[j, k]
        # K(:, 1 + (i - 1) * (ny + nz + np): i * (ny + nz + np)) =
        # U \ (L \ (W *  X_stage(:, 1 + (i - 1) * (ny + nz + np): i * (ny + nz + np)) + F_p - V_j + delta * W_j * Sum2))
        # compute W * X_stage
        matrix_operation_cuda.mat_mul(d_W,
                                      d_X_stage[:, i * (size_y + size_z + size_p): (i + 1) * (
                                              size_y + size_z + size_p)],
                                      d_W_X)
        # compute W_j * Sum2
        matrix_operation_cuda.mat_mul(d_Wj, d_Sum2, d_W_Sum2)
        # compute the residual in the linear equation
        for j in range(size_y + size_z):
            for k in range(size_y + size_z + size_p):
                d_W_X_F_V[j, k] = d_W_X[j, k] + d_F_p[j, k] - d_V[j, k] + delta * d_W_Sum2[j, k]
        # compute P * (W *  X_stage + F_p - V_j + delta * W_j * Sum2) first,
        # the result of the product is saved in d_P_W_X_F_V
        matrix_operation_cuda.mat_mul(d_P, d_W_X_F_V, d_P_W_X_F_V)
        # first, forward solve the linear system L * (U * X) = d_P_W_X_F_V,
        # and the result is saved in d_Y_W_X_F_V
        matrix_factorization_cuda.forward_solve_mat(d_L, d_P_W_X_F_V, d_Y_W_X_F_V, eps)
        # then, backward solve the linear system U * X = Y, and the result is saved in d_X_W_X_F_V
        matrix_factorization_cuda.backward_solve_mat(d_U, d_Y_W_X_F_V, d_X_W_X_F_V, eps)
        # copy the result
        for j in range(size_y + size_z):
            for k in range(size_y + size_z + size_p):
                d_K[j, i * (size_y + size_z + size_p) + k] = d_X_W_X_F_V[j, k]
    # for i =1: coefficients.stages
    #     Sum3 = Sum3 + coefficients.b(i) * K(:, 1 + (i - 1) * (ny + nz + np): i * (ny + nz + np))
    # end
    # zero-initialize Sum3
    for j in range(size_y + size_z):
        for k in range(size_y + size_z + size_p):
            d_Sum3[j, k] = 0.0
    for i in range(stages):
        for j in range(size_y + size_z):
            for k in range(size_y + size_z + size_p):
                d_Sum3[j, k] += d_b[i] * d_K[j, i * (size_y + size_z + size_p) + k]
    # X_next = X0 + delta * Sum3
    for j in range(size_y + size_z):
        for k in range(size_y + size_z + size_p):
            d_X_next[j, k] = d_X0[j, k] + delta * d_Sum3[j, k]
    return


def compute_dae_jacobian(N, size_y, size_z, size_p, t_span, y0, z0, p, alpha0):
    # warp dimension for CUDA kernel
    grid_dims = (N + TPB - 1) // TPB
    # transfer the memory from CPU to GPU
    d_t_span = cuda.to_device(t_span)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_p = cuda.to_device(p)
    # create holders for output
    d_Dg = cuda.device_array((N * size_z, (size_y + size_z + size_p)), dtype=np.float64)
    d_Dg_N = cuda.device_array((size_z, (size_y + size_z + size_p)), dtype=np.float64)
    compute_dae_jacobian_kernel[grid_dims, TPB](
        N, size_y, size_z, size_p, d_t_span, d_y0, d_z0, d_p, alpha0, d_Dg, d_Dg_N)
    return d_Dg, d_Dg_N


@cuda.jit
def compute_dae_jacobian_kernel(N, size_y, size_z, size_p, d_t_span, d_y0, d_z0, d_p, alpha0, d_Dg, d_Dg_N):
    # cuda thread index
    i = cuda.grid(1)
    if i < (N - 1):
        # zero initialize the array first
        for j in range(size_z):
            for k in range(size_y + size_z + size_p):
                d_Dg[i * size_z + j, k] = 0.0
        _abvp_Dg(d_y0[i, 0: size_y], d_z0[i, 0: size_z], d_p, alpha0,
                 d_Dg[i * size_z: (i + 1) * size_z, 0: (size_y + size_z + size_p)])
    elif i == N - 1:
        # zero initialize the array first
        for j in range(size_z):
            for k in range(size_y + size_z + size_p):
                d_Dg[i * size_z + j, k] = 0.0
                d_Dg_N[j, k] = 0.0
        _abvp_Dg(d_y0[i, 0: size_y], d_z0[i, 0: size_z], d_p, alpha0,
                 d_Dg[i * size_z: (i + 1) * size_z, 0: (size_y + size_z + size_p)])
        _abvp_Dg(d_y0[i, 0: size_y], d_z0[i, 0: size_z], d_p, alpha0,
                 d_Dg_N[0: size_z, 0: (size_y + size_z + size_p)])
    return
