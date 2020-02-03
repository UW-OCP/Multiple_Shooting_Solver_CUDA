import numpy as np
from compute_residual_parallel import compute_residual_parallel, compute_residual_parallel_gpu
import time


def line_search_ms(
        N, stages, size_y, size_z, size_p, t_span, y0, z0, p0, alpha, rk, tol,
        delta_s, delta_p, max_line_search, norm_f_s):
    """
    Perform the line-search to find the right descending direction for the Newton's method
    in multiple shooting algorithm.
    :param N:
    :param stages:
    :param size_y:
    :param size_z:
    :param size_p:
    :param t_span:
    :param y0:
    :param z0:
    :param p0:
    :param alpha:
    :param rk:
    :param tol:
    :param delta_s:
    :param delta_p:
    :param max_line_search:
    :param norm_f_s:
    :return:
    err: error code for the line search, if error is 0, the line search succeeds; else if error is 1, the line search
         fails to find the right descending direction.
    y_new: new ODE variables after line search; if line search fails, original y0 is returned.
    z_new: new DAE variables after line search; if line search fails, original z0 is returned.
    p_new: new parameter variables after line search; if line search fails, original p0 is returned.
    """
    # perform line search to find the direction
    err = 0
    alpha0 = 1
    delta_y = delta_s[:, 0: size_y]
    delta_z = delta_s[:, size_y:]
    t_residual = 0
    n_residual = 0
    for i in range(max_line_search):
        y_new = y0 + alpha0 * delta_y
        z_new = z0 + alpha0 * delta_z
        p_new = p0 + alpha0 * delta_p
        start_time_residual = time.time()
        # compute the residual
        norm_f_s_new, _, _, _ = compute_residual_parallel_gpu(
            N, stages, size_y, size_z, size_p, t_span, y_new, z_new, p_new, alpha, rk, tol)
        t_residual += (time.time() - start_time_residual)
        n_residual += 1
        if norm_f_s_new < norm_f_s:
            return err, y_new, z_new, p_new, t_residual, n_residual
        alpha0 /= 2
    # line search fails
    err = 1
    return err, y0, z0, p0, t_residual, n_residual
