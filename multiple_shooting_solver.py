import bvp_problem
import rk_coefficients
import numpy as np
from construct_jacobian_parallel import construct_jacobian_parallel
from compute_residual_parallel import compute_residual_parallel, compute_residual_parallel_gpu
from solve_babd_system import partition_factorization_parallel, construct_babd_mshoot, qr_decomposition, \
    backward_substitution, partition_backward_substitution_parallel, recover_babd_solution
from line_search_parallel import line_search_ms
from mesh_refinement import mesh_refinement
import time
from plot_result import plot_result
from mesh_sanity_check import mesh_sanit_check
from BVPDAEReadWriteData import bvpdae_write_data
import pathlib


def multiple_shooting_solver():
    # construct the bvp-dae problem
    # obtain the initial input
    bvp_dae = bvp_problem.BvpDae()
    success_flag = 0
    size_y = bvp_dae.size_y
    size_z = bvp_dae.size_z
    size_p = bvp_dae.size_p
    size_inequality = bvp_dae.size_inequality
    size_sv_inequality = bvp_dae.size_sv_inequality
    t_span0 = bvp_dae.T0
    N = t_span0.shape[0]
    y0 = bvp_dae.Y0
    z0 = bvp_dae.Z0
    p0 = bvp_dae.P0
    # parameters for numerical solvers
    tol = bvp_dae.tolerance
    max_iter = bvp_dae.maximum_newton_iterations
    max_iter = 500
    max_mesh = bvp_dae.maximum_mesh_refinements
    max_nodes = bvp_dae.maximum_nodes
    min_nodes = 3
    max_line_search = 20
    output_file = bvp_dae.output_file
    alpha = 1  # continuation parameter
    if size_inequality > 0 or size_sv_inequality > 0:
        alpha_m = 1e-6
    else:
        alpha_m = 1
    beta = 0.8  # scale factor
    # specify collocation coefficients
    m = 3  # number of collocation points
    rk = rk_coefficients.RKCoefficients(flag=1)
    stages = rk.stages
    M = 4  # number of blocks used to solve the BABD system in parallel

    para0 = np.copy(p0)
    t_span = np.copy(t_span0)
    success_flag = 1

    # benchmark data
    N_residual = 0
    T_residual = 0
    N_Jacobian = 0
    T_Jacobian = 0
    N_babd = 0
    T_babd = 0

    start_time = time.time()
    for alpha_iter in range(max_iter):
        print("Solving alpha = {}:".format(alpha))
        newton_iter = 0
        mesh_it = 0
        d_lte = np.ones(N - 1)
        for newton_iter in range(max_iter):
            start_time_jacobian = time.time()
            # construct the jacobian
            d_A, d_C, d_H, B_0, B_n, H_n = construct_jacobian_parallel(
                N, stages, size_y, size_z, size_p, t_span, y0, z0, para0, alpha, rk)
            T_Jacobian += (time.time() - start_time_jacobian)
            N_Jacobian += 1
            start_time_residual = time.time()
            # compute the residual
            norm_f_s, d_b, b_n, d_lte = compute_residual_parallel_gpu(
                N, stages, size_y, size_z, size_p, t_span, y0, z0, para0, alpha, rk, tol)
            T_residual += (time.time() - start_time_residual)
            N_residual += 1
            if norm_f_s < tol:
                print('alpha = {}, solution is found. Number of nodes: {}'.format(alpha, N))
                break
            start_time_babd = time.time()
            # solve the BABD system
            # perform the partition factorization on the Jacobian matrix with qr decomposition
            index, R, E, J_reduced, G, d, A_tilde, C_tilde, H_tilde, b_tilde = \
                partition_factorization_parallel(size_y + size_z, size_p, M, N, d_A, d_C, d_H, d_b)
            # construct the partitioned Jacobian system
            sol = construct_babd_mshoot(size_y, size_z, size_p, M, A_tilde, C_tilde, H_tilde, b_tilde, B_0, B_n, H_n, b_n)
            # perform the qr decomposition to transfer the system
            qr_decomposition(size_y + size_z, size_p, M + 1, sol)
            # perform the backward substitution to obtain the solution to the linear system of Newton's method
            backward_substitution(M + 1, sol)
            # obtain the solution from the reduced BABD system
            delta_s_r, delta_p = recover_babd_solution(M, size_y, size_z, size_p, sol)
            # get the solution to the BABD system
            delta_s = partition_backward_substitution_parallel(
                size_y + size_z, size_p, M, N, index, delta_s_r, delta_p, R, G, E, J_reduced, d)
            T_babd += (time.time() - start_time_babd)
            N_babd += 1
            if np.isnan(delta_s).any() or np.isnan(delta_p).any() or np.amax(delta_s) > 1 / tol or \
                    np.amax(delta_s) > 1 / tol:
                N, t_span, y0, z0 = mesh_refinement(N, size_y, size_z, tol, d_lte, t_span, y0, z0)
                print("\tSearch direction is not right, remeshed the problem. Number of nodes = {}".format(N))
                mesh_it += 1
                mesh_sanit_check(N, mesh_it, max_mesh, max_nodes, min_nodes)
                if N > 10000 or mesh_it > 20 or N < min_nodes:
                    break
            else:
                # perform line search to find the direction
                err, y0, z0, para0, t_residual, n_residual = line_search_ms(
                    N, stages, size_y, size_z, size_p, t_span, y0, z0, para0, alpha, rk, tol,
                    delta_s, delta_p, max_line_search, norm_f_s)
                T_residual += t_residual
                N_residual += n_residual
                if err != 0:
                    N, t_span, y0, z0 = mesh_refinement(N, size_y, size_z, tol, d_lte, t_span, y0, z0)
                    print("\tLine search failed. Remeshed the problem. Number of nodes = {}".format(N))
                    mesh_it += 1
                    mesh_sanit_check(N, mesh_it, max_mesh, max_nodes, min_nodes)
                    if N > 10000 or mesh_it > 20 or N < min_nodes:
                        success_flag = 0
                        break
        if newton_iter >= (max_iter - 1) or N > 10000 or mesh_it > 20 or N < min_nodes:
            print("alpha = {}, reach the maximum iteration numbers and the problem does not converge!".format(alpha))
            success_flag = 0
            break
        while np.amax(d_lte) > 1:
            N, t_span, y0, z0 = mesh_refinement(N, size_y, size_z, tol, d_lte, t_span, y0, z0)
            print("\tLTE = {}. Remeshed the problem. Number of nodes = {}".format(np.amax(d_lte), N))
            mesh_it += 1
            mesh_sanit_check(N, mesh_it, max_mesh, max_nodes, min_nodes)
            if N > 10000 or mesh_it > 20 or N < min_nodes:
                success_flag = 0
                break
            start_time_residual = time.time()
            # compute the residuaDo
            _, _, _, d_lte = compute_residual_parallel_gpu(
                N, stages, size_y, size_z, size_p, t_span, y0, z0, para0, alpha, rk, tol)
            T_residual += (time.time() - start_time_residual)
            N_residual += 1
        print("\tAlpha = {}, solution is found with LTE = {}. Number of nodes = {}".format(alpha, np.amax(d_lte), N))
        if alpha <= alpha_m:
            print("Final solution is found, alpha = {}. Number of nodes: {}".format(alpha, N))
            break
        alpha *= beta
    solver_elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(solver_elapsed_time))
    # write benchmark data
    example_name = output_file.split(".")
    example_name = example_name[0]
    benchmark_dir = "./benchmark_results/"
    # create the directory
    pathlib.Path(benchmark_dir).mkdir(0o755, parents=True, exist_ok=True)
    benchmark_file = "./benchmark_results/" + example_name + "_benchmark_M_{}.data".format(M)
    with open(benchmark_file, 'w') as f:
        f.write("Number of times of computing residual: {}\n".format(N_residual))
        f.write("Running time of computing residual: {}\n".format(T_residual))
        f.write("Number of times of constructing Jacobian: {}\n".format(N_Jacobian))
        f.write("Running time of constructing Jacobian: {}\n".format(T_Jacobian))
        f.write("Number of times of solving BABD: {}\n".format(N_babd))
        f.write("Running time of solving BABD: {}\n".format(T_babd))
        f.write("Running time of solver: {}\n".format(solver_elapsed_time))
    # record the solved example
    with open('test_results.txt', 'a') as f:
        if alpha <= alpha_m and success_flag:
            f.write("{} solved successfully. alpha = {}.\n".format(example_name, alpha))
        else:
            f.write("{} solved unsuccessfully. alpha = {}.\n".format(example_name, alpha))
    if alpha <= alpha_m and success_flag:
        # write and plot the result only when problem is solved successfully.
        # write solution to the output file
        error = bvpdae_write_data(output_file, N, size_y, size_z, size_p, t_span, y0, z0, para0)
        if error != 0:
            print("Write file failed.")
        plot_result(size_y, size_z, t_span, y0, z0)


if __name__ == '__main__':
    multiple_shooting_solver()
