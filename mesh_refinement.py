import numpy as np


def mesh_refinement(N, size_y, size_z, tol, lte, t_span, y0, z0):
    y_temp = []
    z_temp = []
    t_temp = []
    lte_temp = []
    N_temp = 0
    # Deleting Nodes
    i = 0
    # number of deleted nodes
    k_d = 0
    # threshold for deleting node
    threshold_delete = 1e-2
    while i < N - 4:
        lte_i = lte[i]
        if lte_i <= threshold_delete:
            lte_i_plus_1 = lte[i + 1]
            lte_i_plus_2 = lte[i + 2]
            lte_i_plus_3 = lte[i + 3]
            lte_i_plus_4 = lte[i + 4]
            if lte_i_plus_1 <= threshold_delete and lte_i_plus_2 <= threshold_delete and \
                    lte_i_plus_3 <= threshold_delete and lte_i_plus_4 <= threshold_delete:
                # delete the second and forth node here
                # which is equal to only add node i, i + 2, and i + 4
                # add node i
                t_temp.append(t_span[i])
                y_temp.append(y0[i, 0: size_y])
                z_temp.append(z0[i, 0: size_z])
                lte_temp.append(lte_i)
                # add node i + 2
                t_temp.append(t_span[i + 2])
                y_temp.append(y0[i + 2, 0: size_y])
                z_temp.append(z0[i + 2, 0: size_z])
                lte_temp.append(lte_i_plus_2)
                # add node i + 4
                t_temp.append(t_span[i + 4])
                y_temp.append(y0[i + 4, 0: size_y])
                z_temp.append(z0[i + 4, 0: size_z])
                lte_temp.append(lte_i_plus_4)
                # record the number of nodes
                N_temp += 3
                # record the number of deleted nodes
                k_d += 2
                i += 5
            else:
                # add the current node only
                t_temp.append(t_span[i])
                y_temp.append(y0[i, :])
                z_temp.append(z0[i, :])
                lte_temp.append(lte_i)
                N_temp += 1
                i += 1
        else:
            # add the current node only
            t_temp.append(t_span[i])
            y_temp.append(y0[i, :])
            z_temp.append(z0[i, :])
            lte_temp.append(lte_i)
            N_temp += 1
            i += 1
    '''
        if the previous loop stop at the ith node which is bigger than (N - 4), those last
        few nodes left are added manually, if the last few nodes have already been processed,
        the index i should be equal to N, then nothing needs to be done
    '''
    if i < N:
        '''
            add the last few nodes starting from i to N - 1, which
            is a total of (N - i) nodes
        '''
        for j in range(N - i):
            # append the N - 4 + j node
            t_temp.append(t_span[i + j])
            y_temp.append(y0[i + j, :])
            z_temp.append(z0[i + j, :])
            lte_temp.append(lte[i + j])
            N_temp += 1
    # convert from list to numpy arrays for the convenience of indexing
    t_temp = np.array(t_temp)
    y_temp = np.array(y_temp)
    z_temp = np.array(z_temp)
    lte_temp = np.array(lte_temp)

    # lists to hold the outputs
    N_new = 0
    t_span_new = []
    y_new = []
    z_new = []
    # Adding Nodes
    i = 0
    # Record the number of the added nodes
    k_a = 0
    # threshold for adding node
    threshold_add = 1
    threshold_add_more = 10

    while i < N_temp - 1:
        lte_i = lte_temp[i]
        if lte_i > threshold_add:
            if lte_i > threshold_add_more:
                # add three uniformly spaced nodes
                # add the time point of new nodes
                delta_t = (t_temp[i + 1] - t_temp[i]) / 4
                t_i = t_temp[i]
                t_i_plus_1 = t_i + delta_t
                t_i_plus_2 = t_i + 2 * delta_t
                t_i_plus_3 = t_i + 3 * delta_t
                t_span_new.append(t_i)
                t_span_new.append(t_i_plus_1)
                t_span_new.append(t_i_plus_2)
                t_span_new.append(t_i_plus_3)
                # add the ys of the new nodes
                y_i = y_temp[i, :]
                y_i_next = y_temp[i + 1, :]
                delta_y = (y_i_next - y_i) / 4
                y_i_plus_1 = y_i + delta_y
                y_i_plus_2 = y_i + 2 * delta_y
                y_i_plus_3 = y_i + 3 * delta_y
                y_new.append(y_i)
                y_new.append(y_i_plus_1)
                y_new.append(y_i_plus_2)
                y_new.append(y_i_plus_3)
                # add the zs of the new nodes
                z_i = z_temp[i, :]
                z_i_next = z_temp[i + 1, :]
                delta_z = (z_i_next - z_i) / 4
                z_i_plus_1 = z_i + delta_z
                z_i_plus_2 = z_i + 2 * delta_z
                z_i_plus_3 = z_i + 3 * delta_z
                z_new.append(z_i)
                z_new.append(z_i_plus_1)
                z_new.append(z_i_plus_2)
                z_new.append(z_i_plus_3)
                # update the index
                # 1 original node + 3 newly added nodes
                N_new += 4
                k_a += 3
                i += 1
            else:
                # add one node to the middle
                # add the time point of the new node
                delta_t = (t_temp[i + 1] - t_temp[i]) / 2
                t_i = t_temp[i]
                t_i_plus_1 = t_i + delta_t
                t_span_new.append(t_i)
                t_span_new.append(t_i_plus_1)
                # add the y of the new node
                y_i = y_temp[i, :]
                y_i_next = y_temp[i + 1, :]
                delta_y = (y_i_next - y_i) / 2
                y_i_plus_1 = y_i + delta_y
                y_new.append(y_i)
                y_new.append(y_i_plus_1)
                # add the z of the new node
                z_i = z_temp[i, :]
                z_i_next = z_temp[i + 1, :]
                delta_z = (z_i_next - z_i) / 2
                z_i_plus_1 = z_i + delta_z
                z_new.append(z_i)
                z_new.append(z_i_plus_1)
                # update the index
                # 1 original node + 1 newly added node
                N_new += 2
                k_a += 1
                i += 1
        else:
            # add the current node only
            # add the time node of the current node
            t_i = t_temp[i]
            t_span_new.append(t_i)
            # add the y of the current node
            y_i = y_temp[i, :]
            y_new.append(y_i)
            # add the z of the current node
            z_i = z_temp[i, :]
            z_new.append(z_i)
            # update the index
            # 1 original node only
            N_new += 1
            i += 1

    # add the final node
    t_span_new.append(t_temp[-1])
    y_new.append(y_temp[-1, :])
    z_new.append(z_temp[-1, :])
    N_new += 1
    # convert from list to numpy arrays for the convenience of indexing
    t_span_new = np.array(t_span_new)
    y_new = np.array(y_new)
    z_new = np.array(z_new)
    # return the output
    return N_new, t_span_new, y_new, z_new
