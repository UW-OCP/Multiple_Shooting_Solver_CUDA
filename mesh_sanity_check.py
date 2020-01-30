def mesh_sanit_check(N, mesh_it, max_mesh, max_nodes, min_nodes):
	if mesh_it > max_mesh:
		print("\tReach maximum times ({}) of mesh refinements!".format(max_mesh))
	if N > max_nodes:
		print("\tReach maximum nodes ({}) of mesh refinements!".format(max_nodes))
	if N < min_nodes:
		print("\tReach minimum nodes ({}) of mesh refinements!".format(min_nodes))
	return
