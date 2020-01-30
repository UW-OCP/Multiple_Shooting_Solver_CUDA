import matplotlib.pyplot as plt


def plot_result(size_y, size_z, t_span, y, z):
	for i in range(size_y):
		plt.figure()
		plt.plot(t_span, y[:, i])
		plt.xlabel("Time")
		plt.ylabel("ODE variable{}".format(i + 1))
		plt.title("ODE variable{}".format(i + 1))
		plt.grid()
		plt.show()
	for i in range(size_z):
		plt.figure()
		plt.plot(t_span, z[:, i])
		plt.xlabel("Time")
		plt.ylabel("DAE variable{}".format(i + 1))
		plt.title("DAE variable{}".format(i + 1))
		plt.grid()
		plt.show()
