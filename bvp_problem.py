# Created by OCP.py
from math import *
from numba import cuda
from BVPDAEReadWriteData import bvpdae_read_data, bvpdae_write_data
import numpy as np


class BvpDae:

	def __init__(self):
		self.size_y = 4
		self.size_z = 3
		self.size_p = 2
		self.size_inequality = 0
		self.size_sv_inequality = 1
		self.output_file = 'ex5.data'
		self.tolerance = 1e-06
		self.maximum_nodes = 2000
		self.maximum_newton_iterations = 200
		self.maximum_mesh_refinements = 10
		self.N = 101
		self.t_initial = 0.0
		self.t_final = 5.0
		self.T0 = np.linspace(self.t_initial, self.t_final, self.N)
		self.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)
		self.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)
		self.P0 = np.ones(self.size_p, dtype=np.float64)
		self._solution_estimate(self.T0, self.Y0, self.Z0, self.P0)

	def _solution_estimate(self, _T, _Y, _Z, _P):
		N = _T.shape[0]
		for i in range(N):
			t = _T[i]
			_Y[i][0] = 1
			_Y[i][1] = 0
			_Z[i][0] = 0

		if _P.shape[0] != 0:
			for i in range(self.size_p):
				_P0 = np.ones(self.size_p, dtype=np.float64)


@cuda.jit(device=True)
def _abvp_f(_y, _z, _p, _f):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_xi1 = _z[1]
	_sigma1 = _z[2]
	_f[0] = x2
	_f[1] = u - x1 + x2*(1.0 - pow(x1, 2))
	_f[2] = -_lambda2*(-2*x1*x2 - 1) - 1.0*x1
	_f[3] = -_lambda1 - _lambda2*(1.0 - pow(x1, 2)) + _xi1 - 1.0*x2


@cuda.jit(device=True)
def _abvp_g(_y, _z, _p, _alpha, _g):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_xi1 = _z[1]
	_sigma1 = _z[2]
	_g[0] = _alpha*u + _lambda2 + 1.0*u
	_g[1] = -_alpha*_xi1 + _sigma1 - x2 - 0.25
	_g[2] = _sigma1 + _xi1 - sqrt(2*_alpha + pow(_sigma1, 2) + pow(_xi1, 2))


def _abvp_r(_y0, _y1, _p, _r):
	_kappa_i1 = _p[0]
	_kappa_i2 = _p[1]
	x1 = _y0[0]
	x2 = _y0[1]
	_lambda1 = _y0[2]
	_lambda2 = _y0[3]
	# initial conditions
	_r[0] = x1 - 1.0
	_r[1] = x2
	_r[2] = _kappa_i1 + _lambda1
	_r[3] = _kappa_i2 + _lambda2
	# final conditions
	x1 = _y1[0]
	x2 = _y1[1]
	_lambda1 = _y1[2]
	_lambda2 = _y1[3]
	_r[4] = _lambda1
	_r[5] = _lambda2


@cuda.jit(device=True)
def _abvp_Df(_y, _z, _p, _Df):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_xi1 = _z[1]
	_sigma1 = _z[2]
	_Df[0][1] = 1
	_Df[1][0] = -2*x1*x2 - 1
	_Df[1][1] = 1.0 - pow(x1, 2)
	_Df[1][4] = 1
	_Df[2][0] = 2*_lambda2*x2 - 1.0
	_Df[2][1] = 2*_lambda2*x1
	_Df[2][3] = 2*x1*x2 + 1
	_Df[3][0] = 2*_lambda2*x1
	_Df[3][1] = -1.0
	_Df[3][2] = -1
	_Df[3][3] = pow(x1, 2) - 1.0
	_Df[3][5] = 1


@cuda.jit(device=True)
def _abvp_Dg(_y, _z, _p, _alpha, _Dg):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_xi1 = _z[1]
	_sigma1 = _z[2]
	_Dg[0][3] = 1
	_Dg[0][4] = _alpha + 1.0
	_Dg[1][1] = -1
	_Dg[1][5] = -_alpha
	_Dg[1][6] = 1
	_Dg[2][5] = -_xi1/sqrt(2*_alpha + pow(_sigma1, 2) + pow(_xi1, 2)) + 1
	_Dg[2][6] = -_sigma1/sqrt(2*_alpha + pow(_sigma1, 2) + pow(_xi1, 2)) + 1


def _abvp_Dr(_y0, _y1, _p, _Dr):
	_kappa_i1 = _p[0]
	_kappa_i2 = _p[1]
	x1 = _y0[0]
	x2 = _y0[1]
	_lambda1 = _y0[2]
	_lambda2 = _y0[3]
	# initial conditions
	_Dr[0][0] = 1
	_Dr[1][1] = 1
	_Dr[2][2] = 1
	_Dr[2][8] = 1
	_Dr[3][3] = 1
	_Dr[3][9] = 1
	# final conditions
	x1 = _y1[0]
	x2 = _y1[1]
	_lambda1 = _y1[2]
	_lambda2 = _y1[3]
	_Dr[4][6] = 1
	_Dr[5][7] = 1


'''

# Shimizu, K. and Ito, S.,
# Constrained optimization in Hilbert space and a generalized dual quasi-newton
# algorithm for state-constrained optimal control problems,
# IEEE Trans. AC, Vol. 39, pp. 982--986, 1994.

StateVariables           = [x1, x2];
ControlVariables         = [u];

InitialTime              = 0.0;
FinalTime                = 5.0;

InitialConstraints       = [x1 - 1.0, x2];

CostFunctional           = 0.5*(x1*x1 + x2*x2 + u*u);

DifferentialEquations    = [ x2,
                            -x1 + (1.0-x1*x1)*x2 + u];

StateVariableInequalityConstraints = [-(x2+0.25)];

StateEstimate            = [1, 0];
ControlEstimate          = [0];

Nodes                    = 101;
Tolerance                = 1.0e-6;
OutputFile               = "ex5.data";

'''

