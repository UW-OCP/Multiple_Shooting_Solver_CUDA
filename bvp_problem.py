# Created by OCP.py
from math import *
from numba import cuda
from BVPDAEReadWriteData import bvpdae_read_data, bvpdae_write_data
import numpy as np


class BvpDae:

	def __init__(self):
		self.size_y = 4
		self.size_z = 5
		self.size_p = 4
		self.size_inequality = 2
		self.size_sv_inequality = 0
		self.output_file = 'ex8.data'
		self.tolerance = 1e-06
		self.maximum_nodes = 2000
		self.maximum_newton_iterations = 200
		self.maximum_mesh_refinements = 10
		error, t0, y0, z0, p0 = bvpdae_read_data("ex7b.data")
		if error != 0:
			print("Unable to read input file!")
			self.N = 100
			self.t_initial = 0.0
			self.t_final = 0.78
			self.T0 = np.linspace(self.t_initial, self.t_final, self.N)
			self.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)
			self.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)
			self.P0 = np.ones((self.size_p), dtype = np.float64)
		if error == 0:
			print("Read input file!")
			self.N = t0.shape[0]
			self.T0 = t0
			self.Y0 = None
			if self.size_y > 0:
				self.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)
			self.Z0 = None
			if self.size_z > 0:
				self.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)
			self.P0 = None
			if self.size_p > 0:
				self.P0 = np.ones(self.size_p, dtype=np.float64)
			self._pack_YZP(self.Y0, self.Z0, self.P0, y0, z0, p0)

	def _pack_YZP(self, _Y, _Z, _P, y0, z0, p0):
		if _Y is not None and y0 is not None:
			_n = self.N
			_m = y0.shape[1] if y0.shape[1] < _Y.shape[1] else _Y.shape[1]
			for i in range(_n):
				for j in range(_m):
					_Y[i][j] = y0[i][j]
		if _Z is not None and z0 is not None:
			_n = self.N
			# only read in enough dtat to fill the controls
			_m = z0.shape[1] if z0.shape[1] < 1 else 1
			for i in range(_n):
				for j in range(_m):
					_Z[i][j] = z0[i][j]
		if _P is not None and p0 is not None:
			_n = p0.shape[0] if p0.shape[0] < _P.shape[0] else _P.shape[0]
			for i in range(_n):
				_P[i] = p0[i]


@cuda.jit(device=True)
def _abvp_f(_y, _z, _p, _f):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_mu1 = _z[1]
	_mu2 = _z[2]
	_nu1 = _z[3]
	_nu2 = _z[4]
	_f[0] = -u*(x1 + 0.25) - 2.0*x1 + (x2 + 0.5)*exp(25.0*x1/(x1 + 2.0)) - 0.5
	_f[1] = -x2 - (x2 + 0.5)*exp(25.0*x1/(x1 + 2.0)) + 0.5
	_f[2] = -_lambda1*(-u + (x2 + 0.5)*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0)) - 2.0) - _lambda2*(-x2 - 0.5)*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0)) - 1.0*x1
	_f[3] = -_lambda1*exp(25.0*x1/(x1 + 2.0)) - _lambda2*(-exp(25.0*x1/(x1 + 2.0)) - 1) - 1.0*x2


@cuda.jit(device=True)
def _abvp_g(_y, _z, _p, _alpha, _g):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_mu1 = _z[1]
	_mu2 = _z[2]
	_nu1 = _z[3]
	_nu2 = _z[4]
	_g[0] = _alpha*u + _lambda1*(-x1 - 0.25) + _mu1 - _mu2
	_g[1] = -_alpha*_mu1 + _nu1 + u - 1.0
	_g[2] = -_alpha*_mu2 + _nu2 - u - 1.0
	_g[3] = _mu1 + _nu1 - sqrt(2*_alpha + pow(_mu1, 2) + pow(_nu1, 2))
	_g[4] = _mu2 + _nu2 - sqrt(2*_alpha + pow(_mu2, 2) + pow(_nu2, 2))


def _abvp_r(_y0, _y1, _p, _r):
	_kappa_i1 = _p[0]
	_kappa_i2 = _p[1]
	_kappa_f1 = _p[2]
	_kappa_f2 = _p[3]
	x1 = _y0[0]
	x2 = _y0[1]
	_lambda1 = _y0[2]
	_lambda2 = _y0[3]
	# initial conditions
	_r[0] = x1 - 0.050000000000000003
	_r[1] = x2
	_r[2] = _kappa_i1 + _lambda1
	_r[3] = _kappa_i2 + _lambda2
	# final conditions
	x1 = _y1[0]
	x2 = _y1[1]
	_lambda1 = _y1[2]
	_lambda2 = _y1[3]
	_r[4] = x1
	_r[5] = x2
	_r[6] = -_kappa_f1 + _lambda1
	_r[7] = -_kappa_f2 + _lambda2


@cuda.jit(device=True)
def _abvp_Df(_y, _z, _p, _Df):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_mu1 = _z[1]
	_mu2 = _z[2]
	_nu1 = _z[3]
	_nu2 = _z[4]
	_Df[0][0] = -u + (x2 + 0.5)*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0)) - 2.0
	_Df[0][1] = exp(25.0*x1/(x1 + 2.0))
	_Df[0][4] = -x1 - 0.25
	_Df[1][0] = (-x2 - 0.5)*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0))
	_Df[1][1] = -exp(25.0*x1/(x1 + 2.0)) - 1
	_Df[2][0] = -_lambda1*((x2 + 0.5)*(50.0*x1/pow(x1 + 2.0, 3) - 50.0/pow(x1 + 2.0, 2))*exp(25.0*x1/(x1 + 2.0)) + (x2 + 0.5)*pow(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0), 2)*exp(25.0*x1/(x1 + 2.0))) - _lambda2*(-x2 - 0.5)*(50.0*x1/pow(x1 + 2.0, 3) - 50.0/pow(x1 + 2.0, 2))*exp(25.0*x1/(x1 + 2.0)) - _lambda2*(-x2 - 0.5)*pow(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0), 2)*exp(25.0*x1/(x1 + 2.0)) - 1.0
	_Df[2][1] = -_lambda1*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0)) + _lambda2*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0))
	_Df[2][2] = u - (x2 + 0.5)*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0)) + 2.0
	_Df[2][3] = (x2 + 0.5)*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0))
	_Df[2][4] = _lambda1
	_Df[3][0] = -_lambda1*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0)) + _lambda2*(-25.0*x1/pow(x1 + 2.0, 2) + 25.0/(x1 + 2.0))*exp(25.0*x1/(x1 + 2.0))
	_Df[3][1] = -1.0
	_Df[3][2] = -exp(25.0*x1/(x1 + 2.0))
	_Df[3][3] = exp(25.0*x1/(x1 + 2.0)) + 1


@cuda.jit(device=True)
def _abvp_Dg(_y, _z, _p, _alpha, _Dg):
	x1 = _y[0]
	x2 = _y[1]
	_lambda1 = _y[2]
	_lambda2 = _y[3]
	u = _z[0]
	_mu1 = _z[1]
	_mu2 = _z[2]
	_nu1 = _z[3]
	_nu2 = _z[4]
	_Dg[0][0] = -_lambda1
	_Dg[0][2] = -x1 - 0.25
	_Dg[0][4] = _alpha
	_Dg[0][5] = 1
	_Dg[0][6] = -1
	_Dg[1][4] = 1
	_Dg[1][5] = -_alpha
	_Dg[1][7] = 1
	_Dg[2][4] = -1
	_Dg[2][6] = -_alpha
	_Dg[2][8] = 1
	_Dg[3][5] = -_mu1/sqrt(2*_alpha + pow(_mu1, 2) + pow(_nu1, 2)) + 1
	_Dg[3][7] = -_nu1/sqrt(2*_alpha + pow(_mu1, 2) + pow(_nu1, 2)) + 1
	_Dg[4][6] = -_mu2/sqrt(2*_alpha + pow(_mu2, 2) + pow(_nu2, 2)) + 1
	_Dg[4][8] = -_nu2/sqrt(2*_alpha + pow(_mu2, 2) + pow(_nu2, 2)) + 1


def _abvp_Dr(_y0, _y1, _p, _Dr):
	_kappa_i1 = _p[0]
	_kappa_i2 = _p[1]
	_kappa_f1 = _p[2]
	_kappa_f2 = _p[3]
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
	_Dr[4][4] = 1
	_Dr[5][5] = 1
	_Dr[6][6] = 1
	_Dr[6][10] = -1
	_Dr[7][7] = 1
	_Dr[7][11] = -1


'''

# Kirk, D. E., Optimal Control Theory: An Introduction,
# Prentice-Hall, 1970, pp. 338 and pp. 406.

Variables               = [a1, a2, a3, a4];
StateVariables          = [x1, x2];
ControlVariables        = [u];
InitialConstraints      = [x1 - 0.05, x2];
TerminalConstraints     = [x1, x2];

CostFunctional          = 0.5*(x1*x1 + x2*x2); # + 1.0e-6*u*u;

a1 = x1 + 0.25;
a2 = x2 + 0.5;
a3 = x1 + 2.0;
a4 = a2*exp(25.0*x1/a3);

DifferentialEquations   = [-2.0*a1 + a4 - a1*u,
                            0.5 - x2 - a4];

InequalityConstraints   = [u - 1.0, -1.0 - u];

InitialTime             = 0.0;
FinalTime               = 0.78;

Tolerance               = 1.0e-6;
InputFile              = "ex7b.data";
OutputFile              = "ex8.data";

'''

