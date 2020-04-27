# Created by OCP.py
from math import *
from numba import cuda
from BVPDAEReadWriteData import bvpdae_read_data, bvpdae_write_data
import numpy as np


class BvpDae:

	def __init__(self):
		m = 10.0
		d = 5.0
		l = 5.0
		I = 12.0
		pi = 3.14159265358979
		rho = 10.0
		self.size_y = 13
		self.size_z = 12
		self.size_p = 13
		self.size_inequality = 4
		self.size_sv_inequality = 0
		self.output_file = 'ex22.data'
		self.tolerance = 1e-06
		self.maximum_nodes = 2000
		self.maximum_newton_iterations = 200
		self.maximum_mesh_refinements = 20
		error, t0, y0, z0, p0 = bvpdae_read_data("ex21.data")
		if error != 0:
			print("Unable to read input file!")
			self.N = 100
			self.t_initial = 0.0
			self.t_final = 1.0
			self.T0 = np.linspace(self.t_initial, self.t_final, self.N)
			self.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)
			self.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)
			self.P0 = np.ones((self.size_p), dtype = np.float64)
			self._solution_estimate(self.T0, self.Y0, self.Z0, self.P0)
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
			self._solution_estimate(self.T0, self.Y0, self.Z0, self.P0)

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
			_m = z0.shape[1] if z0.shape[1] < 4 else 4
			for i in range(_n):
				for j in range(_m):
					_Z[i][j] = z0[i][j]
		if _P is not None and p0 is not None:
			_n = p0.shape[0] if p0.shape[0] < _P.shape[0] else _P.shape[0]
			for i in range(_n):
				_P[i] = p0[i]

	def _solution_estimate(self, _T, _Y, _Z, _P):
		m = 10.0
		d = 5.0
		l = 5.0
		I = 12.0
		pi = 3.14159265358979
		rho = 10.0
		N = _T.shape[0]
		for i in range(N):
			t = _T[i]

		if _P.shape[0] != 0:
			for i in range(self.size_p):
				_P[0] = 0


@cuda.jit(device=True)
def _abvp_f(_y, _z, _p, _f):
	m = 10.0
	d = 5.0
	l = 5.0
	I = 12.0
	pi = 3.14159265358979
	rho = 10.0
	x1 = _y[0]
	x2 = _y[1]
	x3 = _y[2]
	x4 = _y[3]
	x5 = _y[4]
	x6 = _y[5]
	_lambda1 = _y[6]
	_lambda2 = _y[7]
	_lambda3 = _y[8]
	_lambda4 = _y[9]
	_lambda5 = _y[10]
	_lambda6 = _y[11]
	_gamma1 = _y[12]
	u1 = _z[0]
	u2 = _z[1]
	u3 = _z[2]
	u4 = _z[3]
	_mu1 = _z[4]
	_mu2 = _z[5]
	_mu3 = _z[6]
	_mu4 = _z[7]
	_nu1 = _z[8]
	_nu2 = _z[9]
	_nu3 = _z[10]
	_nu4 = _z[11]
	p = _p[0]
	_f[0] = x2*(p + 10)
	_f[1] = (p + 10)*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_f[2] = x4*(p + 10)
	_f[3] = (p + 10)*((u1 + u3)*sin(x5) + (u2 + u4)*cos(x5))/m
	_f[4] = x6*(p + 10)
	_f[5] = (p + 10)*(d*(u1 + u3) - l*(u2 + u4))/I
	_f[6] = 0
	_f[7] = -_lambda1*(p + 10)
	_f[8] = 0
	_f[9] = -_lambda3*(p + 10)
	_f[10] = -_lambda2*(p + 10)*(-(u1 + u3)*sin(x5) + (-u2 - u4)*cos(x5))/m - _lambda4*(p + 10)*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_f[11] = -_lambda5*(p + 10)
	_f[12] = _lambda1*x2 + _lambda2*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m + _lambda3*x4 + _lambda4*((u1 + u3)*sin(x5) + (u2 + u4)*cos(x5))/m + _lambda5*x6 + rho + pow(u1, 2) + pow(u2, 2) + pow(u3, 2) + pow(u4, 2) + _lambda6*(d*(u1 + u3) - l*(u2 + u4))/I


@cuda.jit(device=True)
def _abvp_g(_y, _z, _p, _alpha, _g):
	m = 10.0
	d = 5.0
	l = 5.0
	I = 12.0
	pi = 3.14159265358979
	rho = 10.0
	x1 = _y[0]
	x2 = _y[1]
	x3 = _y[2]
	x4 = _y[3]
	x5 = _y[4]
	x6 = _y[5]
	_lambda1 = _y[6]
	_lambda2 = _y[7]
	_lambda3 = _y[8]
	_lambda4 = _y[9]
	_lambda5 = _y[10]
	_lambda6 = _y[11]
	_gamma1 = _y[12]
	u1 = _z[0]
	u2 = _z[1]
	u3 = _z[2]
	u4 = _z[3]
	_mu1 = _z[4]
	_mu2 = _z[5]
	_mu3 = _z[6]
	_mu4 = _z[7]
	_nu1 = _z[8]
	_nu2 = _z[9]
	_nu3 = _z[10]
	_nu4 = _z[11]
	p = _p[0]
	_g[0] = _lambda2*(p + 10)*cos(x5)/m + _lambda4*(p + 10)*sin(x5)/m - _mu1*(5.0 - u1) - _mu1*(-u1 - 5.0) + 2*u1*(p + 10) + _lambda6*d*(p + 10)/I
	_g[1] = -_lambda2*(p + 10)*sin(x5)/m + _lambda4*(p + 10)*cos(x5)/m - _mu2*(5.0 - u2) - _mu2*(-u2 - 5.0) + 2*u2*(p + 10) - _lambda6*l*(p + 10)/I
	_g[2] = _lambda2*(p + 10)*cos(x5)/m + _lambda4*(p + 10)*sin(x5)/m - _mu3*(5.0 - u3) - _mu3*(-u3 - 5.0) + 2*u3*(p + 10) + _lambda6*d*(p + 10)/I
	_g[3] = -_lambda2*(p + 10)*sin(x5)/m + _lambda4*(p + 10)*cos(x5)/m - _mu4*(5.0 - u4) - _mu4*(-u4 - 5.0) + 2*u4*(p + 10) - _lambda6*l*(p + 10)/I
	_g[4] = -_alpha*_mu1 + _nu1 + (5.0 - u1)*(-u1 - 5.0)
	_g[5] = -_alpha*_mu2 + _nu2 + (5.0 - u2)*(-u2 - 5.0)
	_g[6] = -_alpha*_mu3 + _nu3 + (5.0 - u3)*(-u3 - 5.0)
	_g[7] = -_alpha*_mu4 + _nu4 + (5.0 - u4)*(-u4 - 5.0)
	_g[8] = _mu1 + _nu1 - sqrt(2*_alpha + pow(_mu1, 2) + pow(_nu1, 2))
	_g[9] = _mu2 + _nu2 - sqrt(2*_alpha + pow(_mu2, 2) + pow(_nu2, 2))
	_g[10] = _mu3 + _nu3 - sqrt(2*_alpha + pow(_mu3, 2) + pow(_nu3, 2))
	_g[11] = _mu4 + _nu4 - sqrt(2*_alpha + pow(_mu4, 2) + pow(_nu4, 2))


def _abvp_r(_y0, _y1, _p, _r):
	m = 10.0
	d = 5.0
	l = 5.0
	I = 12.0
	pi = 3.14159265358979
	rho = 10.0
	p = _p[0]
	_kappa_i1 = _p[1]
	_kappa_i2 = _p[2]
	_kappa_i3 = _p[3]
	_kappa_i4 = _p[4]
	_kappa_i5 = _p[5]
	_kappa_i6 = _p[6]
	_kappa_f1 = _p[7]
	_kappa_f2 = _p[8]
	_kappa_f3 = _p[9]
	_kappa_f4 = _p[10]
	_kappa_f5 = _p[11]
	_kappa_f6 = _p[12]
	x1 = _y0[0]
	x2 = _y0[1]
	x3 = _y0[2]
	x4 = _y0[3]
	x5 = _y0[4]
	x6 = _y0[5]
	_lambda1 = _y0[6]
	_lambda2 = _y0[7]
	_lambda3 = _y0[8]
	_lambda4 = _y0[9]
	_lambda5 = _y0[10]
	_lambda6 = _y0[11]
	_gamma1 = _y0[12]
	# initial conditions
	_r[0] = x1
	_r[1] = x2
	_r[2] = x3
	_r[3] = x4
	_r[4] = x5
	_r[5] = x6
	_r[6] = _kappa_i1 + _lambda1
	_r[7] = _kappa_i2 + _lambda2
	_r[8] = _kappa_i3 + _lambda3
	_r[9] = _kappa_i4 + _lambda4
	_r[10] = _kappa_i5 + _lambda5
	_r[11] = _kappa_i6 + _lambda6
	_r[12] = _gamma1
	# final conditions
	x1 = _y1[0]
	x2 = _y1[1]
	x3 = _y1[2]
	x4 = _y1[3]
	x5 = _y1[4]
	x6 = _y1[5]
	_lambda1 = _y1[6]
	_lambda2 = _y1[7]
	_lambda3 = _y1[8]
	_lambda4 = _y1[9]
	_lambda5 = _y1[10]
	_lambda6 = _y1[11]
	_gamma1 = _y1[12]
	_r[13] = x1 - 4.0
	_r[14] = x2
	_r[15] = x3 - 4.0
	_r[16] = x4
	_r[17] = -0.25*pi + x5
	_r[18] = x6
	_r[19] = -_kappa_f1 + _lambda1
	_r[20] = -_kappa_f2 + _lambda2
	_r[21] = -_kappa_f3 + _lambda3
	_r[22] = -_kappa_f4 + _lambda4
	_r[23] = -_kappa_f5 + _lambda5
	_r[24] = -_kappa_f6 + _lambda6
	_r[25] = _gamma1


@cuda.jit(device=True)
def _abvp_Df(_y, _z, _p, _Df):
	m = 10.0
	d = 5.0
	l = 5.0
	I = 12.0
	pi = 3.14159265358979
	rho = 10.0
	x1 = _y[0]
	x2 = _y[1]
	x3 = _y[2]
	x4 = _y[3]
	x5 = _y[4]
	x6 = _y[5]
	_lambda1 = _y[6]
	_lambda2 = _y[7]
	_lambda3 = _y[8]
	_lambda4 = _y[9]
	_lambda5 = _y[10]
	_lambda6 = _y[11]
	_gamma1 = _y[12]
	u1 = _z[0]
	u2 = _z[1]
	u3 = _z[2]
	u4 = _z[3]
	_mu1 = _z[4]
	_mu2 = _z[5]
	_mu3 = _z[6]
	_mu4 = _z[7]
	_nu1 = _z[8]
	_nu2 = _z[9]
	_nu3 = _z[10]
	_nu4 = _z[11]
	p = _p[0]
	_Df[0][1] = p + 10
	_Df[0][25] = x2
	_Df[1][4] = (p + 10)*(-(u1 + u3)*sin(x5) + (-u2 - u4)*cos(x5))/m
	_Df[1][13] = (p + 10)*cos(x5)/m
	_Df[1][14] = -(p + 10)*sin(x5)/m
	_Df[1][15] = (p + 10)*cos(x5)/m
	_Df[1][16] = -(p + 10)*sin(x5)/m
	_Df[1][25] = ((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_Df[2][3] = p + 10
	_Df[2][25] = x4
	_Df[3][4] = (p + 10)*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_Df[3][13] = (p + 10)*sin(x5)/m
	_Df[3][14] = (p + 10)*cos(x5)/m
	_Df[3][15] = (p + 10)*sin(x5)/m
	_Df[3][16] = (p + 10)*cos(x5)/m
	_Df[3][25] = ((u1 + u3)*sin(x5) + (u2 + u4)*cos(x5))/m
	_Df[4][5] = p + 10
	_Df[4][25] = x6
	_Df[5][13] = d*(p + 10)/I
	_Df[5][14] = -l*(p + 10)/I
	_Df[5][15] = d*(p + 10)/I
	_Df[5][16] = -l*(p + 10)/I
	_Df[5][25] = (d*(u1 + u3) - l*(u2 + u4))/I
	_Df[7][6] = -p - 10
	_Df[7][25] = -_lambda1
	_Df[9][8] = -p - 10
	_Df[9][25] = -_lambda3
	_Df[10][4] = -_lambda2*(p + 10)*((-u1 - u3)*cos(x5) - (-u2 - u4)*sin(x5))/m - _lambda4*(p + 10)*(-(u1 + u3)*sin(x5) + (-u2 - u4)*cos(x5))/m
	_Df[10][7] = -(p + 10)*(-(u1 + u3)*sin(x5) + (-u2 - u4)*cos(x5))/m
	_Df[10][9] = -(p + 10)*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_Df[10][13] = _lambda2*(p + 10)*sin(x5)/m - _lambda4*(p + 10)*cos(x5)/m
	_Df[10][14] = _lambda2*(p + 10)*cos(x5)/m + _lambda4*(p + 10)*sin(x5)/m
	_Df[10][15] = _lambda2*(p + 10)*sin(x5)/m - _lambda4*(p + 10)*cos(x5)/m
	_Df[10][16] = _lambda2*(p + 10)*cos(x5)/m + _lambda4*(p + 10)*sin(x5)/m
	_Df[10][25] = -_lambda2*(-(u1 + u3)*sin(x5) + (-u2 - u4)*cos(x5))/m - _lambda4*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_Df[11][10] = -p - 10
	_Df[11][25] = -_lambda5
	_Df[12][1] = _lambda1
	_Df[12][3] = _lambda3
	_Df[12][4] = _lambda2*(-(u1 + u3)*sin(x5) + (-u2 - u4)*cos(x5))/m + _lambda4*((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_Df[12][5] = _lambda5
	_Df[12][6] = x2
	_Df[12][7] = ((u1 + u3)*cos(x5) - (u2 + u4)*sin(x5))/m
	_Df[12][8] = x4
	_Df[12][9] = ((u1 + u3)*sin(x5) + (u2 + u4)*cos(x5))/m
	_Df[12][10] = x6
	_Df[12][11] = (d*(u1 + u3) - l*(u2 + u4))/I
	_Df[12][13] = _lambda2*cos(x5)/m + _lambda4*sin(x5)/m + 2*u1 + _lambda6*d/I
	_Df[12][14] = -_lambda2*sin(x5)/m + _lambda4*cos(x5)/m + 2*u2 - _lambda6*l/I
	_Df[12][15] = _lambda2*cos(x5)/m + _lambda4*sin(x5)/m + 2*u3 + _lambda6*d/I
	_Df[12][16] = -_lambda2*sin(x5)/m + _lambda4*cos(x5)/m + 2*u4 - _lambda6*l/I


@cuda.jit(device=True)
def _abvp_Dg(_y, _z, _p, _alpha, _Dg):
	m = 10.0
	d = 5.0
	l = 5.0
	I = 12.0
	pi = 3.14159265358979
	rho = 10.0
	x1 = _y[0]
	x2 = _y[1]
	x3 = _y[2]
	x4 = _y[3]
	x5 = _y[4]
	x6 = _y[5]
	_lambda1 = _y[6]
	_lambda2 = _y[7]
	_lambda3 = _y[8]
	_lambda4 = _y[9]
	_lambda5 = _y[10]
	_lambda6 = _y[11]
	_gamma1 = _y[12]
	u1 = _z[0]
	u2 = _z[1]
	u3 = _z[2]
	u4 = _z[3]
	_mu1 = _z[4]
	_mu2 = _z[5]
	_mu3 = _z[6]
	_mu4 = _z[7]
	_nu1 = _z[8]
	_nu2 = _z[9]
	_nu3 = _z[10]
	_nu4 = _z[11]
	p = _p[0]
	_Dg[0][4] = -_lambda2*(p + 10)*sin(x5)/m + _lambda4*(p + 10)*cos(x5)/m
	_Dg[0][7] = (p + 10)*cos(x5)/m
	_Dg[0][9] = (p + 10)*sin(x5)/m
	_Dg[0][11] = d*(p + 10)/I
	_Dg[0][13] = 2*_mu1 + 2*p + 20
	_Dg[0][17] = 2*u1
	_Dg[0][25] = _lambda2*cos(x5)/m + _lambda4*sin(x5)/m + 2*u1 + _lambda6*d/I
	_Dg[1][4] = -_lambda2*(p + 10)*cos(x5)/m - _lambda4*(p + 10)*sin(x5)/m
	_Dg[1][7] = -(p + 10)*sin(x5)/m
	_Dg[1][9] = (p + 10)*cos(x5)/m
	_Dg[1][11] = -l*(p + 10)/I
	_Dg[1][14] = 2*_mu2 + 2*p + 20
	_Dg[1][18] = 2*u2
	_Dg[1][25] = -_lambda2*sin(x5)/m + _lambda4*cos(x5)/m + 2*u2 - _lambda6*l/I
	_Dg[2][4] = -_lambda2*(p + 10)*sin(x5)/m + _lambda4*(p + 10)*cos(x5)/m
	_Dg[2][7] = (p + 10)*cos(x5)/m
	_Dg[2][9] = (p + 10)*sin(x5)/m
	_Dg[2][11] = d*(p + 10)/I
	_Dg[2][15] = 2*_mu3 + 2*p + 20
	_Dg[2][19] = 2*u3
	_Dg[2][25] = _lambda2*cos(x5)/m + _lambda4*sin(x5)/m + 2*u3 + _lambda6*d/I
	_Dg[3][4] = -_lambda2*(p + 10)*cos(x5)/m - _lambda4*(p + 10)*sin(x5)/m
	_Dg[3][7] = -(p + 10)*sin(x5)/m
	_Dg[3][9] = (p + 10)*cos(x5)/m
	_Dg[3][11] = -l*(p + 10)/I
	_Dg[3][16] = 2*_mu4 + 2*p + 20
	_Dg[3][20] = 2*u4
	_Dg[3][25] = -_lambda2*sin(x5)/m + _lambda4*cos(x5)/m + 2*u4 - _lambda6*l/I
	_Dg[4][13] = 2*u1
	_Dg[4][17] = -_alpha
	_Dg[4][21] = 1
	_Dg[5][14] = 2*u2
	_Dg[5][18] = -_alpha
	_Dg[5][22] = 1
	_Dg[6][15] = 2*u3
	_Dg[6][19] = -_alpha
	_Dg[6][23] = 1
	_Dg[7][16] = 2*u4
	_Dg[7][20] = -_alpha
	_Dg[7][24] = 1
	_Dg[8][17] = -_mu1/sqrt(2*_alpha + pow(_mu1, 2) + pow(_nu1, 2)) + 1
	_Dg[8][21] = -_nu1/sqrt(2*_alpha + pow(_mu1, 2) + pow(_nu1, 2)) + 1
	_Dg[9][18] = -_mu2/sqrt(2*_alpha + pow(_mu2, 2) + pow(_nu2, 2)) + 1
	_Dg[9][22] = -_nu2/sqrt(2*_alpha + pow(_mu2, 2) + pow(_nu2, 2)) + 1
	_Dg[10][19] = -_mu3/sqrt(2*_alpha + pow(_mu3, 2) + pow(_nu3, 2)) + 1
	_Dg[10][23] = -_nu3/sqrt(2*_alpha + pow(_mu3, 2) + pow(_nu3, 2)) + 1
	_Dg[11][20] = -_mu4/sqrt(2*_alpha + pow(_mu4, 2) + pow(_nu4, 2)) + 1
	_Dg[11][24] = -_nu4/sqrt(2*_alpha + pow(_mu4, 2) + pow(_nu4, 2)) + 1


def _abvp_Dr(_y0, _y1, _p, _Dr):
	m = 10.0
	d = 5.0
	l = 5.0
	I = 12.0
	pi = 3.14159265358979
	rho = 10.0
	p = _p[0]
	_kappa_i1 = _p[1]
	_kappa_i2 = _p[2]
	_kappa_i3 = _p[3]
	_kappa_i4 = _p[4]
	_kappa_i5 = _p[5]
	_kappa_i6 = _p[6]
	_kappa_f1 = _p[7]
	_kappa_f2 = _p[8]
	_kappa_f3 = _p[9]
	_kappa_f4 = _p[10]
	_kappa_f5 = _p[11]
	_kappa_f6 = _p[12]
	x1 = _y0[0]
	x2 = _y0[1]
	x3 = _y0[2]
	x4 = _y0[3]
	x5 = _y0[4]
	x6 = _y0[5]
	_lambda1 = _y0[6]
	_lambda2 = _y0[7]
	_lambda3 = _y0[8]
	_lambda4 = _y0[9]
	_lambda5 = _y0[10]
	_lambda6 = _y0[11]
	_gamma1 = _y0[12]
	# initial conditions
	_Dr[0][0] = 1
	_Dr[1][1] = 1
	_Dr[2][2] = 1
	_Dr[3][3] = 1
	_Dr[4][4] = 1
	_Dr[5][5] = 1
	_Dr[6][6] = 1
	_Dr[6][27] = 1
	_Dr[7][7] = 1
	_Dr[7][28] = 1
	_Dr[8][8] = 1
	_Dr[8][29] = 1
	_Dr[9][9] = 1
	_Dr[9][30] = 1
	_Dr[10][10] = 1
	_Dr[10][31] = 1
	_Dr[11][11] = 1
	_Dr[11][32] = 1
	_Dr[12][12] = 1
	# final conditions
	x1 = _y1[0]
	x2 = _y1[1]
	x3 = _y1[2]
	x4 = _y1[3]
	x5 = _y1[4]
	x6 = _y1[5]
	_lambda1 = _y1[6]
	_lambda2 = _y1[7]
	_lambda3 = _y1[8]
	_lambda4 = _y1[9]
	_lambda5 = _y1[10]
	_lambda6 = _y1[11]
	_gamma1 = _y1[12]
	_Dr[13][13] = 1
	_Dr[14][14] = 1
	_Dr[15][15] = 1
	_Dr[16][16] = 1
	_Dr[17][17] = 1
	_Dr[18][18] = 1
	_Dr[19][19] = 1
	_Dr[19][33] = -1
	_Dr[20][20] = 1
	_Dr[20][34] = -1
	_Dr[21][21] = 1
	_Dr[21][35] = -1
	_Dr[22][22] = 1
	_Dr[22][36] = -1
	_Dr[23][23] = 1
	_Dr[23][37] = -1
	_Dr[24][24] = 1
	_Dr[24][38] = -1
	_Dr[25][25] = 1


'''

Variables               = [c5, s5, w];
Constants               = [m = 10.0, d = 5.0, l = 5.0, I = 12.0,
                           pi = 3.14159265358979, rho = 10.0];
InitialTime             = 0.0;
FinalTime               = 1.0;
StateVariables          = [x1, x2, x3, x4, x5, x6];
ControlVariables        = [u1, u2, u3, u4];
ParameterVariables      = [p];
InitialConstraints      = [x1, x2, x3, x4, x5, x6];
TerminalConstraints     = [x1 - 4.0, x2, x3 - 4.0, x4, x5 - pi/4.0, x6];

c5 = cos(x5);
s5 = sin(x5);
w = p+10;
DifferentialEquations   = [w*x2,
        w*((u1+u3)*c5 - (u2+u4)*s5)/m,
        w*x4,
        w*((u1+u3)*s5 + (u2+u4)*c5)/m,
        w*x6,
        w*((u1+u3)*d - (u2+u4)*l)/I];

#TerminalPenalty         = w;
CostFunctional          = w*(rho + u1*u1 + u2*u2 + u3*u3 + u4*u4);

InequalityConstraints   = [ -(u1 - 5.0)*(-5.0 - u1),
                            -(u2 - 5.0)*(-5.0 - u2),
                            -(u3 - 5.0)*(-5.0 - u3),
                            -(u4 - 5.0)*(-5.0 - u4)];

Tolerance               = 1.0e-6;
MaximumMeshRefinements  = 20;
MaximumNodes            = 2000;

InputFile               = "ex21.data";
OutputFile              = "ex22.data";
ParameterEstimate       = [0];

'''

