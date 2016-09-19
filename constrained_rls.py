#    Copyright (c) 2014 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    wmil is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import fmin_slsqp
from numpy.linalg import norm
from numpy import dot, matrix
from scipy import sparse

class cRLS:
	def __init__(self, alpha=0.0):
		self.alpha = alpha
		self.coef_ = []

	def loss(self, coef_, X, y):
 		return  pow(y - (dot(X, coef_)), 2) + self.alpha*norm(coef_,2)

	def fprime(self, coef_, X, y):
		ders = np.zeros(len(coef_))
		for i, der in enumerate(ders):
			ders[i] = 2 * X[i] * (X[i]*coef_[i] - y) + (self.alpha * coef_[i])/norm(coef_,2)
		return  ders

 	def eq_con(self, coef_, *params):
 		return np.array([sum(coef_) - 1.])

	def fit(self, X, Y, P):
		X = X.view(np.ndarray)[0]
		self.coef_ = P.view(np.ndarray)[0]
		bounds = []
		for c in self.coef_:
			bounds.append( (0.00001,1.00001) )
		out = fmin_slsqp(self.loss, self.coef_, args=(X, Y),
						f_eqcons=self.eq_con, iter=50,
						bounds=bounds, iprint=0, full_output=0)
		self.coef_ = out
		return matrix(self.coef_)
