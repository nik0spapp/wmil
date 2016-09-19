#    Copyright (c) 2014 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    wmil is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.


import sys
import numpy as np
from sklearn import linear_model
from constrained_rls import cRLS
from sklearn.base import BaseEstimator
from scipy.sparse import lil_matrix, csr_matrix, vstack
from sklearn.metrics import mean_absolute_error

class APWeights(BaseEstimator):

	def __init__(self, iterations, e1=1.0, e2=1.0, e3=1.0):
		self.e1 = e1					# regularization term for f1
		self.e2 = e2  					# regularization term for f2
		self.e3 = e3					# regularization tern for f3
		self.iterations = iterations  	# number of iterations
		self.f1 = self.f2 = self.f3 = None

	def predict(self, X):
		H = np.matrix(np.zeros((len(X), X[0].get_shape()[1])))
		for i, Xi in enumerate(X):
			Wi = np.matrix(self.f3.predict(Xi)).view(np.ndarray)[0]
			H[i] =  np.dot(Wi, Xi.todense())[0]
		pred = self.f2.predict(H)
		return pred

	def fit(self, X, Y):
<<<<<<< Updated upstream
		M = X[0].get_shape()[1]      			# number of features
		N = len(X)				     	# number of bags 
		F = np.random.ranf((1,M))    			# regression hyperplane
 		H = np.matrix(np.zeros((N,M)))			# bag representations
		self.P = []					# instance weights
=======
		M = X[0].get_shape()[1]      	# number of features
		N = len(X)				     	# number of bags
		F = np.random.ranf((1,M))    	# regression hyperplane
 		H = np.matrix(np.zeros((N,M)))	# bag representations
		self.P = []						# instance weights
>>>>>>> Stashed changes
		self.X_w = []					# flatten instances
		self.Y_w = []					# flatten instance weights

		converged = False
		prev_error = sys.maxsize
		it = 0
		#print "-"*100
		#print "e1: %f" % self.e1
		#print "e2: %f" % self.e2
		#print "e3: %f" % self.e3
		#print "M: %d" % M
		#print "N: %d" % N

		print
		print "[+] Training..."
		print "--/start"
		while(not converged and it < self.iterations):
			for i, Xi in enumerate(X):
				if it == 0:
					if self.X_w == []:
						self.X_w = Xi
					else:
						self.X_w = vstack([self.X_w, Xi])
					self.P.append(np.ones((1,X[i].get_shape()[0])))
					self.Y_w.append([])
				Xi = Xi.tocsr()
				if self.f2:
					HC = np.matrix(self.f2.predict(Xi)).T
				else:
					HC = Xi.dot(F.T).T
				self.f1 = cRLS(alpha=self.e1)
				self.P[i] = self.f1.fit(HC,Y[i],self.P[i])
				self.Y_w[i] = self.f1.coef_
				cur_p = csr_matrix(self.f1.coef_)
				H[i] = cur_p.dot(Xi).todense()

			self.f2 = linear_model.Ridge(alpha=self.e2)
			self.f2.fit(H,Y)
			cur_error = mean_absolute_error(self.f2.predict(H),Y)
			print "iteration %d -> (MAE: %f) " % (it, cur_error)
			if prev_error - cur_error < 0.000001:
				converged = True
			prev_error = cur_error
			it += 1
		self.f3 = linear_model.Ridge(alpha=self.e3)
		self.f3.fit(self.X_w,np.hstack(self.Y_w))
		print "--/end"
