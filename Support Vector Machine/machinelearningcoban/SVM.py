from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)

means =[[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

# create 2 classes of data

# Parameters
# mean: 1-D array_like, of length N
#      Mean of the N-dimensional distribution.

# cov: 2-D array_like, of shape (N, N)
#      Covariance matrix of the distribution. It must be symmetric and positive-semidefinite for proper sampling.

# N: size of generated sample

X0 = np.random.multivariate_normal(means[0], cov, N)    # class 1  Draw random samples from a multivariate normal distribution.
X1 = np.random.multivariate_normal(means[1], cov, N)    # class -1  Draw random samples from a multivariate normal distribution.
X = np.concatenate((X0.T, X1.T), axis = 1)  # all training data
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)     # labels    # create a Nth dimensional 1's vector and a Nth dimensional -1's vector and concatenate them by the rows
#print('X0 = ', X0)
#print('\nX1 = ', X1)
#print('\nX = ', X)
#print('\ny = ', y)

# CVXOPT là một thư viện miễn phí trên Python giúp giải rất nhiều 
# các bài toán trong cuốn sách Convex Optimization
from cvxopt import matrix, solvers

# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V))

p = matrix(-np.ones((2 * N, 1)))    # all-one vector

# build A, b, G, h
G = matrix(-np.eye(2 * N))  # for all lambda_n >= 0
h = matrix(np.zeros((2 * N, 1)))
A = matrix(y)   # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1)))

solvers.options['show_progress'] = False 
sol = solvers.qp(K, p, G, h, A, b)  # solve quadratic programing problem

l = np.array(sol['x'])  # get the lambda

print('lambda = ', l.T)

epsilon = 1e-6

S = np.where(l > epsilon)[0]    # find lambdas which are greater than epsilon

VS = V[:, S]    # VS is the support vectors, columns contains x_m for m in S
XS = X[:, S]    # XS is the support vectors, columns contains x_m for m in S
yS = y[:, S]    # yS is the support vectors, columns contains y_m for m in S
lS = l[S]    # lS is the support vectors, value is the Sth's values in l

# calculate w and b
w = VS.dot(lS)  # find w by VS * lS    (formula (16))
b = np.mean(yS.T - w.T.dot(XS))   # find b by yS - w^T * XS (formula (17))

print('w = ', w.T)
print('b = ', b)