# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:22:24 2018

Functions to transform standard SOCP form into the form used by cvxopt
Standard form is
                min   f^T x
                s.t.  ||A_i x + b_i ||_2 \leq c_i^T x + d_i   i=1,...m
                      Fx = g

Example: solve convex QCQP for mean-variance optimization
                max   r^T x
                s.t.  x^T SIGMA x \leq var_target
                      e^T x = 1
                      x \geq 0

@author: aklamun
"""

import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

###############################################################################
def convert_constraint(A,b,c,d):
    '''Convert SOCP constraint of form ||Ax+b||_2 \leq c^T x + d to cvxopt form Gx + s = h'''
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    
    G = np.zeros((np.shape(A)[0]+1, np.shape(A)[1]))
    G[0,:] = -np.transpose(c)
    G[1:,:] = -A
    h = np.zeros((np.shape(A)[0]+1,1))
    h[0,0] = d[0,0]
    h[1:,0] = b[:,0]
    return matrix(G,tc='d'), matrix(h,tc='d')

def convert_SOCP(f,A,b,c,d,F,g):
    '''convert SOCP of form     min   f^T x
                                s.t.  ||A_i x + b_i ||_2 \leq c_i^T x + d_i   i=1,...m
                                      Fx = g
    to the form needed for cvxopt solver
    A is list of A_i, b is list of b_i, c is list of c_i, d is list of d_i'''
    assert len(A) == len(b) == len(c) == len(d)
    G = []
    h = []
    for i in range(len(A)):
        Gi, hi = convert_constraint(A[i],b[i],c[i],d[i])
        G.append(Gi)
        h.append(hi)
    return matrix(f,tc='d'), G, h, matrix(F,tc='d'), matrix(g,tc='d')

def solve_SOCP(f,A,b,c,d,F,g):
    '''solve SOCP of form     min   f^T x
                                s.t.  ||A_i x + b_i ||_2 \leq c_i^T x + d_i   i=1,...m
                                      Fx = g
    using cvxopt'''
    c, G, h, A, b = convert_SOCP(f,A,b,c,d,F,g)
    sol = solvers.socp(c=c, Gq=G, hq=h, A=A, b=b)
    return sol

###############################################################################
def mean_var_opt(rets, cov, var_target):
    '''maximize expected return subject to max target variance, no shorting
    this is a convex QCQP, which can be expressed as a SOCP and solved in cvxopt
    rets = expected returns vector
    cov = covariance matrix'''
    try:
        chol = np.transpose(np.linalg.cholesky(cov))
    except:
        raise Exception('Covariance matrix is not positive definite')
    
    #put into SOCP standard form
    n = len(rets)
    f = -rets
    F = np.ones((1,n))
    g = 1
    A = [chol] + [np.zeros((1,n)) for i in range(n)]
    b = [np.zeros((n,1))] + [[[0]] for i in range(n)]
    c = [np.zeros((n,1))]
    for i in range(n):
        ci = np.zeros((n,1))
        ci[i,0] = 1
        c.append(ci)
    d = [[[np.sqrt(var_target)]]] + [[[0]] for i in range(n)]
    
    sol = solve_SOCP(f,A,b,c,d,F,g)
    return np.array(sol['x'])
    
    
