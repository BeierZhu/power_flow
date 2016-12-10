import time
import numpy as np
import numpy.linalg as LA
import math
import delta_P_Q
import util

# BB--------------------------------------------------------------------------------------------------------------------
def BB(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    Y_p = Y.copy()
    B_p = np.zeros((n_P,n_P))
    B_pp = np.zeros((n_PQ,n_PQ))
    #--------------------------------------------------
    for i in xrange(case_number):
        Y_p[i][i] = complex(0,0)
        for j in xrange(case_number):
            if i != j:
                Y_p[i][i] -= Y_p[i][j]

    B = np.imag(Y_p)
    for i in xrange(n_P):
        for j in xrange(0, n_P):
            B_p[i][j] = B[index_P[i]][index_P[j]]
    #--------------------------------------------------
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_pp[i][j] = B[index_PQ[i]][index_PQ[j]]

    return B_p, B_pp

# A.M Van Amerongen-----------------------------------------------------------------------------------------------------
def BX(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    Y_p = Y.copy()
    X_p = np.zeros((case_number, case_number))
    B_p = np.zeros((n_P,n_P))
    B_pp = np.zeros((n_PQ,n_PQ))

    #-------------------------------------------------
    for i in xrange(case_number):
        Y_p[i][i] = complex(0,0)
        for j in xrange(case_number):
            if i != j:
                Y_p[i][i] -= Y_p[i][j]
    B = np.imag(Y_p)
    for i in xrange(n_P):
        for j in xrange(n_P):
            B_p[i][j] = B[index_P[i]][index_P[j]]
    #-------------------------------------------------
    g_b_round = np.zeros(case_number)
    for i in xrange(case_number):
        a = np.sum(Y[i])
        if LA.norm(a) > 1e-5:
            g_b_round[i] = np.reciprocal(np.imag(np.reciprocal(a)))

    for i in xrange(case_number):
        for j in xrange(case_number):
            if LA.norm(Y[i][j]) > 1e-5 and i!=j:
                X_p[i][j] = np.reciprocal(np.imag(np.reciprocal(Y[i][j])))
    for i in xrange(case_number):
        X_p[i][i] = g_b_round[i]
        for j in xrange(case_number):
            if i != j:
                X_p[i][i] -= X_p[i][j]
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_pp[i][j] = X_p[index_PQ[i]][index_PQ[j]]

    return B_p, - B_pp

# Stott ================================================================================================================
# Stott Original--------------------------------------------------------------------------------------------------------
def XB(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    _, B = util.get_G_B(Y)
    X = np.zeros((case_number, case_number))
    B_p = np.zeros((n_P,n_P))
    B_pp = np.zeros((n_PQ,n_PQ))
    for i in xrange(case_number):
        for j in xrange(case_number):
            if LA.norm(Y[i][j]) > 1e-5 and i != j:
                X[i][j] = np.reciprocal(np.imag(np.reciprocal(Y[i][j])))

    for i in xrange(case_number):
        for j in xrange(case_number):
            if i != j:
                X[i][i] -= X[i][j]

    for i in xrange(0, n_P):
        for j in xrange(0, n_P):
            B_p[i][j] = X[index_P[i]][index_P[j]]
    #---------------------------------------------------
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_pp[i][j] = B[index_PQ[i]][index_PQ[j]]

    return - B_p, B_pp

# Stott count r in B'---------------------------------------------------------------------------------------------------
def XB_r(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    _, B = util.get_G_B(Y)
    X = np.zeros((case_number, case_number))
    B_p = np.zeros((n_P,n_P))
    B_pp = np.zeros((n_PQ,n_PQ))
    for i in xrange(case_number):
        for j in xrange(case_number):
            if i != j:
                X[i][j] = B[i][j]

    for i in xrange(case_number):
        for j in xrange(case_number):
            if i != j:
                X[i][i] -= X[i][j]

    for i in xrange(0, n_P):
        for j in xrange(0, n_P):
            B_p[i][j] = X[index_P[i]][index_P[j]]
    #---------------------------------------------------
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_pp[i][j] = B[index_PQ[i]][index_PQ[j]]

    return B_p, B_pp
# Stott count ground in B'----------------------------------------------------------------------------------------------
def XB_ground(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    _, B = util.get_G_B(Y)
    X_p = np.zeros((case_number, case_number))
    B_p = np.zeros((n_P, n_P))
    B_pp = np.zeros((n_PQ, n_PQ))
    g_b_round = np.zeros(case_number)
    for i in xrange(case_number):
        a = np.sum(Y[i])
        if LA.norm(a) > 1e-5:
            g_b_round[i] = np.reciprocal(np.imag(np.reciprocal(a)))

    for i in xrange(case_number):
        for j in xrange(case_number):
            if LA.norm(Y[i][j]) > 1e-5 and i!=j:
                X_p[i][j] = np.reciprocal(np.imag(np.reciprocal(Y[i][j])))
    for i in xrange(case_number):
        X_p[i][i] = g_b_round[i]
        for j in xrange(case_number):
            if i != j:
                X_p[i][i] -= X_p[i][j]

    for i in xrange(0, n_P):
        for j in xrange(0, n_P):
            B_p[i][j] = X_p[index_P[i]][index_P[j]]
    # ---------------------------------------------------
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_pp[i][j] = B[index_PQ[i]][index_PQ[j]]

    return - B_p, B_pp

# Stott ================================================================================================================
# XX--------------------------------------------------------------------------------------------------------------------
def XX(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    X_p = np.zeros((case_number, case_number))
    B_p = np.zeros((n_P,n_P))
    B_pp = np.zeros((n_PQ,n_PQ))
    for i in xrange(case_number):
        for j in xrange(case_number):
            if LA.norm(Y[i][j]) > 1e-5 and i != j:
                X_p[i][j] = np.reciprocal(np.imag(np.reciprocal(Y[i][j])))

    for i in xrange(case_number):
        for j in xrange(case_number):
            if i != j:
                X_p[i][i] -= X_p[i][j]

    for i in xrange(0, n_P):
        for j in xrange(0, n_P):
            B_p[i][j] = X_p[index_P[i]][index_P[j]]
    #-------------------------------------------------
    g_b_round = np.zeros(case_number)
    for i in xrange(case_number):
        a = np.sum(Y[i])
        if LA.norm(a) > 1e-5:
            g_b_round[i] = np.reciprocal(np.imag(np.reciprocal(a)))

    for i in xrange(case_number):
        for j in xrange(case_number):
            if LA.norm(Y[i][j]) > 1e-5 and i!=j:
                X_p[i][j] = np.reciprocal(np.imag(np.reciprocal(Y[i][j])))
    for i in xrange(case_number):
        X_p[i][i] = g_b_round[i]
        for j in xrange(case_number):
            if i != j:
                X_p[i][i] -= X_p[i][j]

    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_pp[i][j] = X_p[index_PQ[i]][index_PQ[j]]
    return - B_p, - B_pp

def solve_V_theta(Y, index_PQ, index_P, n_PQ, n_P,\
                  V, theta, P, Q, case_number,\
                  ksai, max_iter, name, verbose=0):
    n_iter = 0
    G, B = util.get_G_B(Y)
    if name == 'BB':
        B_p, B_pp = BB(Y,index_PQ, index_P, n_PQ, n_P)
    elif name == 'BX':
        B_p, B_pp = BX(Y,index_PQ, index_P, n_PQ, n_P)
    elif name == 'XB':
        B_p, B_pp = XB(Y,index_PQ, index_P, n_PQ, n_P)
    elif name == 'XX':
        B_p, B_pp = XX(Y,index_PQ, index_P, n_PQ, n_P)
    elif name == 'XB_r':
        B_p, B_pp = XB_r(Y,index_PQ, index_P, n_PQ, n_P)
    elif name == 'XB_ground':
        B_p, B_pp = XB_ground(Y,index_PQ, index_P, n_PQ, n_P)

    B_p = np.mat(B_p)
    B_pp = np.mat(B_pp)

    start_time = time.time()

    while True:
        delta_P = delta_P_Q.cal_delta_P(V, theta, G, B, P, index_P, n_P, case_number)
        delta_P_by_V = np.mat(delta_P / V[index_P]).T

        V_delta_theta =  -np.array(B_p.I*delta_P_by_V)[:,0]
        # delta_theta = V_delta_theta/V[index_P]
        # TODO if V is not around 1 use the delta_theta above
        delta_theta = V_delta_theta
        theta[index_P] += delta_theta
        # ----------------------------------------------------------------
        delta_Q = delta_P_Q.cal_delta_Q(V, theta, G, B, Q, index_PQ, n_PQ, case_number)
        delta_Q_by_V = np.mat(delta_Q / V[index_PQ]).T
        delta_V = - np.array(B_pp.I*delta_Q_by_V)[:,0]
        V[index_PQ] += delta_V

        max_delta = np.max(np.abs(np.append(delta_P, delta_Q)))

        n_iter += 1
        if verbose >= 1:
            print n_iter
            print max_delta
        if n_iter > max_iter or math.isnan(max_delta) or math.isinf(max_delta):
            print 'not converge'
            exit(0)
            break
        if max_delta < ksai:
            print 'converge'
            break
    end_time = time.time()
    time_elapsed = end_time - start_time

    return V, theta, n_iter, time_elapsed