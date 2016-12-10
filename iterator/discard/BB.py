import time

import numpy as np

import iterator.delta_P_Q
import iterator.util


# BB
def BB(Y,index_PQ, index_P, n_PQ, n_P):
    case_number, _ = np.shape(Y)
    Y_p = Y
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

def solve_V_theta(Y, index_PQ, index_P, n_PQ, n_P,\
                  V, theta, P, Q, case_number,\
                  ksai, max_iter, verbose=0):
    n_iter = 0
    G, B = iterator.util.get_G_B(Y)
    B_p, B_pp = BB(Y,index_PQ, index_P, n_PQ, n_P)
    B_p = np.mat(B_p)
    B_pp = np.mat(B_pp)

    start_time = time.time()

    while True:
        delta_P = iterator.delta_P_Q.cal_delta_P(V, theta, G, B, P, index_P, n_P, case_number)
        delta_P_by_V = np.mat(delta_P / V[index_P]).T

        V_delta_theta =  -np.array(B_p.I*delta_P_by_V)[:,0]
        delta_theta = V_delta_theta/V[index_P]

        theta[index_P] += delta_theta
        # ----------------------------------------------------------------
        delta_Q = iterator.delta_P_Q.cal_delta_Q(V, theta, G, B, Q, index_PQ, n_PQ, case_number)
        delta_Q_by_V = np.mat(delta_Q / V[index_PQ]).T
        delta_V = - np.array(B_pp.I*delta_Q_by_V)[:,0]
        V[index_PQ] += delta_V

        max_delta = np.max(np.abs(np.append(delta_theta, delta_V)))

        n_iter += 1
        if verbose >= 1:
            print n_iter
            print max_delta

        if n_iter > max_iter or max_delta < ksai:
            break
    end_time = time.time()
    time_elapsed = end_time - start_time

    return V, theta, n_iter, time_elapsed