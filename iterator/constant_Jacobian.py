#-*-coding:utf-8-*-
import time
import math
import numpy as np
import delta_P_Q


def constant_Jacobian(G, B, index_PQ, index_P, n_PQ, n_P):
    J = np.zeros((n_P + n_PQ, n_P + n_PQ))

    B_H = np.zeros((n_P, n_P))
    for i in xrange(0, n_P):
        for j in xrange(0, n_P):
            B_H[i][j] = B[index_P[i]][index_P[j]]

    G_N = np.zeros((n_P, n_PQ))
    for i in xrange(0, n_P):
        for j in xrange(0, n_PQ):
            G_N[i][j] = G[index_P[i]][index_PQ[j]]

    G_M = np.zeros((n_PQ, n_P))
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_P):
            G_M[i][j] = G[index_PQ[i]][index_P[j]]

    B_L = np.zeros((n_PQ, n_PQ))
    for i in xrange(0, n_PQ):
        for j in xrange(0, n_PQ):
            B_L[i][j] = B[index_PQ[i]][index_PQ[j]]

    J[0:n_P, 0:n_P] = B_H
    J[0:n_P,n_P: n_P + n_PQ] = -G_N
    J[n_P:n_P + n_PQ, 0:n_P] = G_M
    J[n_P:n_P + n_PQ, n_P:n_P + n_PQ] = B_L

    return J

def solve_V_theta(G, B, index_PQ, index_P, n_PQ, n_P,\
                  V, theta, P, Q, case_number, \
                  ksai, max_iter,verbose=0):
    n_iter = 0

    J = constant_Jacobian(G, B, index_PQ, index_P, n_PQ, n_P)
    J = np.mat(J)

    start_time = time.time()
    while True:
        delta_P = delta_P_Q.cal_delta_P(V, theta, G, B, P, index_P, n_P, case_number)
        delta_Q = delta_P_Q.cal_delta_Q(V, theta, G, B, Q, index_PQ, n_PQ, case_number)

        delta_P_by_V = delta_P / V[index_P]
        delta_Q_by_V = delta_Q / V[index_PQ]

        # 求得左手项
        delta_P_Q_by_V = np.mat(np.append(delta_P_by_V, delta_Q_by_V)).T
        delta_V_theta = - np.array(J.I * delta_P_Q_by_V)[:, 0]

        delta_theta = delta_V_theta[0:n_P] / V[index_P]
        delta_V = delta_V_theta[n_P:]

        # 更新V, theta
        V[index_PQ] += delta_V
        theta[index_P] += delta_theta

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