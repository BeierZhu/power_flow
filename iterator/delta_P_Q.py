import numpy as np
import math
def cal_delta_P(V, theta, G, B, P,\
                index_P, n_P, case_number):
    # case_number: alias for n + 1
    delta_P = np.zeros(n_P)

    for i in xrange(n_P):
        i_map = index_P[i]
        delta_P[i] = P[i_map]
        for j in xrange(case_number):
            delta_P[i] -= V[i_map]*V[j]*( G[i_map][j]*math.cos(theta[i_map] - theta[j]) + B[i_map][j]*math.sin(theta[i_map]-theta[j]) )

    return  delta_P

def cal_delta_Q(V, theta, G, B, Q,\
                index_PQ, n_PQ, case_number):
    # case_number is the alias for n + 1
    delta_Q = np.zeros(n_PQ)

    for i in xrange(n_PQ):
        i_map = index_PQ[i]
        delta_Q[i] = Q[i_map]
        for j in xrange(case_number):
            delta_Q[i] -= V[i_map]*V[j]*( G[i_map][j]*math.sin(theta[i_map] - theta[j]) - B[i_map][j]*math.cos(theta[i_map]-theta[j]) )

    return  delta_Q