import numpy as np
def load_case(case_number, scale):
    if scale == '1':
        case = np.load('data/npy_data/case%d.npz' % case_number)
    else:
        case = np.load('data/npy_data/case%d.%s.npz' % (case_number, scale))
    Y = case['Y']
    bus = case['bus']
    gen = case['gen']
    baseMVA = case['baseMVA'][0][0]
    result_V = case['V'].reshape(case_number)
    result_theta = case['theta'].reshape(case_number)
    return Y, bus, gen, baseMVA, result_V, result_theta

def get_index_count_PQ_PV_slack(bus, case_number):
    n_PQ = 0
    n_PV = 0
    index_PQ = np.zeros(case_number, dtype=int)
    index_PV = np.zeros(case_number, dtype=int)
    index_slack = -1

    for i in xrange(case_number):
        type = bus[i][1]  # 3 slack bus; 2 PV bus; 1 PQ bus
        if type == 3.:
            if index_slack == -1:
                index_slack = i
            else:
                print 'more than one slack bus'
                exit(0)
        if type == 2.:
            index_PV[n_PV] = i
            n_PV += 1

        if type == 1.:
            index_PQ[n_PQ] = i
            n_PQ += 1

    index_PQ = index_PQ[0:n_PQ]
    index_PV = index_PV[0:n_PV]

    return index_PQ, n_PQ, index_PV, n_PV, index_slack

def get_index_count_P(index_PQ, index_PV):
    index_P = np.sort(np.append(index_PQ, index_PV))
    n_P = index_P.size
    return index_P, n_P

def get_G_B(Y):
    G = np.real(Y)
    B = np.imag(Y)
    return G, B

def get_P_Q_V_theta(bus, gen, case_number,baseMVA):
    gen_index = gen[:,0].astype(np.int) - 1
    P = np.zeros(case_number)
    Q = np.zeros(case_number)
    V = np.ones(case_number)
    theta = np.zeros(case_number)

    for i in xrange(case_number):
        type = bus[i][1]
        if type == 3.:
            V[i] = bus[i][7]
            theta[i] = np.deg2rad(bus[i][8])
        if type == 2.:
            if i in gen_index:
                j = np.where(gen_index == i );index_j = j[0][0]
                P[i] = gen[index_j][1] - bus[i][2]
            else:
                P[i] = -bus[i][2]
            V[i] = bus[i][7]
        if type == 1.:
            if i in gen_index:
                j = np.where(gen_index == i); index_j = j[0][0]
                P[i] = gen[index_j][1] - bus[i][2]
                Q[i] = gen[index_j][2] - bus[i][3]
            else:
                P[i] = - bus[i][2]
                Q[i] = - bus[i][3]

    return P/baseMVA, Q/baseMVA, V, theta

def difference_max_V_theta(V_cal, V_result, theta_cal, theta_result):
    difference_V = np.max(np.abs(V_result - V_cal))
    difference_theta = np.max(np.abs(np.deg2rad(theta_result) - theta_cal))
    return difference_V, difference_theta