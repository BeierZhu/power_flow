#-*-coding:utf-8-*-
# Description: Calculate power flow
# Author: Beier ZHU

import argparse
import numpy as np

from iterator import FDL, constant_Jacobian
from iterator import util

# Read parameters from command line-------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--iteration_matrix",'-m',help="iteration matrix: 1 constant Jacobian 2: XB", default=2)
parser.add_argument("--case_number", default=14)
parser.add_argument("--ksai",default=1e-6)
parser.add_argument("--max_iter",default=1e3)
parser.add_argument("--verbose",help="print detailed information", default=0)
parser.add_argument("--scale", help='r scale for compare BX et BX', default='1')

args = parser.parse_args()
ksai = float(args.ksai)
max_iter = int(args.max_iter)
case_number = int(args.case_number) # alias: N + 1 or n + 1 if #slack == 1
verbose = int(args.verbose)
matrix_type = int(args.iteration_matrix)
scale = args.scale
# mapping the matrix type (int) to matrix name (string)
switcher = {
    0: "Constant Jacobian",
    1: 'BB',
    2: 'XB',
    3: 'BX',
    4: 'XX',
    5: 'XB_r',
    6: 'XB_ground'
}

matrix_name = switcher.get(matrix_type, 'Unknown')
print 'Method: %s' %matrix_name
print 'case number: %d' %case_number
print 'r scale: %s' %scale
# ----------------------------------------------------------------------

# load Y and bus information
# 记录平衡节点位置,PQ,PV数量和index
# 生成P节点位置与数量
Y, bus, gen, baseMVA, result_V, result_theta = util.load_case(case_number, scale)
P, Q, V, theta = util.get_P_Q_V_theta(bus, gen, case_number, baseMVA)

G, B = util.get_G_B(Y)
index_PQ, n_PQ, index_PV, n_PV, index_slack= util.get_index_count_PQ_PV_slack(bus, case_number)
index_P, n_P = util.get_index_count_P(index_PQ, index_PV)


if matrix_name == "Constant Jacobian":
    V, theta, n_iter, time_elapsed = constant_Jacobian.solve_V_theta(G, B, index_PQ, index_P, n_PQ, n_P, \
                                                                     V, theta, P, Q, case_number, \
                                                                     ksai, max_iter, verbose)
else:
    V, theta, n_iter, time_elapsed = FDL.solve_V_theta(Y, index_PQ, index_P, n_PQ, n_P, \
                                                                     V, theta, P, Q, case_number, \
                                                                     ksai, max_iter, matrix_name, verbose)

difference_V, difference_theta = util.difference_max_V_theta(V, result_V, theta, result_theta)

print 'converge at iteration: %d' %n_iter
print 'time elapsed: %f s' %time_elapsed


if verbose >= 2:
    print 'V calculated by powerflow:'
    print result_V
    print 'V calculated by handcrafted program:'
    print V
    print 'difference max V : %f' % difference_V
    print 'theta calculated by powerflow:'
    print result_theta
    print 'theta calculated by handcrafted program:'
    print np.rad2deg(theta)
    print 'differecne max theta: %f' % difference_theta

