import scipy.io as sio
import numpy as np
case_number = 14
r_x_scale = ".0.75"
case_name = 'mat_data/case%d%s.mat' % (case_number,r_x_scale)

case = sio.loadmat(case_name)
Y = case['Y']
bus = case['bus']
gen = case['gen']
baseMVA = case['baseMVA']
V = case['V']
theta = case['theta']
np.savez('npy_data/case%d%s'% (case_number, r_x_scale), bus=bus,gen=gen,Y=Y, baseMVA=baseMVA, V=V, theta=theta)