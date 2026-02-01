import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import statistics
import cmath



ham = np.load('ham.npy')
A = np.load('A0.npy')
P = np.load('Parity.npy')

w0_list = np.load('w_list_extext.npy')
jw0_list = np.load('jw_list_extext.npy')
dim = len(ham[0])

Amplitudes = [] 
Energy_gaps = []

eig,vec = np.linalg.eigh(ham)

idx_gs = np.argsort(eig)[0]
vec_gs = vec[:,idx_gs]
E0 = eig[idx_gs].real
for m in range(dim):

    print(m)

    Em = eig[m].real 
    vec_m = vec[:,m]

    amplitude = abs(np.vdot(vec_m,A.dot(vec_gs)))**2
    energy_gap = Em - E0

    if(abs(amplitude) > 1e-7):
        Amplitudes.append(amplitude)
        Energy_gaps.append(energy_gap)

np.save('Amplitudes_GS.npy',Amplitudes)
np.save('Energy_gap_GS.npy',Energy_gaps)

plt.plot(w0_list, jw0_list)
plt.scatter(Energy_gaps, Amplitudes)
plt.show()





