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
init_state = np.load('thermal_state.npy')

dim = len(ham[0])

T0 = np.load('T1.npy')
T1 = np.load('T2.npy')
T2 = np.load('T3.npy')
T3 = np.load('T4.npy')

num_T = 1000
nT = 50
T= nT*math.pi
dt = T/num_T


T_axis = np.zeros(num_T)

Realcomponent_ext_axis = np.zeros(num_T)
Realcomponent_noi_axis = np.zeros(num_T) 
Imagcomponent_ext_axis = np.zeros(num_T)
Imagcomponent_noi_axis = np.zeros(num_T)

noisy_init_state = np.load('new_init_state.npy')

Id= np.identity(dim)
Kick_Operator1 = (Id + 1j*A)/math.sqrt(2)
kicked_state1 = Kick_Operator1.dot((init_state).dot(np.conj(Kick_Operator1).T))
kicked_state2 = Kick_Operator1.dot((noisy_init_state).dot(np.conj(Kick_Operator1).T))

Kick_Operator2 = (P + A)/math.sqrt(2)
total_parity = np.trace(init_state @ P).real
total_parity_noi = np.trace(noisy_init_state @ P).real  
print(total_parity_noi - total_parity)

rho_sym = (init_state + init_state @ P)/ (1 + total_parity)
kicked_state3 = Kick_Operator2.dot(rho_sym.dot(np.conj(Kick_Operator2).T))

noisy_rho_sym = (noisy_init_state + noisy_init_state @ P)/ (1 + total_parity_noi)
kicked_state4 = Kick_Operator2.dot(noisy_rho_sym.dot(np.conj(Kick_Operator2).T))

rho_asym = (init_state - init_state @ P)/ (1 - total_parity)
kicked_state5 = Kick_Operator2.dot(rho_asym.dot(np.conj(Kick_Operator2).T))

noisy_rho_asym = (noisy_init_state - noisy_init_state @ P)/ (1 - total_parity_noi)
kicked_state6 = Kick_Operator2.dot(noisy_rho_asym.dot(np.conj(Kick_Operator2).T))

U_Tro_dt = scipy.linalg.expm(-1j*T3*dt)
U_Tro_dt = scipy.linalg.expm(-1j*T2*dt).dot(U_Tro_dt)
U_Tro_dt = scipy.linalg.expm(-1j*T1*dt).dot(U_Tro_dt)
U_Tro_dt = scipy.linalg.expm(-1j*T0*dt).dot(U_Tro_dt)

U_Ext_dt = scipy.linalg.expm(-1j*ham*dt)

At_Tro = A.copy()
At_Ext = A.copy()

for n in range(num_T):

    print(n)

    t = (n+1)*dt
    At_Tro = (np.conj(U_Tro_dt).T).dot(At_Tro.dot(U_Tro_dt))
    At_Ext = (np.conj(U_Ext_dt).T).dot(At_Ext.dot(U_Ext_dt))

    w1 = np.trace(kicked_state1.dot(At_Ext)).real
    w2 = np.trace(kicked_state2.dot(At_Tro)).real

    v1 = np.trace(kicked_state3.dot(At_Ext)).real
    v2 = np.trace(kicked_state4.dot(At_Tro)).real
    v3 = np.trace(kicked_state5.dot(At_Ext)).real
    v4 = np.trace(kicked_state6.dot(At_Tro)).real

    T_axis[n] = t

    Imagcomponent_ext_axis[n] = w1
    Imagcomponent_noi_axis[n] = w2

    Realcomponent_ext_axis[n] = 0.5*(1 + total_parity) * v1 - 0.5*(1 - total_parity) * v3
    Realcomponent_noi_axis[n] = 0.5*(1 - total_parity_noi) * v2 - 0.5*(1 - total_parity_noi) * v4


plt.plot(T_axis, Imagcomponent_ext_axis,label = 'Im, Exact + Exact')
plt.plot(T_axis, Imagcomponent_noi_axis,label = 'Im, Noisy + Trotter')
plt.plot(T_axis, Realcomponent_ext_axis,label = 'Re, Exact + Exact')
plt.plot(T_axis, Realcomponent_noi_axis,label = 'Re, Noisy + Trotter')

np.save('signal_Im_ExtExt.npy',Imagcomponent_ext_axis)
np.save('signal_Im_NoiTro.npy',Imagcomponent_noi_axis)
np.save('signal_Re_ExtExt.npy',Realcomponent_ext_axis)
np.save('signal_Re_NoiTro.npy',Realcomponent_noi_axis)
plt.legend()
plt.show()



