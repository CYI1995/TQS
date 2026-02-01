
import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random
import scipy.linalg
import statistics
import cmath
from functools import reduce
from numpy import kron

beta = 1
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

def pauli_x(j: int, n: int) -> np.ndarray:

    ops = [I] * n
    ops[j] = X
    return reduce(kron, ops)

def pauli_y(j: int, n: int) -> np.ndarray:

    ops = [I] * n
    ops[j] = Y
    return reduce(kron, ops)

def pauli_z(j: int, n: int) -> np.ndarray:
    
    ops = [I] * n
    ops[j] = Z
    return reduce(kron, ops)

def JordanWigner_ci(i, num):
    dim = 2**num
    X = pauli_x(i, num)
    Z = pauli_z(i, num)
    Y = -1j * X @ Z
    Product_Z = np.eye(dim)
    for j in range(i):
        Product_Z = Product_Z @ pauli_z(j, num)
    sigma_minus = (X - 1j * Y) / 2
    return Product_Z @ sigma_minus

def JordanWigner_ni(i,num):

    dim = 2**num 
    return 0.5*(np.identity(dim) - pauli_z(i,num))



num_sites = 12
dim = 2**num_sites

t = 1
U = 0.1

# Or more compactly using list of operators
c_list = [JordanWigner_ci(i, num_sites) for i in range(num_sites)]

T1 = np.zeros((dim,dim),dtype = complex)
T2 = np.zeros((dim,dim),dtype = complex)
T3 = np.zeros((dim,dim),dtype = complex)
T4 = np.zeros((dim,dim),dtype = complex)

for i in np.array([0,1,6,7]):
    j = (i + 2)
    T1 -= t * (c_list[i].conj().T @ c_list[j] + c_list[j].conj().T @ c_list[i])

for i in np.array([2,3,8,9]):
    j = (i + 2)
    T2 -= t * (c_list[i].conj().T @ c_list[j] + c_list[j].conj().T @ c_list[i])

for i in np.array([0,1,2,3,4,5]):
    j = (i + 6)
    T3 -= t * (c_list[i].conj().T @ c_list[j] + c_list[j].conj().T @ c_list[i])

for i in range(num_sites):
    j = (i + 1) % num_sites
    ni = JordanWigner_ni(i, num_sites)
    nj = JordanWigner_ni(j, num_sites)
    T4 += U * (ni @ nj)  # nearest-neighbor density-density interaction

ham0 = T1 + T2 + T3 + T4
eig = np.linalg.svdvals(ham0)
norm = np.max(eig)
ham = ham0 * math.pi/norm 

c0 = c_list[0]
A0 = (c0 + np.conj(c0).T)/2
P = np.eye(dim)
for j in range(num_sites):
    P = P @ pauli_z(j, num_sites)



np.save('ham.npy',ham)
np.save('T1.npy',T1*math.pi/norm)
np.save('T2.npy',T2*math.pi/norm)
np.save('T3.npy',T3*math.pi/norm)
np.save('T4.npy',T4*math.pi/norm)
np.save('A0.npy',A0)
np.save('Parity.npy',P)


eigs = np.linalg.eigvalsh(ham0)
E0 = np.min(eigs)
ham = ham - E0 * np.identity(dim)
thermal_state = scipy.linalg.expm(-beta*ham)
thermal_state = thermal_state/np.trace(thermal_state).real
np.save('thermal_state.npy',thermal_state)

Gaussian_noise_real = np.random.randn(dim,dim)
Gaussian_noise_imag = np.random.randn(dim,dim)
for i in range(dim):
    Gaussian_noise_real[i][i] = 0
    Gaussian_noise_imag[i][i] = 0
Gaussian_noise = Gaussian_noise_real + 1j*Gaussian_noise_imag 
Gaussian_noise = 0.5*(Gaussian_noise + np.conj(Gaussian_noise).T)
svls = np.linalg.svdvals(Gaussian_noise)
Noise_Hamiltonian = Gaussian_noise/np.sum(svls)


eps = 0.1
Gaussian_Noise = scipy.linalg.expm(-Noise_Hamiltonian)
trace = np.trace(Gaussian_Noise).real
Gaussian_Noise = Gaussian_Noise/trace
noisy_init_state = thermal_state + eps * Gaussian_Noise 

new_init_state = noisy_init_state/np.trace(noisy_init_state).real
np.save('new_init_state.npy',new_init_state)




