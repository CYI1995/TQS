import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg

import statistics
import cmath

from scipy.linalg import hankel

def music(signal, w_min, w_max, dt, num_points=1000):
    num_T = len(signal)
    M = num_T // 2
    
    # 1. Construct Hankel Matrix
    H = hankel(signal[:M], signal[M-1:])
    
    # 2. SVD to find Noise Subspace
    U, S, Vh = np.linalg.svd(H)
    
    # Standard MUSIC: Identify the noise subspace dimension
    # Instead of S > 1, we usually look for the 'knee' in the singular values
    # Here we assume signal dimension 'k'. If unknown, thresholding works:
    n_signal = np.sum(S > (0.1 * np.max(S))) 
    Un = U[:, n_signal:] # Extract the noise subspace columns
    
    # 3. Create frequency grid (Vectorized)
    w_list = np.linspace(w_min, w_max, num_points) * dt
    m_indices = np.arange(M).reshape(-1, 1)
    
    # Steering Matrix: Each column is a phi_w vector
    steering_matrix = np.exp(1j * m_indices @ w_list.reshape(1, -1)) / np.sqrt(M)
    
    # 4. Project onto Noise Subspace: ||Un^H * phi_w||^2
    # Un.conj().T @ steering_matrix is a vectorized projection
    projection = np.linalg.norm(Un.conj().T @ steering_matrix, axis=0)**2
    
    # 5. Return Pseudo-spectrum (1/projection) so frequencies are peaks
    pseudo_spectrum = 1.0 / projection
    
    return w_list / dt, pseudo_spectrum


def signal(frequencies, amplitudes,t):

    s = 0 + 1j*0 
    L = len(frequencies)

    for l in range(L):
        s = s + amplitudes[l] * cmath.exp(1j*frequencies[l] * t)

    return s

        

num_T = 1000
nT = 50
T= nT*math.pi
dt = T/num_T


state1 = np.load('thermal_state.npy')
state2 = np.load('new_init_state.npy')



P = np.load('Parity.npy')
print(np.trace(state1 @ P))
print(np.trace(state2 @ P))


