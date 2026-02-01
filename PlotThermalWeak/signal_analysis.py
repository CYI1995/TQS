import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg

import statistics
import cmath

from scipy.linalg import hankel

def music(signal, w_min, w_max, dt, num_points=10000):
    num_T = len(signal)
    M = num_T // 2
    
    # 1. Construct Hankel Matrix
    H = hankel(signal[:M], signal[M-1:])
    
    # 2. SVD to find Noise Subspace
    U, S, Vh = np.linalg.svd(H)
    
    # Standard MUSIC: Identify the noise subspace dimension
    # Instead of S > 1, we usually look for the 'knee' in the singular values
    # Here we assume signal dimension 'k'. If unknown, thresholding works:
    n_signal = np.sum(S > (0.01 * np.max(S))) 
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
    
    return w_list / dt, projection


def extract_amplitudes(signal,frequencies,dt):

    L1 = len(frequencies)
    L2 = len(signal) 

    Jump = int(L2/L1)

    Kernel = np.zeros((L1,L1))
    SignalOnKernel = np.zeros(L1)
    for i in range(L1):
        t_temp = (Jump*i+1)*dt
        SignalOnKernel[i] = signal[Jump*i+1]
        for j in range(L1):
            Kernel[i][j] = math.sin(frequencies[j]*t_temp)

    Amplitues = np.linalg.inv(Kernel).dot(SignalOnKernel)

    return Amplitues

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

signal_ext = 1j * np.load('signal_Im_ExtExt.npy') + np.load('signal_Re_ExtExt.npy')
signal_noi = 1j * np.load('signal_Im_NoiTro.npy') + np.load('signal_Re_NoiTro.npy')

w_list,jw_list = music(signal_ext ,-math.pi/2,math.pi/2,dt)
np.save('w_list_extext.npy',w_list)
np.save('jw_list_extext.npy',jw_list/np.max(jw_list))
plt.plot(w_list,jw_list,c = 'k',label = 'Ext')


w_list,jw_list = music(signal_noi,-math.pi/2,math.pi/2,dt)
np.save('w_list_noitro.npy',w_list)
np.save('jw_list_noitro.npy',jw_list/np.max(jw_list))
plt.plot(w_list,jw_list,c = 'b',linestyle = '--',label = 'Noi')

plt.legend()
plt.show()





