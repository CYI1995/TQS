import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import string
import matplotlib as mpl


def sparse(array,L,k):

    M = int(L/k)
    new_array = np.zeros(M)
    for j in range(M):
        idx = (j+1)*k
        new_array[j] = array[idx-1]
    return new_array

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

font = {'family' : 'normal',
         'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 




T_list = np.linspace(0,50*math.pi,1000)
signalA= np.load('signal_Im_ExtExt.npy')
signalC= np.load('signal_Im_NoiTro.npy')
signalD= np.load('signal_Re_ExtExt.npy')
signalF= np.load('signal_Re_NoiTro.npy')


w0_list = np.load('w_list_extext.npy')
jw0_list = np.load('jw_list_extext.npy')
w2_list = np.load('w_list_noitro.npy')
jw2_list = np.load('jw_list_noitro.npy')

fig, axs = plt.subplot_mosaic([['(a)'], ['(b)'],['(c)']],
                              figsize = (6,12),
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == '(a)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'Im $[C(A,A,t)]$')
        ax.plot(T_list,signalA,label = 'Exact + Exact')
        ax.plot(T_list,signalC,linestyle = 'dashed',label = 'Noisy + Trotter')
        ax.set_xlim(0,50*math.pi)
        ax.legend(loc = 1)
    if(label == '(b)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'Re $[C(A,A,t)]$')
        ax.plot(T_list,signalD,label = 'Exact + Exact')
        ax.plot(T_list,signalF,linestyle = 'dashed',label = 'Noisy + Trotter')
        ax.set_xlim(0,50*math.pi)
        ax.legend(loc = 1)
    if(label == '(c)'):
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$R(\omega)$')
        ax.plot(w0_list,jw0_list,c = 'k',label = 'Exact + Exact')
        ax.plot(w2_list,jw2_list,c = 'b',linestyle = 'dashed',label = 'Noisy + Trotter')
        ax.set_xticks(np.arange(-np.pi/2, np.pi/2+0.01, np.pi/4))
        labels = [r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$']
        ax.set_xticklabels(labels)
        ax.set_xlim(-math.pi/2, math.pi/2 + 0.01)
        ax.set_ylim(-0.1,1.1)
        ax.legend(loc = 1)

plt.savefig('Thermal_Weak.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()

