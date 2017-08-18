# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:17:26 2017

@author: mitni349
"""
import numpy as np
from numpy.linalg import eigh


TMIN=2
TMAX=20002
TN=10000

#H=np.matrix([[0,-1,0,0,0],[-1,0,-1,0,0],[0,-1,0,-1,0],[0,0,-1,0,-1],[0,0,0,-1,0]])


#(eighevs,eigvecs)=eigh(H)
#eigvals=[np.sqrt(3),1,0,-1,-np.sqrt(3)]

#print(eigvecs)
#
#a=(eigvecs[3,:])
#
#a=np.transpose(a)
#print(a)
#b=H*a
#print(b)


TIMES=np.linspace(TMIN,TMAX,TN)

def diffcalc(times):
    epsilonvalues=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    counter=0
    current_eps=epsilonvalues[counter]
    closetimes=[0]*7

    f = lambda t: 1/12 * np.exp(1j * np.sqrt(3)  * t)  + \
                  1/4  * np.exp(1j               * t)  + \
                  1/3                                  + \
                  1/4  * np.exp(-1j              * t)  + \
                  1/12 * np.exp(-1j * np.sqrt(3) * t)  - 1

    for t in times:
        if np.abs(f(t))**2 < current_eps and counter < 6:
            closetimes[counter] = t
            counter += 1
            current_eps = epsilonvalues[counter]

    return (epsilonvalues,closetimes)

(epsvals,close) = diffcalc(TIMES)
print(epsvals)
print(close)    