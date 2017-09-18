#!/usr/bin/env python
import sys
import time
import numpy as np
import random as ra
from math import sin, cos, acos, pi
from cmath import exp
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy.linalg import solve


def hamSparse(psi):
    global N, nx, ny, U
    for ind in range(N):
        ix=ind%nx;
        iy=((ind-ix)/nx);
      
        rjn=ind-1;
        djn=ind-nx;
        ljn=ind+1;   
        ujn=ind+nx;
      
        if (ix==0):
            rjn=ind-1+nx
        if (iy==0):
            djn=ind-nx+N
        if (ix==nx-1):
            ljn=ind+1-nx
        if (iy==ny-1):
            ujn=ind+nx-N

        phi[ind] =       hop[0][rjn]  * psi[rjn] \
                +        hop[1][djn]  * psi[djn] \
                +np.conj(hop[0][ind]) * psi[ljn] \
                +np.conj(hop[1][ind]) * psi[ujn] \
                +U*np.power(np.abs(psi[ind]), 2)*psi[ind]
        
    return phi

nx=10
ny=nx
N=nx*ny
hop=np.zeros((3,N),dtype=np.complex128)

jAmp=-1.0
U=0.0
for ind in range(N):
  ix=ind%nx
  iy=(ind-ix)/nx
  hop[0][ind]=jAmp
  hop[1][ind]=jAmp
  if ix==nx-1:
    hop[0][ind]=0.0
  if iy==ny-1:
    hop[1][ind]=0.0
  hop[2][ind]=0


#w, v= eigh(hFull)
#idx=w.argsort()[::-1]
#w=w[idx]
#v=v[:,idx]
#
#plt.plot(w)
#plt.savefig("spectrum.png")
#plt.clf()

psi=np.matrix(np.zeros(N,dtype=np.complex128)).T
phi=np.matrix(np.zeros(N,dtype=np.complex128)).T

psi[:][:]=1.0

t=0

T=100
dt=1e-1
nStep=1000

psiSP=psi.copy()

t=0.0
tic=time.time()
outputs=[]

tolerance=1e-6
scale=1.0
kay=0

plt.ioff()

while (kay<nStep):

  fig = plt.figure(figsize=(3.375,3.375))
  imSP=pow(abs(np.array(psiSP).reshape(-1)),2).reshape(nx,ny)
  plt.imshow(imSP,vmin=0.0,vmax=1.0)
  plt.savefig("psi_%04d" % kay+ "_u_%+07.4f.png" % U)
  plt.close()
  plt.clf()
  current_probs=np.diagflat(np.abs(psiSP))*np.abs(psiSP)
  kay+=1
  print("t=%.3f, " %t + "dt: %f, " %dt + "dN:%7.3e" %(1.0-np.sum(current_probs)))

  Tloc=float(kay)*float(T)/float(nStep)
  print("stepping up to time %f" % Tloc)

  while (t<Tloc):
    while (True):
      delt=-1.0j*dt
      k1=delt*hamSparse(psiSP)
      k2=delt*hamSparse(psiSP+(1.0/4.0)*k1)
      k3=delt*hamSparse(psiSP+(3.0/32.0)*k1+(9.0/32.0)*k2)
      k4=delt*hamSparse(psiSP+(1932.0/2197.0)*k1-(7200.0/2197.0)*k3+(7296.0/2197.0)*k3)
      k5=delt*hamSparse(psiSP+(439.0/216.0)*k1-(8.0)*k2+(3680.0/513.0)*k3-(845.0/4104.0)*k4)
      k6=delt*hamSparse(psiSP-(8.0/27.0)*k1+(2.0)*k2-(3544.0/2565.0)*k3+(1859.0/4104.0)*k4-(11.0/40.0)*k5)

      residual_error=(1.0/dt)*abs(1.0/360.0*k1-128.0/4275.0*k3-2197.0/75240.0*k4+1.0/50.0*k5+2.0/55.0*k6).max()

      #rescale timestep and retry.
      scale=0.84*(tolerance/residual_error)**(0.25)
      if scale<0.1:
        dt=0.1*dt
      elif scale>4.0:
        dt=4.0*dt
      else:
        dt=scale*dt

      if (residual_error<tolerance):
        #exit the loop
        break

      #print("err=%.3e " %residual_error + "dt=%.3f, " %dt + "rescale:%7.3e" %(scale))

    psiSP+=25.0/216.0*k1+1408.0/2565.0*k3+2197.0/4104.0*k4-1.0/5.0*k5    
    t+=dt
    

  #current_first_site=current_probs[0]
  #current_first_site=np.asarray(current_first_site)[0][0]
  #outputs.append([t,current_first_site])

print("Time evolution required:         ", time.time() - tic, " seconds.")  

#outputs=np.asarray(outputs)
#plt.plot(outputs[:,0],outputs[:,1])  
#plt.title("RKF GPE code for 3by3")
#plt.xlabel("time")
#plt.ylabel("<n1>")

  
