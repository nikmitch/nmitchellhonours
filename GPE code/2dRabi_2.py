#!/usr/bin/env python
import sys
import numpy as np
import random as ra
from math import sin, cos, acos, pi
from cmath import exp
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy.linalg import solve

def hamSparse(psi):
  global N, nx, ny
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

    phi[ind] = (+       (hop[0][rjn])*psi[rjn]
                +       (hop[1][djn])*psi[djn]
                +np.conj(hop[0][ind])*psi[ljn]
                +np.conj(hop[1][ind])*psi[ujn]
                +U*pow(abs(psi[ind]),2);
  return phi

def hMatFull(hop):
  global hFull, nx, ny
  for ind in range(N):
    ix=ind%nx
    iy=(ind-ix)/nx

    rjn=ind+1;
    djn=ind+nx;

    if (ix==nx-1):
      rjn=ind+1-nx
    if (iy==ny-1):
      djn=ind+nx-N

    hFull[ind][rjn]=hop[0][ind]
    hFull[ind][djn]=hop[1][ind]
    hFull[ind][ind]=hop[2][ind]/2.0
  hFull =np.matrix(hFull)
  hFull+=hFull.getH()
  return hFull

nx=5
ny=nx
N=nx*ny
hop=np.zeros((3,N),dtype=np.complex128)

jAmp=1.0

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

hFull=np.zeros((N,N),dtype=np.complex128)
hFull=hMatFull(hop)
plt.imshow(abs(hFull),cmap='Greys')
plt.colorbar()
plt.savefig("hmat.png")
plt.clf()

w, v= eigh(hFull)
idx=w.argsort()[::-1]
w=w[idx]
v=v[:,idx]

plt.plot(w)
plt.savefig("spectrum.png")
plt.clf()

psi=np.matrix(np.zeros(N,dtype=np.complex128)).T
phi=np.matrix(np.zeros(N,dtype=np.complex128)).T

psi[0][0]=1.0

t=0

T=1.0e1
nStep=100
dt=T/float(nStep)

psiSP=psi.copy()
psiFU=psi.copy()
psiEO=v.T*psi

t=0.0
for ind in range(nStep+1):
  delt=-1.0j*dt

  psiSP1=delt*hamSparse(psiSP)
  psiSP2=delt*hamSparse(psiSP1)
  psiSP3=delt*hamSparse(psiSP2)
  psiSP+=psiSP1+psiSP2/2.0+psiSP3/6.0

  psiFU1=delt*hFull*psiFU
  psiFU2=delt*hFull*psiFU1
  psiFU3=delt*hFull*psiFU2
  psiFU+=psiFU1+psiFU2/2.0+psiFU3/6.0

  wMat=np.matrix(np.diagflat(np.exp(1.0j*t*w)))
  psiET=v*wMat*psiEO
  t+=dt
  fig = plt.figure(figsize=(3.375,3.375))
  imET=pow(abs(np.array(psiET).reshape(-1)),2).reshape(nx,ny)
  plt.imshow(imET,vmin=0.0,vmax=1.0)
  plt.savefig("psi_%04d.png" % ind)
  plt.clf()
  #  fig = plt.figure(figsize=(3.375,3.375))
  #  lgut=0.02
  #  bgut=0.02
  #  vgap=0.1
  #  hgap=0.1
  #  phig=(1.0-2.0*vgap-bgut)/2.0
  #  pwid=(1.0-2.0*hgap-lgut)/2.0
  #  
  #  ax1  = fig.add_axes([lgut+0*(pwid+vgap),bgut+1*(phig+vgap),pwid,phig])
  #  ax2  = fig.add_axes([lgut+1*(pwid+vgap),bgut+1*(phig+vgap),pwid,phig])
  #  ax3  = fig.add_axes([lgut+0*(pwid+vgap),bgut+0*(phig+vgap),pwid,phig])
  #  ax4  = fig.add_axes([lgut+1*(pwid+vgap),bgut+0*(phig+vgap),pwid,phig])
  #  
  #  print "writing sparse method at t=%.3f, " %t + "current norm:%.6f" % (psiSP.H*psiSP)[0,0]
  #  imSP=pow(abs(np.array(psiSP).reshape(-1)),2).reshape(nx,ny)
  #  imFU=pow(abs(np.array(psiFU).reshape(-1)),2).reshape(nx,ny)
  #  imET=pow(abs(np.array(psiET).reshape(-1)),2).reshape(nx,ny)
  #  ima1=ax1.imshow(imSP)
  #  ima2=ax2.imshow(imFU)
  #  ima3=ax3.imshow(imET)

  #  plt.savefig("psi_%06.3f.png" % t)
  #  plt.clf()
