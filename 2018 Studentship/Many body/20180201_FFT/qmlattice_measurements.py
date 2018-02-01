#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.fftpack as spfft
import scipy.sparse.linalg as splin




# Non-standard run-time parameters
np.set_printoptions(linewidth=250)

# == FUNCTION DECLARATIONS =====================================================
def occupation(d, parameters):
    '''Occupation of site.'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]
    number_of_sites = nx * ny

    number_states = parameters["List of number states"]

    d2 = np.square(np.abs(d))
    (nrow, ncol) = np.shape(d2)
    nexp = np.zeros((nrow, number_of_sites), dtype=np.float32)
    for site in range(number_of_sites):
        nj = np.array([state[site] for state in number_states])
        for n in range(nrow):
            nexp[n, site] = np.dot(d2[n, :], nj)

    return(nexp)


def ft_Rigol(d, BB, parameters):
    '''Fourier transformation as defined in Rigol's 2007 paper, eq.(4).'''
    # M. Rigol, V. Dunjko, V. Yurovsky, and M. Olshanii,
    # Relaxation in a Completely Integrable Many-Body Quantum System: An Ab
    # Initio Study of the Dynamics of the Highly Excited States of 1D Lattice
    # Hard-Core Bosons, PRL 98, 050405 (2007)

    # Calculating zero momenta at different times using a loop
    pmax = np.shape(d)[0]
    fk = np.zeros((pmax, 1))
    for p in range(pmax):
        dp = d[p,:]
        tmp = np.conj(dp) * BB * np.transpose(dp)
        fk[p, 0] = np.real(tmp[0, 0])

    return(fk)


def ft(f, parameters, normalise=True):
    '''Fourier transform of f(x)'''
    (nrows, ncols) = np.shape(f)

    fk = spfft.fft(f)
    fk = fk[:, 0:ncols//2+1]
    fk = np.power(np.abs(fk), 2)

    if normalise:
        P = parameters["Total number of particles"]
        fk = fk/(P**2)
    return(fk)


def expectation(operator, state):
    '''Instantaneous value of the hopping energy, i.e, <state| Operator |state>'''
    expval = None
    state = np.ndarray.flatten(state)
    bra = np.conj(state)
    ket = operator.dot(state)
    expval = bra.dot(ket)
    return(expval)

# ==============================================================================
if __name__ == "__main__":
    pass
