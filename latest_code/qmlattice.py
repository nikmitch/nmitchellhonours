#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Quantum chain project:
    Main script file for starting a simulation
'''

# Standard libraries
import time
import random
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.linalg import eigh


# Complementary, and nearly standard libraries
import matplotlib.pyplot as plt
import matplotlib as mpl

# Utilities for this project
import qmlattice_utils as qm

# == PHYSICAL PARAMETERS =======================================================
nx = 3
ny = nx
P  = 1

J  = -1/np.sqrt(2)
JP = -2/5*np.sqrt(2)
U  =  0.0

LATTICE_TYPE = "full"    # Options are: "full", "random_one", "manual"

# If "manual" is chosen, then a list of pairs of indices has to be provided, in
# the format, e.g., [((0,6),(2,8))], which means that there is a link between
# site (0,6) = 6th site on chain 0th and the 8th site on the 2nd chain. No check
# is made that the links are # physically meaningful or not!
MANUAL_LINKS = [((0,0), (1,0)),
                ((0,1), (1,1)),
                ((1,0), (2,0)),
                ((1,1), (2,1)),
                ((1,2), (2,2)),
               ]

# == SIMULATION PARAMETERS =====================================================
np.set_printoptions(linewidth=250)

TMIN = 0
TMAX = int(100)
TN   = int(1e5)

BLOCK_LENGTH = 3000

parameters = {"Number of sites per chain":       nx,
              "Number of chains":                ny,
              "Total number of particles":       P,
              "Coupling within chain (J)":       J,
              "Coupling between chains (J')":    JP,
              "Onsite interaction strength (U)": U,
              "Connections between chains":      LATTICE_TYPE,
              "Prescribed links between chains": MANUAL_LINKS,
              "Time starts":                     TMIN,
              "Time ends":                       TMAX,
              "Number of time steps":            TN,
              "Maximal block length in export":  BLOCK_LENGTH,
             }

# ==============================================================================
if __name__ == "__main__":
    tic = time.time()

    # Enumerating the possible states in number state representation
    number_states = qm.states_generator(parameters, report=False)
    number_states = sorted(number_states, reverse=False)
    parameters["List of number states"] = number_states

    # The pairs of site indices between which hopping is allowed
    qm.index_pairs(parameters)

    # Hamiltonian operator in number state representation
    H_onsite = qm.onsite_hamiltonian(parameters)
    H_hopping = qm.hopping_hamiltonian(parameters)
    H_full = H_hopping + H_onsite
    

    # Energy eigenvalues and eigenvectors (for time evolution)
    spectrum = qm.calculate_spectrum(H_full, method="dense")
    qm.export_spectrum(parameters, spectrum)

    # Initial state of the time evolution
    initial_state = np.zeros(len(number_states), dtype=np.complex64)
    initial_state[0] = 1
    print("Setting up calculation required: ", time.time() - tic, " seconds.")

    tic = time.time()
    qm.evolve(initial_state, spectrum, parameters)
    print("Time evolution required:         ", time.time() - tic, " seconds.")

    qm.plot_lattice(parameters)

