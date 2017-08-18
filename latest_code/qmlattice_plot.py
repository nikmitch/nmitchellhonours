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

# Complementary, and nearly standard libraries
import matplotlib.pyplot as plt
import matplotlib as mpl

# Utilities for this project
import qmlattice_utils as qm
import qmlattice as qmmain

# ==============================================================================
if __name__ == "__main__":
    
    # Enumerating the possible states in number state representation
#    number_states = qm.states_generator(parameters, report=False)
#    number_states = sorted(number_states, reverse=False)
#    qmmain.parameters["List of number states"] = number_states    
    
    fname = "20170817T152102_evolution_n.dat"
    (t, n) = qm.read_evolve_file(fname)
    print(t)
#    plt.figure()
#    plt.plot(t, d2, '-')
#    plt.title("Expansion coefficient |d_{0}(t)|^2 in the basis of number states.")
#    plt.grid(True)
#    plt.show()

    plt.plot(t, n[:,qmmain.nx*qmmain.ny-1])
    plt.title("Quantum expectation of number operator n on site {0:<3d}".format(qmmain.nx*qmmain.ny-1))
    plt.show()
