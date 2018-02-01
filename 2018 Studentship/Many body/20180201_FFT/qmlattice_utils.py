#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import random
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin

# Complementary, and nearly standard libraries
import matplotlib.pyplot as plt
# import matplotlib as mpl

# Utilities for this project
import qmlattice_io as qmio
import qmlattice_plotting as qmplt
import qmlattice_measurements as qmmeas

# Non-standard run-time parameters
np.set_printoptions(linewidth=250)

# == FUNCTION DECLARATIONS =====================================================
def ip_slow(n):
    '''Integer partitioning (slow)'''
    answer = set()
    answer.add((n, ))
    for x in range(1, n):
        for y in ip_slow(n - x):
            answer.add(tuple(sorted((x, ) + y)))
    return(answer)


def ip_fast(n):
    '''Integer partitioning (fast)'''
    a = [0 for _ in range(n+1)]
    k = 1
    y = n-1
    while (k != 0):
        x = a[k-1] + 1
        k = k-1
        while (2*x <= y):
            a[k] = x
            y = y-x
            k = k+1

        l = k + 1

        while (x <= y):
            a[k] = x
            a[l] = y
            yield a[:k+2]
            x = x+1
            y = y-1

        a[k] = x+y
        y = x+y-1

        yield(a[:k+1])


def partitions(n, method=None):
    '''Generate all partitions of n (wrapping function)'''
    if method == "slow":
        p = ip_slow(n)
    else:
        p = set(tuple(x) for x in ip_fast(n))

    p = sorted([list(x) for x in p])
    return(p)


def padding_partition(partition, length):
    '''Padding a partition with zeros to have the prescribed length.'''
    if (len(partition) <= length):
        partition.extend([0] * (length - len(partition)))
    return(partition)


def permutations(partitions):
    '''Permutations of partitions.'''
    # The partition determines how we split up the number of particles, while
    # permutation gives on which site we put down these portion of particles.
    all_permutations = []
    for p in partitions:
        for element in itertools.permutations(p):
            if element not in all_permutations:
                all_permutations.append(element)

    all_permutations = sorted(all_permutations, reverse=True)
    return(all_permutations)


def unique_permutations(seq):
    '''Unique permutations of seq in an efficient way (Knuth "Algorithm L").'''
    # For speed: precalculate the indices to iterate over
    i_indices = range(len(seq) - 1, -1, -1)
    k_indices = i_indices[1:]

    # Start with the sorted sequence
    seq = sorted(seq)

    while True:
        yield(tuple(seq))
        # Working backwards from the last-but-one index,           k
        # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # The else branch is executed only if the break statement was never
            # reached, thus seq is weakly decreasing, and we're done.
            return

        # Get item from sequence only once, for speed
        k_val = seq[k]

        # Working backwards starting with the last item,           k     i
        # find the first one greater than the one at k       0 0 1 0 1 1 1 0
        for i in i_indices:
            if k_val < seq[i]:
                break

        # Swap them                                    k     i
        (seq[k], seq[i]) = (seq[i], seq[k])    # 0 0 1 1 1 1 0 0

        # Reverse the part after but not               k
        # including k, also efficiently.         0 0 1 1 0 0 1 1
        seq[k+1:] = seq[-1 : k : -1]


def states_generator(parameters, report=False, method=None):
    '''Number states for a given geometry and particle number.'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]
    number_of_sites = nx * ny
    number_of_particles = parameters["Total number of particles"]

    all_partitions = partitions(number_of_particles)
    if report:
        parameters["Number of partitions"] = len(all_partitions)
        print("Number of partitions = ", len(all_partitions))
        print(all_partitions)

    # Padding partitions to the length required by number_of_sites
    # Padding or dropping partitions whose length differs from the given length.
    fixed_partitions = []
    for partition in all_partitions:
        if (len(partition) <= number_of_sites):
            fixed_partitions.append(padding_partition(partition, number_of_sites))

    if report:
        print("Number of fixed partitions = ", len(fixed_partitions))
        print(fixed_partitions)

    if (method == "slow"):
        states = permutations(fixed_partitions)
    else:
        states = []
        for partition in fixed_partitions:
            for p in unique_permutations(partition):
                states.append(p)

    states = sorted(states)
    parameters["Number of number-states"] = len(states)

    if report:
        print("Number of possible states = ", len(states))

    return(states)


def site2index(nx, ny):
    '''Sites are labelled either by tuple (i,j) or by index.'''
    pairs = [(j, i) for j in range(ny) for i in range(nx)]
    mapping = {}
    for (k, pair) in enumerate(pairs):
        mapping[pair] = k

    return(mapping)


def index2site(nx, ny):
    '''Sites are labelled either by tuple (i,j) or by index.'''
    mapping = site2index(nx, ny)
    _invmap = {val:key for (key, val) in mapping.items()}
    mapping = _invmap
    return(mapping)


def state_index(all_states, state, default=None):
    '''Find the index of state in all_states, or return default.'''
    try:
        index = all_states.index(state)
    except ValueError:
        index = default
    return(index)


def index_pairs(parameters):
    '''Index pairs for the a_{i}^{\dag}a_{j} operator product'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]

    in_chain = []
    between_chains = []
    for i in range(ny):
        for j in range(nx):
            # Left pairs
            if (j > 0):
                in_chain.append(((i, j), (i, j-1)))
            # Right pairs
            if (j < nx-1):
                in_chain.append(((i, j), (i, j+1)))

            # Above pairs
            if (i > 0):
                between_chains.append(((i, j), (i-1, j)))
            # Below pairs
            if (i < ny-1):
                between_chains.append(((i, j), (i+1, j)))

    mapping = site2index(nx, ny)
    in_chain = [(mapping[x], mapping[y]) for (x, y) in in_chain]
    between_chains = [(mapping[x], mapping[y]) for (x, y) in between_chains]

    parameters.update({"Links within chains": in_chain,
                       "Links between chains": between_chains})
    return


def prune_between_chains(parameters):
    '''Drop some of the between-chain linkages'''
    lattice_type = parameters["Connections between chains"].lower()
    if (lattice_type == "full"):
        pass
    else:
        nx = parameters["Number of sites per chain"]
        ny = parameters["Number of chains"]
        kept = []

        if (lattice_type == "manual"):
            manual = set(parameters["Prescribed links between chains"])
            tmp = list(manual)
            tmp.extend([(link[1], link[0]) for link in manual])
            tmp = list(tmp)

        if (lattice_type == "random_one"):
            tmp = []
            for row in range(ny-1):
                rnd_col = random.randint(0, nx-1)
                link = ((row, rnd_col), (row+1, rnd_col))
                # Link added symmetrically, so that it results in symmetric H
                tmp.extend([link, (link[1], link[0])])

        # Convert all links back to index notation
        all_links = site2index(nx, ny)
        for (s1, s2) in tmp:
            (s1, s2) = (all_links[s1], all_links[s2])
            kept.append((s1, s2))

        parameters["Links between chains"] = kept
    return


def destroy(index, state):
    '''Destruction operator on state'''
    prefactor = 0
    if state:
        state = list(state)
        if (state[index] == 0):
            prefactor = 0
            state = None
        else:
            prefactor = np.sqrt(state[index])
            state[index] = state[index]-1
            state = tuple(state)

    return(prefactor, state)


def create(index, state, maxparticle):
    '''Construction operator on state'''
    prefactor = 0
    if state:
        state = list(state)
        if (state[index] == maxparticle):
            prefactor = 0
            state = None
        else:
            prefactor = np.sqrt(state[index]+1)
            state[index] = state[index]+1
            state = tuple(state)

    return(prefactor, state)


def inner_product(s1, s2):
    '''Inner product of two number states'''
    if s1 == s2:
        result = 1
    else:
        result = 0
    return(result)


def normalise_vector(v, expected_norm=1):
    '''Normalize a vector to a prescribed value'''
    current_norm = np.linalg.norm(v)
    v = (expected_norm/current_norm) * v
    return(v)


def onsite_hamiltonian(parameters):
    '''Calculate H = Sum(U_{i}/2 n_{i} (n_{i}-1)'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]
    U = parameters["Onsite interaction strength (U)"]
    states = parameters["List of number states"]

    # Empty Hamiltonian matrix with the correct size
    H = sp.dok_matrix((len(states), len(states)), dtype=np.float32)

    # For fast lookup-table store states and their index in a dictionary
    tmp = dict(zip(states, range(len(states))))

    # All terms are contributing by the half of the onsite interaction
    Uhalf = U / 2.0

    # Loop over all possible sites
    for site in range(nx*ny):
        # Loop over all states
        for (i, state) in enumerate(states):
            n = state[site]
            # If site has more than 1 particle, then H[site, site] is non-zero
            if n > 1:
                H[i, i] = H[i, i] + Uhalf * n * (n-1)

    return(H)


def hopping_hamiltonian(parameters):
    '''Calculate H = Sum(J_{i,j,k,l} a^{\dag}_{i, j} a_{k,l})'''
    P = parameters["Total number of particles"]
    J = parameters["Coupling within chain (J)"]
    Jprime = parameters["Coupling between chains (J')"]
    states = parameters["List of number states"]

    n = len(states)
    # Empty Hamiltonian matrix with the correct size
    H = sp.dok_matrix((n, n), dtype=np.float32)

    # For fast lookup-table store states and their index in a dictionary
    tmp = dict(zip(states, range(n)))

    # Loop over all possible operator pairing within a chain
    for (source, target) in parameters["Links within chains"]:

        # Loop over all states
        for (i, state) in enumerate(states):
            # Act on state with the destroy operator
            (norm1, state) = destroy(source, state)

            # Act on the new state with the creation operator
            (norm2, state) = create(target, state, P)

            # Find the resulting state in the list of states
            j = tmp.setdefault(state, None)

            # If the resulting state can be found the Hamiltonian is updated.
            if j is not None:
                H[i, j] = H[i, j] + J * norm1 * norm2

    # If needed some between-chain linkages are elminiated
    lattice_type = parameters["Connections between chains"].lower()
    if (lattice_type != "full"):
        prune_between_chains(parameters)

    # Loop over all possible operator pairing between chains
    for (source, target) in parameters["Links between chains"]:
        # Loop over all states
        for (i, state) in enumerate(states):
            # Act on state with the destroy operator
            (norm1, state) = destroy(source, state)

            # Act on the new state with the creation operator
            (norm2, state) = create(target, state, P)

            # Find the resulting state in the list of states
            j = tmp.setdefault(state, None)

            # If the resulting state can be found the Hamiltonian is updated.
            if j is not None:
                H[i, j] = H[i, j] + Jprime * norm1 * norm2

    return(H)


def calculate_spectrum(hamiltonian, method="dense"):
    '''Spectrum of the system.'''
    (nrow, _) = np.shape(hamiltonian)

    # Above a fixed matrix size sparse algorithm is enforced
    if (nrow > 13000):
        method = "sparse"

    if (method == "sparse"):
        print("!WARNING: all but one eigenvalues are determined.")
        (eigvals, eigvects) = splin.eigsh(hamiltonian, k=nrow-1)
    else:
        # In the earlier code only the eigenvalues were calculated:
        # eigvals = np.linalg.eigvalsh(hamiltonian.todense())
        (eigvals, eigvects) = np.linalg.eigh(hamiltonian.todense())

    spectrum = []
    for (k, eigvalk) in enumerate(eigvals):
        # spectrum[eigvalk] = np.reshape(eigvects[:, k], (1, nrow))
        spectrum.append((eigvalk, np.asarray(eigvects[:, k]).flatten()))

    spectrum = sorted(spectrum, key=lambda x: x[0])
    return(spectrum)


def calculate_groundstate(hamiltonian, parameters):
    '''Calculate only the smallest eigen-energy and the corresponding state.'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]
    number_of_sites = nx * ny

    (ground_energy, weights) = splin.eigsh(hamiltonian, k=1, which="SA",
                                           return_eigenvectors=True)
    ground_energy = np.real(ground_energy[0])

    ground_state = np.zeros(number_of_sites, dtype=np.complex)
    for (k, weight) in enumerate(weights):
        number_state = np.asarray(parameters["List of number states"][k])
        ground_state += weight * number_state

    return(ground_energy, ground_state, weights)


def E2s(spectrum):
    '''Overlap of the two sets of eigenstates: <energy state|number state>.'''
    n = len(spectrum)
    ckl = np.zeros((n, n))
    for (k, (_, state)) in enumerate(spectrum):
        ckl[ :,k] = np.conjugate(state)
    return(ckl)


def s2E(energy_states):
    '''Overlap of the two sets of eigenstates: <number state|energy state>.'''
    ckl = E2s(energy_states)
    clk = np.conjugate(np.transpose(ckl))
    return(clk)

"""This is the old version of bb that calculated the off-diagonal terms"""
#def bb(parameters):
#    '''Expectation value of a_{i}^{+} * a_{i'} between two number states'''
#    # The construction of matrix B_{pq} is very similar to that of the hopping
#    # Hamiltonian, however, indices i and i' can correspond to arbitrary sites,
#    # and not necessarily neighbouring sites.
#    P = parameters["Total number of particles"]
#    nx = parameters["Number of sites per chain"]
#    ny = parameters["Number of chains"]
#    states = parameters["List of number states"]
#
#    site_indices = range(nx*ny)
#
#    # Empty Hamiltonian matrix with the correct size
#    L = len(states)
#    B = sp.dok_matrix((L, L), dtype=np.complex)
#
#    # For fast lookup-table store states and their index in a dictionary
#    tmp = dict(zip(states, range(L)))
#
#    def _double_sum(state_p, state_q, k):
#        '''The double sum on i and iprime'''
#        double_sum = 0
#        for site_i in site_indices:
#            for site_ip in site_indices:
#                (norm1, tmp_state) = destroy(site_ip, state_q)
#                (norm2, tmp_state) = create(site_i, tmp_state, P)
#
#                if tmp_state is not None:
#                    bracket = norm1 * norm2 * inner_product(state_p, tmp_state)
#                    exponent = -1j * 2 * np.pi * k * (site_i - site_ip) / L
#                    double_sum = double_sum  + np.exp(exponent) * bracket
#
#        return(double_sum)
#
#    # Assemble matrix B_{pq} for all pairs of states (p, q)
#    momentum = 0
#    for (p, state_p) in enumerate(states):
#        for (q, state_q) in enumerate(states):
#            B[p, q] = _double_sum(state_p, state_q, momentum)
#
#    return(B)

# Just calculating diagonal terms here -----------------------------------------
def bb(parameters):
    '''Expectation value of a_{i}^{+} * a_{i} between two identical number
    states'''
    # The construction of matrix B_{pq} is very similar to that of the hopping
    # Hamiltonian, however, indices i and i' can correspond to arbitrary sites,
    # and not necessarily neighbouring sites.
    P = parameters["Total number of particles"]
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]
    states = parameters["List of number states"]

    site_indices = range(nx*ny)

    # Empty Hamiltonian matrix with the correct size
    L = len(states)
    B = sp.dok_matrix((L, L), dtype=np.complex)

    # For fast lookup-table store states and their index in a dictionary
    tmp = dict(zip(states, range(L)))

    def _single_sum(state_p, state_q, k):
        '''The single sum (since i = iprime)'''
        single_sum = 0
        for site_i in site_indices:
            for site_ip in site_indices:
                if site_i == site_ip:
                    (norm1, tmp_state) = destroy(site_ip, state_q)
                    (norm2, tmp_state) = create(site_i, tmp_state, P)

                    if tmp_state is not None:
                        bracket = norm1 * norm2 * inner_product(state_p, tmp_state)
                        exponent = -1j * 2 * np.pi * k * (site_i - site_ip) / L
                        single_sum = single_sum + np.exp(exponent) * bracket

        return(single_sum)

    # Assemble matrix B_{pq} for all pairs of states (p, q)
    momentum = 0
    for (p, state_p) in enumerate(states):
        for (q, state_q) in enumerate(states):
            B[p, q] = _single_sum(state_p, state_q, momentum)

    return(B)


def update_d(d, spectrum, t):
    '''Calculate d 'exactly' at time t'''
    c = E2s(spectrum)

    # Calculating the time-dependent exponential term
    energy = np.array([energy for (energy, _) in spectrum])
    e = np.exp(1j * energy * t)

    # This is where the "magic" happens.
    dc = np.dot(d, c)
    ce = np.multiply(np.conj(c), e)
    d = np.dot(ce, np.transpose(dc))
    return(d)


def evolve(d0, spectrum, parameters):
    '''Evolve the initial_state according to the entire spectrum'''
    tmin = parameters["Time starts"]
    tmax = parameters["Time ends"]
    tn = parameters["Number of time steps"]
    block_length = parameters["Maximal block length in export"]

    # Setting up the data files
    qmio.create_datafiles(parameters)

    # Time-step
    dt = (tmax - tmin)/(tn-1)

    #time vector for plotting
    times = np.linspace(tmin, tmax, tn)

    # Auxiliary variables
    c = E2s(spectrum)

    # Calculating the time-dependent exponential term
    energy = np.array([energy for (energy, _) in spectrum])

    # Auxiliary, constant matrices
    d0c = np.dot(d0, c)
    dcc = np.multiply(d0c, np.conj(c))
    BB = bb(parameters)

    t = tmin
    for cycle in range(tn // block_length):
        t = tmin + np.arange(block_length) * dt
        e = np.exp(1j * np.outer(energy, t))
        e = np.asmatrix(e)

        # Evolve the expansion coefficients d_k(t)
        d = np.transpose(dcc * e)

        # Calculating the number density
        n = qmmeas.occupation(d, parameters)

        # Calculate Fourier transforms if requested
        fkR = None
        if "fkRigol" in parameters["Exported quantities"]:
            fkR = qmmeas.ft_Rigol(d, BB, parameters)

        fk = None
        if "fk" in parameters["Exported quantities"]:
            fk = qmmeas.ft(n, parameters)

        # Export simulation data
        all_data = {"t": t, "d": d, "n": n, "fkRigol": fkR, "fk": fk}
        qmio.export_data(all_data, parameters)

        # Shift the origin of time for the next block
        tmin = t[-1] + dt

    # If block_length does not divide tn, then we create the last block manually
    missing = tn - (tn//block_length) * block_length
    cycle = cycle + 1
    if missing:
        t = tmin + np.arange(missing) * dt

        e = np.exp(1j * np.outer(energy, t))
        e = np.asmatrix(e)

        # Evolve the expansion coefficients d_k(t)
        d = np.transpose(dcc * e)

        # Calculating the number density
        n = qmmeas.occupation(d, parameters)

        # Calculate Fourier transforms if requested
        if "fkRigol" in parameters["Exported quantities"]:
            fkR = qmmeas.ft_Rigol(d, BB, parameters)

        if "fk" in parameters["Exported quantities"]:
            fk = qmmeas.ft(n, parameters)

        # Export simulation data
        all_data = {"t": t, "d": d, "n": n, "fkRigol": fkR, "fk": fk}
        qmio.export_data(all_data, parameters)

    return

# ==============================================================================
if __name__ == "__main__":
    pass
