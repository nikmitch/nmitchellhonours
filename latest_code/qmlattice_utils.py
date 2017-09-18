#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Standard libraries
import os
import time

import random
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin

# Complementary, and nearly standard libraries
import matplotlib.pyplot as plt
# import matplotlib as mpl


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


def E2s(spectrum):
    '''Overlap of the two sets of eigenstates: <energy state|number state>.'''
    n = len(spectrum)

    ckl = np.zeros((n, n))
    for (k, (_, state)) in enumerate(spectrum):
        ckl[k, :] = np.conjugate(state)

    return(ckl)


def s2E(energy_states):
    '''Overlap of the two sets of eigenstates: <number state|energy state>.'''
    ckl = E2s(energy_states)
    clk = np.conjugate(np.transpose(ckl))
    return(clk)


def export_header(fname, parameters):
    '''Create data file and export simulation parameters as header.'''
    # Full path for the datafile
    if "Timestamp" not in parameters:
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        parameters.update({"Timestamp": timestamp})

    directory = os.getcwd()
    timestamp = parameters["Timestamp"]
    filename = "".join([timestamp, "_", fname, ".dat"])
    datafile = os.path.normpath("".join([directory, "/", filename]))

    with open(datafile, "a+") as fid:
        for p in sorted(parameters):
            fid.write("# {0:<32s}: {1:>s}\n".format(p, str(parameters[p])))

        fid.write("# " + "-" * 44 + "\n")

    return(datafile)


def export_spectrum(parameters, spectrum):
    '''Save the spectrum to a simple data file.'''
    fname = export_header("spectrum", parameters)

    with open(fname, "a+") as fid:
        fid.write("#{0: >5s}    {1:>14s}\n".format("i", "energy"))
        fid.write("# " + "-" * 44 + "\n")

        for (k, (energy, _)) in enumerate(spectrum):
            fid.write("{0: >6d}    {1:>14.10f}\n".format(k, energy))

        fid.write("\n"*2)
    return


def export_block(fname, time, block, parameters):
    '''Export a block of expansion coefficients into file.'''
    with open(fname, "a") as fid:
        (nrow, ncol) = np.shape(block)
        # Export time and real data line by line.
        for n in range(nrow):
            fid.write("{0:>18.10f}".format(time[n]))
            for k in range(ncol):
                fid.write("{0:>16.10f}".format(block[n, k]))
            fid.write("\n")

    return(fname)


def read_evolve_file(fname, cols):
    '''Read the data file exported from the evolve function.'''
    try:
        tmp = np.loadtxt(fname, usecols=cols, comments="#")
    except IOError:
        print("!WARNING: reading file has failed {0:>s}".format(fname))

    if len(tmp):
        time = tmp[:,0]
        d = tmp[:, 1:]

    return((time, d))


def update_d(d, spectrum, t):
    '''Calculate d 'exactly' at time t'''
    c = E2s(spectrum)

    # Calculating the time-dependent exponential term
    energy = np.array([energy for (energy, _) in spectrum])
    e = np.exp(1j * energy * t)

    # This is where the magic happens.
    dc = np.dot(d, c)
    ce = np.multiply(np.conj(c), e)
    d = np.dot(ce, np.transpose(dc))
    return(d)


def evolve(d0, spectrum, parameters):
    '''Evolve the initial_state according to the entire spectrum'''
    tmin = parameters["Time starts"]
    tmax = parameters["Time ends"]
    tn = parameters["Number of time steps"]

    # Time-step
    dt = (tmax - tmin)/(tn-1)

    # Auxiliary variables
    c = E2s(spectrum)

    # Calculating the time-dependent exponential term
    energy = np.array([energy for (energy, _) in spectrum])
    e = np.exp(1j * energy * dt)

    # The matrix ce is independent of d and time, thus can be pre-calculated.
    ce = np.multiply(np.conj(c), e)

    # Time evolution and export in blocks
    fname_d = export_header("evolution_d", parameters)
    fname_n = export_header("evolution_n", parameters)

    block_length = parameters["Maximal block length in export"]
    block = np.zeros((block_length+1, len(d0)), dtype=np.complex128)
    d = block[0, :] = update_d(d0, spectrum, tmin)

    t = np.zeros(block_length+1, dtype=np.float64)
    t[0] = tmin

    counter = 1
    for k in range(1, tn):
        dc = np.dot(d, c)
        d = np.dot(ce, np.transpose(dc))
        block[counter, :] = d
        t[counter] = tmin + k * dt

        if (counter > 0) and (counter % block_length == 0):
            d2 = np.square(np.abs(block))
            n_exp = occupation(d2, parameters)
            fname_d = export_block(fname_d, t, d2, parameters)
            fname_n = export_block(fname_n, t, n_exp, parameters)
            counter = 0

            # Due to the iterative nature of this algorithm from time to time,
            # the expansion coefficients are updated in an "exact" manner.
            d = update_d(d0, spectrum, tmin + k*dt)
        else:
            counter = counter + 1

    # Saving the last block
    block = block[0:counter, :]
    block = np.square(np.abs(block))
    n_exp = occupation(block, parameters)
    fname_d = export_block(fname_d, t, block, parameters)
    fname_n = export_block(fname_n, t, n_exp, parameters)

    return


def occupation(d2, parameters):
    '''Occupation of site.'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]
    number_of_sites = nx * ny

    number_states = parameters["List of number states"]

    (nrow, ncol) = np.shape(d2)
    nexp = np.zeros((nrow, number_of_sites), dtype=np.float32)
    for site in range(number_of_sites):
        nj = np.array([state[site] for state in number_states])
        for n in range(nrow):
            nexp[n, site] = np.dot(d2[n, :], nj)

    return(nexp)


def measure_Ehopp(d2, parameters):
    '''Instantaneous value of the hopping energy, i.e, <Psi| H_hopp | Psi>'''
    return



def plot_lattice(parameters):
    '''Plot the lattice and all connections'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]

    mapping = site2index(nx, ny)
    inverse = {v: k for (k, v) in mapping.items()}

    # Creating the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Links between the chains
    for (site1, site2) in parameters["Links between chains"]:
        # The strange order lies in the numbering scheme we settled on
        (y1, x1) = inverse[site1]
        (y2, x2) = inverse[site2]

        # Shift is necessary as we start numbering from the top chain
        y1 = ny - (y1 + 1)
        y2 = ny - (y2 + 1)

        plt.plot([x1, x2], [y1, y2], 'r--', linewidth=2)

    # Links within the chains
    for (site1, site2) in parameters["Links within chains"]:
        # The strange order lies in the numbering scheme we settled on
        (y1, x1) = inverse[site1]
        (y2, x2) = inverse[site2]

        # Shift is necessary as we start numbering from the top chain
        y1 = ny - (y1 + 1)
        y2 = ny - (y2 + 1)

        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

    # Blob for all sites
    for site in mapping:
        (y1, x1) = site
        label = str(mapping[site])

        # Shift is necessary as we start numbering from the top chain
        y1 = ny - (y1 + 1)

        # All sites
        plt.plot(x1, y1, 'bo', markersize=10)
        ax.annotate(label, xy=(x1, y1), xytext=(x1+0.05, y1+0.05), size=15)

    # Global settings for the graph
    plt.grid(True)                                 # Turn grid on
    plt.setp(ax.get_xticklabels(), visible=False)  # Turn xticks and labels off
    plt.setp(ax.get_yticklabels(), visible=False)  # Turn yticks and labels off
    plt.xlim([-0.5, nx-0.5])                       # Giving some margin in x
    plt.ylim([-0.5, ny-0.5])                       # Giving some margin in y
    plt.close(fig)                                 # Close figure, so no wait

    # Full path of the graphics file
    directory = os.getcwd()
    timestamp = parameters["Timestamp"]
    figfname = "".join([timestamp, "_geometry.png"])
    figfname = os.path.normpath("".join([directory, "/", figfname]))

    # Save figure
    fig.savefig(figfname, paper="a4", orientation="landscape", dpi=300,
                bbox_inches="tight")
    return

# ==============================================================================
if __name__ == "__main__":
    pass
