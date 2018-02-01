#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import os
import time

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin

# Non-standard run-time parameters
np.set_printoptions(linewidth=250)

# == FUNCTION DECLARATIONS =====================================================
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


def create_datafiles(parameters):
    '''Create all requested datafiles with headers.'''
    quantities = parameters["Exported quantities"]
    for quantity in quantities:
        fname = "evolution_" + quantity
        quantities[quantity] = export_header(fname, parameters)

    # Update the list of exported quantities with corresponding file handlers.
    parameters["Exported quantities"] = quantities
    return


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


def export_data(data, parameters):
    '''Export those quantities which are requested'''

    if "d2" in parameters["Exported quantities"]:
        fname = parameters["Exported quantities"]["d2"]
        d2 = np.square(np.abs(data["d"]))
        export_block(fname, data["t"], d2, parameters)

    if "n" in parameters["Exported quantities"]:
        fname = parameters["Exported quantities"]["n"]
        export_block(fname, data["t"], data["n"], parameters)

    if "fkRigol" in parameters["Exported quantities"]:
        fname = parameters["Exported quantities"]["fkRigol"]
        export_block(fname, data["t"], data["fkRigol"], parameters)

    if "fk" in parameters["Exported quantities"]:
        fname = parameters["Exported quantities"]["fk"]
        export_block(fname, data["t"], data["fk"], parameters)

    return


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


# ==============================================================================
if __name__ == "__main__":
    pass