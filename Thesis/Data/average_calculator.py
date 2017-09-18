# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:38:49 2017

@author: mitni349
"""

import numpy as np
import qmlattice_utils as qm


(_,data) = qm.read_evolve_file("3by1_U0.5_T2e4.dat", cols=[0,2])
#(_,data) = qm.read_evolve_file("c.dat", [0,9])

average=np.mean(data)
print(average)
