# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:38:49 2017

@author: mitni349
"""

import numpy as np
import qmlattice_utils as qm

wholething="10by1_U0.1_T2e4.dat"
(_,data1) = qm.read_evolve_file(wholething, cols=[0,1])
(_,data2) = qm.read_evolve_file(wholething, cols=[0,2])
(_,data3) = qm.read_evolve_file(wholething, cols=[0,3])


average1=np.mean(data1)
average2=np.mean(data2)
average3=np.mean(data3)

averagemat=np.zeros((1,3))
averagemat[0,0]=average1
averagemat[0,1]=average2
averagemat[0,2]=average3

print(wholething+" averages =")
print(averagemat)

