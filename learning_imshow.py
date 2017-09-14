# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:05:31 2017

@author: mitni349
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import eigh

A=np.matrix([[ 0 -1  0 -1  0  0  0  0  0]
 [-1  0 -1  0 -1  0  0  0  0]
 [ 0 -1  0  0  0 -1  0  0  0]
 [-1  0  0  0 -1  0 -1  0  0]
 [ 0 -1  0 -1  0 -1  0 -1  0]
 [ 0  0 -1  0 -1  0  0  0 -1]
 [ 0  0  0 -1  0  0  0 -1  0]
 [ 0  0  0  0 -1  0 -1  0 -1]
 [ 0  0  0  0  0 -1  0 -1  0]])

(w,v)=eigh(A)
print(w)