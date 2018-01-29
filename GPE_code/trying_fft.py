# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:37:20 2017

@author: mitni349
"""

import numpy as np

v=np.array([2,3,4,5+5j])
w=np.fft.fft(v)
print(w)
