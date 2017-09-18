# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:38:49 2017

@author: mitni349
"""

import numpy as np
import qmlattice_utils_messing_around as qm


(time,data1,data2,data3,data4,data5,data6,data7,data8) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,1,2,3,4,5,6,7,8])

#(_,data) = qm.read_evolve_file("c.dat", [0,9])

average1=np.mean(data1)
print("average1="+average1)
average2=np.mean(data2)
print("average2="+average2)
average3=np.mean(data3)
print("average3="+average3)
average4=np.mean(data4)
print("average4="+average4)
average5=np.mean(data5)
print("average5="+average5)
average6=np.mean(data6)
print("average6="+average6)
average7=np.mean(data7)
print("average7="+average7)
average8=np.mean(data8)
print("average8="+average8)
#average9=np.mean(data9)
#print("average9="+average9)

