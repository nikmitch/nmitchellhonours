# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:38:49 2017

@author: mitni349
"""

import numpy as np
import qmlattice_utils as qm

probsmat=np.zeros((3,3))

(_,data1) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,1])
probsmat[0,0]=np.mean(data1)
print(probsmat)


(_,data2) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,2])
probsmat[0,1]=np.mean(data2)
print(probsmat)

(_,data3) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,3])
probsmat[0,2]=np.mean(data3)
print(probsmat)

(_,data4) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,4])
probsmat[1,0]=np.mean(data4)
print(probsmat)

(_,data5) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,5])
probsmat[1,1]=np.mean(data5)
print(probsmat)

(_,data6) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,6])
probsmat[1,2]=np.mean(data6)
print(probsmat)

(_,data7) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,7])
probsmat[2,0]=np.mean(data7)
print(probsmat)

(_,data8) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,8])
probsmat[2,1]=np.mean(data8)
print(probsmat)

(_,data9) = qm.read_evolve_file("4by4_U0.1_T1e6dt1.dat", cols=[0,9])
probsmat[2,2]=np.mean(data9)
print(probsmat)




#(_,data1) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,1])
#average1=np.mean(data1)
#print("average1="+average1)
#
#(_,data2) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,2])
#average2=np.mean(data2)
#print("average2="+average2)
#
#(_,data3) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,3])
#average3=np.mean(data3)
#print("average3="+average3)
#
#(_,data4) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,4])
#average4=np.mean(data4)
#print("average4="+average4)
#
#(_,data5) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,5])
#average5=np.mean(data5)
#print("average5="+average5)
#
#(_,data6) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,6])
#average6=np.mean(data6)
#print("average6="+average6)
#
#(_,data7) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,7])
#average7=np.mean(data7)
#print("average7="+average7)
#
#(_,data8) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,8])
#average8=np.mean(data8)
#print("average8="+average8)
#
#(_,data9) = qm.read_evolve_file("3by3_U0.1_T1e4.dat", cols=[0,9])
#average9=np.mean(data9)
#print("average9="+average9)