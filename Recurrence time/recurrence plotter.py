# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:20:00 2017

@author: mitni349
"""
import numpy as np
import matplotlib.pyplot as plt

#*** epsilon = 1.0e-1, tstar = 18.2104999999709, f(tstar) = 0.0999829666714949
# *** epsilon = 1.0e-2, tstar = 25.2064999999546, f(tstar) = 0.0099972051550079
# *** epsilon = 1.0e-3, tstar = 94.2547000020054, f(tstar) = 0.00099906001272887
# *** epsilon = 1.0e-4, tstar = 351.857199983313, f(tstar) = 9.91072890377254e-5
# *** epsilon = 1.0e-5, tstar =   1313.184599742, f(tstar) = 9.74361845684779e-6
# *** epsilon = 1.0e-6, tstar = 4900.88400532812, f(tstar) = 9.51986134034577e-7
# *** epsilon = 1.0e-7, tstar = 18290.3522752436, f(tstar) = 7.18960744272579e-8
# *** epsilon = 1.0e-8, tstar = 49970.1726016378, f(tstar) = 8.77679739996751e-9
# *** epsilon = 1.0e-9, tstar = 186491.223053476, f(tstar) = 7.1541417234755e-10
#
#
#
tstars=np.array([18.2104999999709, 25.2064999999546,94.2547000020054,351.857199983313, \
1313.184599742, 4900.88400532812,  18290.3522752436, 49970.1726016378, 186491.223053476])


 
epsilons=np.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9])
one_over=1/epsilons



plt.figure()
plt.xlabel("$\log_{10}(1/\epsilon)$")
plt.ylabel("$t*$")
plt.title("Time needed to get arbitrarily close to initial state vector")
plt.plot(np.log10(one_over),tstars,"o")
