#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Finding the "near miss" times for a sum of periodic functions
'''

import mpmath as mp

def find_tstar(func, epsilon, dt=mp.mpf("1e-4"), tmin=mp.mpf("0.1"), tmax=mp.mpf("1e15")):
    '''Minimal t* value for which func approaches zero closer than epsilon.'''
    tstar = None

    t = tmin
    while (t < tmax):
        if (func(t) < epsilon):
            tstar = t
            break
        else:
            t = t + dt

    return(tstar)


def psi(t, eigenvalues, coefficients, f0=mp.mpf("1.0")):
    '''The nearly periodic function'''
    f = None

    if len(eigenvalues) == len(coefficients):
        f = [mp.expj(E*t) for E in eigenvalues]
        f = mp.fdot(coefficients, f)
        f = mp.fabs(f - f0)

    return(f)


# ******************************************************************************
# *** MAIN
# ***
# *** EXAMPLE:
# *** For a non-interacting system with five lattice sites:
# *** EIGENVALUES  = [-mp.sqrt(3), mp.mpf("-1"), mp.mpf("0"), mp.mpf("1"), mp.sqrt(3)]
# *** COEFFICIENTS = [mp.mpf("1/12"), mp.mpf("1/ 4"), mp.mpf("1/ 3"), mp.mpf("1/ 4"), mp.mpf("1/12")]
# ***
# *** RESULT:
# *** epsilon = 1.0e-1, tstar = 18.2104999999709, f(tstar) = 0.0999829666714949
# *** epsilon = 1.0e-2, tstar = 25.2064999999546, f(tstar) = 0.0099972051550079
# *** epsilon = 1.0e-3, tstar = 94.2547000020054, f(tstar) = 0.00099906001272887
# *** epsilon = 1.0e-4, tstar = 351.857199983313, f(tstar) = 9.91072890377254e-5
# *** epsilon = 1.0e-5, tstar =   1313.184599742, f(tstar) = 9.74361845684779e-6
# *** epsilon = 1.0e-6, tstar = 4900.88400532812, f(tstar) = 9.51986134034577e-7
# *** epsilon = 1.0e-7, tstar = 18290.3522752436, f(tstar) = 7.18960744272579e-8
# *** epsilon = 1.0e-8, tstar = 49970.1726016378, f(tstar) = 8.77679739996751e-9
# *** epsilon = 1.0e-9, tstar = 186491.223053476, f(tstar) = 7.1541417234755e-10
# ******************************************************************************
if (__name__ == "__main__"):

    mp.dps = 10
    mp.pretty = True

    # Eigenvalues
    EIGENVALUES  = [-mp.sqrt(3),
                     mp.mpf("-1"),
                     mp.mpf("0"),
                     mp.mpf("1"),
                     mp.sqrt(3)]
    COEFFICIENTS = [mp.mpf("1/12"),
                    mp.mpf("1/ 4"),
                    mp.mpf("1/ 3"),
                    mp.mpf("1/ 4"),
                    mp.mpf("1/12")]


    fmt   = "epsilon = {0:>10s},   tstar = {1:>20s}     f(tstar) = {2:>20s}"
    func = lambda t: psi(t, EIGENVALUES, COEFFICIENTS)

    TSTAR = []
    for exponent in sorted(range(-9,0), reverse=True):
        epsilon = mp.mpf("1E{0:<-1d}".format(exponent))
        tstar = find_tstar(func, epsilon, tmin=mp.mpf("0.5"))
        TSTAR.append((epsilon, tstar, func(tstar)))
        print(fmt.format(str(epsilon), str(tstar), str(func(tstar))))

    print(TSTAR)
