#################################################
# TATR
# Julie Butler Hartley
# Version 1.0.0
# Date Created: July 23, 2021
# Last Modified: July 28, 2021
#
# Creates a T-Amplitude Tensor Representation from a given list of MBPT or CC
# amplitudes.  Based on the work found here: 
# https://pubs.acs.org/doi/10.1021/acs.jpca.8b04455
#################################################


#############################
# IMPORTS
#############################
# THIRD-PARTY IMPORTS
# for exponential
import math
# For arange and zeros
import numpy as np


#############################
# GAUSSIAN
#############################
def gaussian (x, mu, sigma):
	"""
		Inputs:
			x (a float): the x value to find the value of the gaussian at
			mu (a float): the mean of the gaussian (i.e. the x value of its 
				center)
			sigma  (a float): the standard deviation of the gaussian
		Returns:
			Unnamed (a float: the value of a gaussian at point x, centered at 
				mu and with a standard deviation of sigma
		Computes the value of a gaussian at the point x.  The gaussian has a
		mean/center at mu and a standard deviation of sigma.
	"""
    piece1 = (x-mu)**2
    piece2 = 2*sigma**2
    return math.exp(-piece1/piece2)


#############################
# G
#############################
def g (dx, t, sigma):
	"""
		Inputs:
			dx
			t
			sigma
	"""
	 (Eq. 5)
    x = np.arange(-1, 1+dx, dx)
    v = []
    for i in x:
        v.append(gaussian(i, t, sigma))
    return np.asarray(v)

#############################
# TATR
#############################
def TATR (dx, amps, sigma):
    tatr = np.zeros(len(np.arange(-1, 1+dx, dx)))
    for amp in amps:
        tatr += g(dx, amp, sigma)
    return tatr