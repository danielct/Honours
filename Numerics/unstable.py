from __future__ import division
import Simulator
import UtilityFunctions
import pint
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry()

# Simulation parameters
N = 1024
max_XY_scaled = 80.0
T_MAX = 100

# GPE Parameters from Wouters
hbar = ureg.hbar.to_base_units()
# Note Wouters denotes the coefficient of |\psi| ^ 2 as \hbar * g. We denote it
# as g.

g_C = 1.0
g_R = 2 * g_C
gamma_C = 4.5
gamma_R = 1.0
R = 1.0

Pth = gamma_C * gamma_R / R
P0 = 1.2 * Pth

params = {'g_C': g_C, 'g_R': g_R, 'gamma_C': gamma_C, 'gamma_R': gamma_R,
          'R': R, 'Pth': Pth}

pumpFunction = lambda x, y: P0
# Uniform pump function

# Initial wavefunction - uniform with some noise

psi0Function = lambda x, y: (np.sqrt((P0 - Pth) / gamma_C) +
                             (np.abs(np.random.normal(size=(N, N),
                                                      scale=10e-4, loc=10e-3))
                              + 0.0j*np.random.normal(size=(N, N),
                                                      scale=10e-3, loc=10e-2)))

# Make Grid
grid = Simulator.Grid(1.0, max_XY=max_XY_scaled, N=N)
solver = Simulator.GPESolver(grid.getSpatialGrid(), grid.getkGrid(),
                             N_THREADS=4)
