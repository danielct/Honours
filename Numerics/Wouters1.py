import Simulator
import UtilityFunctions
import pint
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry()

# Simulation parameters
N = 2048
max_XY_unscaled = (8 * ureg.micrometer).to_base_units().magnitude
T_MAX = 100

# GPE Parameters from Wouters
hbar = ureg.hbar.to_base_units()
# Note Wouters denotes the coefficient of |\psi| ^ 2 as \hbar * g. We denote it
# as g.
g_C = (0.015 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_C = g_C.to_base_units().magnitude
g_R = 0
gamma_C = (0.5 * ureg.millielectron_volt) / hbar
gamma_C = gamma_C.to_base_units().magnitude
gamma_R = 4.0 * gamma_C
R = (0.05 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer) / hbar
R = R.to_base_units().magnitude
# Wouters does not provide m. Let us assume it is 10e-4 m_e
m = 10e-4

woutersParams = Simulator.ParameterContainer(g_C=g_C, g_R=g_R, gamma_C=gamma_C,
                                             gamma_R=gamma_R, R=R, m=m)
params = woutersParams.getGPEParams()

# Pump parameters for small and large spots from Wouters
# P0 are both in units of Pth
P0_large = 2.0
P0_small = 8.0
sigma_large = (20.0 * ureg.micrometer).to_base_units().magnitude
sigma_small = (2.0 * ureg.micrometer).to_base_units().magnitude

# Make pumps
largePump = UtilityFunctions.GaussianPump(sigma_large, P0_large,
                                          params['Pth'])
largePumpFunction = largePump.scaledFunction(woutersParams.charL)
smallPump = UtilityFunctions.GaussianPump(sigma_small, P0_small,
                                          params['Pth'])
smallPumpFunction = smallPump.scaledFunction(woutersParams.charL)

# Make Grid
grid = Simulator.Grid(woutersParams.charL, max_XY=max_XY_unscaled, N=N)
# solver = Simulator.GPESolver(grid.getSpatialGrid(), grid.getkGrid(),
#                              N_THREADS=1)
