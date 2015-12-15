import Simulator
import UtilityFunctions
import pint
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry()

# Simulation parameters
N = 1 * 1024
max_XY_unscaled = (25 * ureg.micrometer).to_base_units().magnitude
T_MAX = 100

# GPE Parameters from Billiard paper
hbar = ureg.hbar.to_base_units()
g_C = (2e-3 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_C = g_C.to_base_units().magnitude
g_R = 2.0 * g_C
gamma_C = (0.1 / ureg.picosecond)
gamma_C = gamma_C.to_base_units().magnitude
gamma_R = 10 * gamma_C
R = (6e-4 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer) / hbar
R = R.to_base_units().magnitude
# Wouters does not provide m. Let us assume it is 10e-4 m_e
m = 5e-5

billiardParams = Simulator.ParameterContainer(g_C=g_C, g_R=g_R, gamma_C=gamma_C,
                                              gamma_R=gamma_R, R=R, m=m)
params = billiardParams.getGPEParams()


# Make pumps
annularPump = UtilityFunctions.AnnularPump(10e-6, 5e-6, 1.1 * params['Pth'], 3)
annularPumpFunction = annularPump.scaledFunction(billiardParams.charL)
# Make Grid
grid = Simulator.Grid(billiardParams.charL, max_XY=max_XY_unscaled, N=N)
solver = Simulator.GPESolver(grid.getSpatialGrid(), grid.getkGrid(),
                             N_THREADS=3)
