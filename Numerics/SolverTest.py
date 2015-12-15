import Simulator
import UtilityFunctions
import pint
import matplotlib.pyplot as plt
import numpy as np
import itertools
from UtilityFunctions import diagnosticPlot

ureg = pint.UnitRegistry()

def getPlotName(params):
    outParams = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'charT']
    outStrings = ["=".join([outParam, str(params[outParam])])
                  for outParam in outParams]
    return ", ".join(outStrings)

# Simulation parameters
N = 512
max_XY_unscaled = (100 * ureg.micrometer).to_base_units().magnitude
T_MAX = 30
dt = 1e-3

# GPE Parameters from Billiard Paper
hbar = ureg.hbar.to_base_units()
# Ratio of gamma_R to gamma_C
a = 2.0

g_C = 10 * (2e-3 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_C = g_C.to_base_units().magnitude
g_R = 2.0 * g_C
gamma_C = (0.1 * ureg.picosecond ** -1)
gamma_C = gamma_C.to_base_units().magnitude
gamma_R = a * gamma_C
R = (6e-4 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer) / hbar
R = R.to_base_units().magnitude
m = 5e-5
me = ureg.electron_mass.to_base_units().magnitude

g_Cs = g_C * np.logspace(-1, 1, num=5)
# gamma_Cs = gamma_C * np.logspace(-1, 1, num=3)
Rs = R * np.logspace(-1, 1, num=5)

r = (25 * ureg.micrometer).to_base_units().magnitude
sigma = (5 * ureg.micrometer).to_base_units().magnitude

hbar = hbar.to_base_units().magnitude
charT = 1 / gamma_C
charL = 1 * np.sqrt(hbar / (2 * m * me * gamma_C))

ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=g_R, gamma_C=gamma_C,
                                          gamma_R=gamma_R, R=R, m=m,
                                          charL=charL, charT=charT)
params = ringParams.getGPEParams()

P0 = 4.0 * params['Pth']
pump = UtilityFunctions.AnnularPump(r, P0, sigma)
pumpFunction = pump.scaledFunction(ringParams.charL)
# Make Grid
grid = Simulator.Grid(ringParams.charL, max_XY=max_XY_unscaled, N=N)

x, y = grid.getSpatialGrid()
x_us, y_us = grid.getSpatialGrid(scaled=False)
solver = Simulator.GPESolver(params, dt, grid.getSpatialGrid(),
                             grid.getKSquaredGrid(), pumpFunction,
                             psiInitial=(lambda x, y: 0.1 *
                                         pumpFunction(x, y) / P0),
                             N_THREADS=5, gpu=False)

# solver = SimulatorOld.GPESolver(grid.getSpatialGrid(), grid.getkGrid(),
#                              N_THREADS=4)
P0 = np.max(pumpFunction(x, y))
stabParam = (P0 / params['Pth']) / ((params['g_R'] * params['gamma_C'])
                                    / (params['g_C'] *
                                       params['gamma_R']))
print("Stability parameter %f" % stabParam)
# energyTimes, energy = solver.stepTo(T_MAX, stepsPerObservation=1e3)
solver.stepTo(T_MAX)
energyTimes = np.array([])
energy = energyTimes
# solver.solve(params, pumpFunction,
#              psi0Function=lambda x, y: 0.1 * pumpFunction(x, y) / P0, dt=dt,
#              T_MAX=T_MAX, method='split-step')
#
f = diagnosticPlot(1e6 * x_us, 1e6 * y_us, solver, energyTimes, energy,
                   pump.scaledFunction(charL=1e-6))
# f = diagnosticPlot(1e6 * x_us, 1e6 * y_us, solver, solver.energyTimes,
#                    solver.energy, pump.scaledFunction(charL=1e-6))
f.show()
