import SimulatorOld as Simulator
from UtilityFunctions import GaussianPump
import numpy as np
# import UtilityFunctions
# oldParams = {'g_C': 0.187, 'g_R': 2.0 * 0.187, 'gamma_C': 1.0, 'gamma_R': 1.5,
#              'R': 0.1, 'Pth': 15.0}
oldParams = {'g_C': 0.187, 'g_R': 2*0.187, 'gamma_C': 1.0, 'gamma_R': 1.5,
             'R': 0.1}
oldParams['Pth'] = oldParams['gamma_C'] * oldParams['gamma_R'] / oldParams['R']
N = 1024
max_XY = 100
sig = 15.0
T_MAX = 50
grid = Simulator.Grid(1, N=1024, max_XY=max_XY)
print "Starting solver"
solver = Simulator.GPESolver(grid.getSpatialGrid(), grid.getkGrid(),
                             N_THREADS=3)
print "Making pump"
#
# pumpFunction = lambda x, y: oldParams['Pth'] * 4.0
# pumpFunction = lambda x, y: 15.0
#P0 = oldParams['Pth'] * 4.0
# pump = UtilityFunctions.GaussianPump(sig, 1.5 * 200 / (sig**2),
#                                      oldParams['Pth'])
pumpFunction = GaussianPump(4e4, 4.0, oldParams['Pth'],
                            exponent=6).unscaledFunction()
print "Pump done"
# solver.solve(oldParams, pump.unscaledFunction(), T_MAX=T_MAX)

#Modulational instability. If this parameter is >= 1, we are motionally stable
P0 = pumpFunction(0, 0)
stabParam = (P0 / oldParams['Pth']) / ((oldParams['g_R'] * oldParams['gamma_C'])
                                       / (oldParams['g_C'] *
                                          oldParams['gamma_R']))
print("Stability parameter %f" % stabParam)

# def psiIFunction(x, y):
#     d0 =np.sqrt((P0 - oldParams['Pth']) / oldParams['gamma_C'])
#     return  d0 + np.random.normal(loc=d0, scale = 1e-1*d0)
np.random.seed(10)
def psiIFunction(x, y):
    return 1e-3 * np.random.normal()
