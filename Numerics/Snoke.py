import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import Simulator
import UtilityFunctions
import pint
import numpy as np
from UtilityFunctions import diagnosticPlot

def getPlotName(params, **kwargs):
    outParams = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R']
    outStrings = (["=".join([outParam, "%.2e" % (params[outParam])])
                  for outParam in outParams] +
                  ["=".join([argName, "%.2e" % (argVal)])
                   for argName, argVal in kwargs.iteritems()])
    return ", ".join(outStrings)

ureg = pint.UnitRegistry()

# Simulation parameters
N = 2 * 512
max_XY_unscaled = (110 * ureg.micrometer).to_base_units().magnitude
T_MAX = 10
dt = 0.1 * 5e-5

a = 2.0
P = 3.0
# GPE Parameters from Wouters
hbar = ureg.hbar.to_base_units()
g_C = (0.88 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_C = g_C.to_base_units().magnitude
g_R = 2.0 * g_C
gamma_Cav = (135 * ureg.picosecond) ** -1
gamma_Cav = gamma_Cav.to_base_units().magnitude
gamma_C = 0.3 * gamma_Cav
gamma_R = 10 * gamma_C
# R from billiard paper. Increase by a factor of 10
R = 10 *(0.05 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer) / hbar
R = R.to_base_units().magnitude
Rs = R * np.logspace(-2, 2, num=5)
m = 1e-4

rMiddle = 45 * ureg.micrometer.to_base_units().magnitude
width = 3 * ureg.micrometer.to_base_units().magnitude
diffusionLength = 3 * ureg.micrometer.to_base_units().magnitude


charT = 1 / gamma_C
# charL = 10 * np.sqrt(hbar / (2 * m * me * gamma_C))
charL = (22 * ureg.micrometer).to_base_units().magnitude


def runIt(g_C, R, width, rMiddle, P):
    """
    Do a simulation, plot and save a diagnostic plot.
    """
    # if rMiddle <= 10 * ureg.micrometer.to_base_units().magnitude:
    #     N = 512
    # else:
    #     N = 1024
    ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=2.0*g_C,
                                              gamma_C=gamma_C,
                                              gamma_R=a*gamma_C, R=R, m=m,
                                              charL=charL, charT=charT)

    # ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=2.0*g_C,
    #                                           gamma_C=gamma_C,
    #                                           gamma_R=a*gamma_C, R=R, m=m)
    params = ringParams.getGPEParams()
    # params['charT'] = ringParams.charT
    P0 = P * params['Pth']
    # Make Grid
    # grid = Simulator.Grid(ringParams.charL, max_XY=max_XY_unscaled, N=N)
    # grid = Simulator.Grid(ringParams.charL, max_XY=3.0 * (rMiddle + width), N=N)
    grid = Simulator.Grid(ringParams.charL, max_XY=2.0 * (rMiddle + width), N=N)
    sigma = grid.toPixels(diffusionLength)
    x, y = grid.getSpatialGrid()
    x_us, y_us = grid.getSpatialGrid(scaled=False)
    pump = UtilityFunctions.AnnularPumpFlat(rMiddle, width, P, params['Pth'],
                                            sigma=sigma)
    # pump = UtilityFunctions.AnnularPumpGaussian(rMiddle, P, params['Pth'],
    #                                         sigma=width)

    # pump = UtilityFunctions.GaussianPump(rMiddle, P, params['Pth'])
    pumpFunction = pump.scaledFunction(ringParams.charL)
    P0 = np.max(pumpFunction(x, y))
    stabParam = (P0 / params['Pth']) / ((params['g_R'] * params['gamma_C'])
                                        / (params['g_C'] *
                                           params['gamma_R']))
    # Gaussian in the middle of the trap
    psi0 = UtilityFunctions.GaussianPump(0.2*rMiddle,
                                         0.01, 1, exponent=1.1).\
        scaledFunction(ringParams.charL)
    # psi0 = lambda x, y: 0.01 * pumpFunction(x, y) / P0
    print("Stability parameter %f (should be > 1)" % stabParam)
    dx = grid.dx_scaled
    print("Numerical Stability Parameter %f (should be < 1)" %
          ((np.pi * dt) / dx**2))
    print("P = %f" % P)
    print("Pth = %f" % params['Pth'])
    print("P0 = %f" % P0)
    print("P0/Pth = %f" % (P0 / params['Pth']))

    solver = Simulator.GPESolver(params, dt, grid.getSpatialGrid(),
                                 grid.getKSquaredGrid(), pumpFunction,
                                 psiInitial=psi0, gpu=True)
    energyTimes, energy = solver.stepTo(T_MAX, stepsPerObservation=100)
    # solver.stepTo(T_MAX, stepsPerObservation=1000)

    f = diagnosticPlot(1e6 * x_us, 1e6 * y_us, solver, energyTimes, energy,
                       pump.scaledFunction(charL=1e-6))
    f.savefig("Snoke" + getPlotName(params, r=rMiddle, m=m, sigma=diffusionLength) +
              ".png", dpi=800)
    # f.savefig("testNew" + ".png", dpi=800)
    plt.close(f)

runIt(g_C, R, width, rMiddle, 6.0)


# ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=2.0*g_C,
#                                           gamma_C=gamma_C,
#                                           gamma_R=a*gamma_C, R=R, m=m,
#                                           charL=charL, charT=charT)
#
# # ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=2.0*g_C,
# #                                           gamma_C=gamma_C,
# #                                           gamma_R=a*gamma_C, R=R, m=m)
# params = ringParams.getGPEParams()
# # params['charT'] = ringParams.charT
# P0 = P * params['Pth']
# # Make Grid
# # grid = Simulator.Grid(ringParams.charL, max_XY=max_XY_unscaled, N=N)
# # grid = Simulator.Grid(ringParams.charL, max_XY=3.0 * (rMiddle + width), N=N)
# grid = Simulator.Grid(ringParams.charL, max_XY=2.0 * (rMiddle + width), N=N)
# sigma = grid.toPixels(diffusionLength)
# x, y = grid.getSpatialGrid()
# x_us, y_us = grid.getSpatialGrid(scaled=False)
# pump = UtilityFunctions.AnnularPumpFlat(rMiddle, width, P, params['Pth'],
#                                         sigma=sigma)
# # pump = UtilityFunctions.AnnularPumpGaussian(rMiddle, P, params['Pth'],
# #                                         sigma=width)
#
# # pump = UtilityFunctions.GaussianPump(rMiddle, P, params['Pth'])
# pumpFunction = pump.scaledFunction(ringParams.charL)
# P0 = np.max(pumpFunction(x, y))
# stabParam = (P0 / params['Pth']) / ((params['g_R'] * params['gamma_C'])
#                                     / (params['g_C'] *
#                                        params['gamma_R']))
# # Gaussian in the middle of the trap
# psi0 = UtilityFunctions.GaussianPump(0.2*rMiddle,
#                                      0.01, 1, exponent=1.1).\
#     scaledFunction(ringParams.charL)
# # psi0 = lambda x, y: 0.01 * pumpFunction(x, y) / P0
# print("Stability parameter %f (should be > 1)" % stabParam)
# dx = grid.dx_scaled
# print("Numerical Stability Parameter %f (should be < 1)" %
#       ((np.pi * dt) / dx**2))
# print("P = %f" % P)
# print("Pth = %f" % params['Pth'])
# print("P0 = %f" % P0)
# print("P0/Pth = %f" % (P0 / params['Pth']))
#
# # solver = Simulator.GPESolver(params, dt, grid.getSpatialGrid(),
# #                              grid.getKSquaredGrid(), pumpFunction,
# #                              psiInitial=psi0, gpu=True)
# # energyTimes, energy = solver.stepTo(T_MAX, stepsPerObservation=100)
# # # solver.stepTo(T_MAX, stepsPerObservation=1000)
# #
# # f = diagnosticPlot(1e6 * x_us, 1e6 * y_us, solver, energyTimes, energy,
# #                    pump.scaledFunction(charL=1e-6))
# # f.savefig("Snoke" + getPlotName(params, r=rMiddle, m=m, sigma=diffusionLength) +
# #           ".png", dpi=800)
# # # f.savefig("testNew" + ".png", dpi=800)
# # plt.close(f)
