import matplotlib
# Default seems to give us segfaults (!) occassionally.
# GTKAgg seems to work
# Can use Agg if running without X
# matplotlib.use('GTKAgg')
matplotlib.use('Agg')
import Simulator
import UtilityFunctions
import pint
from matplotlib import pyplot as plt
import numpy as np
# import itertools
from UtilityFunctions import diagnosticPlot, getPlotName
# TODO: rmiddle was changed

ureg = pint.UnitRegistry()

# --- Sim Params ---
singleComp = True
N = 2 * 512
dt = 1e-3
max_XY_unscaled = (20 * ureg.micrometer)
# The simulation will run to T_STABLE and then run for long enough to record the
# spectrum
T_STABLE = 50
# This length ensures that dE ~ 0.1meV
spectLength = 1e4 * dt
T_MAX = T_STABLE + spectLength + 5*dt
# To ensure that dx = 0.1 in scaled units and that dt=5e-3 is a good timestep.
# This approach to scaling and grid size seems to work well.
charL = max_XY_unscaled / 100
# charT = 1 / gamma_C
# Corresponds to
charT = 5 * ureg.picosecond


# --- GPE Params ---

# Snoke
# g_C = (0.88 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)

# Billiard
# g_C = (2e-3 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
# g_R = 2.0 * g_C
# R = (6e-4 * ureg.millieV * ureg.micrometer**2 * ureg.hbar**-1)
# gamma_C = (0.1 * ureg.picosecond ** -1)
# gamma_R = a * gamma_C
# m = 5e-5 * ureg.electron_mass

# Liew Robust
# g_C = (2.4e-3 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_C = (6.2e-3 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_R = 2.0 * g_C
# R = (1.2e-3 * ureg.millieV * ureg.micrometer**2 * ureg.hbar**-1)
R = (1.7e-3 * ureg.millieV * ureg.micrometer**2 * ureg.hbar**-1)
gamma_C = (0.2 * ureg.picosecond ** -1)
a = 1.1
gamma_R = a * gamma_C
m = 7e-5 * ureg.electron_mass

# Single component parameters
gamma_nl = 0.3 * g_C / ureg.hbar
print gamma_nl
# Best Params from scan
# g_C = (2e-2 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
# R = (6e-2 * ureg.millieV * ureg.micrometer**2 * ureg.hbar**-1)
# gamma_C = (0.05 * ureg.picosecond ** -1)
# gamma_R = a * gamma_C
# m = 7e-5 * ureg.electron_mass

# --- Ring Params ---

# From experiment picture
# rMiddle = (7.5 * ureg.micrometer).to_base_units().magnitude
# For thin ring
rMiddle = (5.5 * ureg.micrometer).to_base_units().magnitude
# From experiment picture
# width = (5 * ureg.micrometer).to_base_units().magnitude
# Thin ring
width = (1 * ureg.micrometer).to_base_units().magnitude
# for experiment picture
# diffusionLength = (2 * ureg.micrometer).to_base_units().magnitude
# Thin ring
diffusionLength = (0.5 * ureg.micrometer).to_base_units().magnitude


# Things for scan
# Ps = np.linspace(0.01, 0.5, num=10)
# Ps = np.linspace(0.1, 3.5, num=25)
# Ps = np.linspace(3.94, 7.5, num=9)
Ps = np.linspace(0.5, 15, num=20)
# g_Cs = g_C * np.linspace(0.2, 5.0, num=5)
# Rs = R * np.linspace(0.2, 5.0, num=5)


def runIt(g_C, R, gamma_C, a, width, diffusionLength, rMiddle, P):
    """
    Do a simulation, plot and save a diagnostic plot.
    """
    if singleComp:
        ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=2.0*g_C,
                                                  gamma_nl=gamma_nl,
                                                  gamma_C=gamma_C,
                                                  gamma_R=a*gamma_C, R=R, m=m,
                                                  charL=charL, charT=charT)
    else:
        ringParams = Simulator.ParameterContainer(g_C=g_C, g_R=2.0*g_C,
                                                  gamma_C=gamma_C,
                                                  gamma_R=a*gamma_C, R=R, m=m,
                                                  charL=charL, charT=charT)
    params = ringParams.getGPEParams()
    print params
    Pth = params["Pth"]
    # --- Grid, associated quantities ---
    grid = Simulator.Grid(ringParams.charL,
                          max_XY=max_XY_unscaled, N=N)
    sigma = grid.toPixels(diffusionLength)
    x, y = grid.getSpatialGrid()
    x_us, y_us = grid.getSpatialGrid(scaled=False)
    dx = grid.dx_scaled
    # Time to record spectrum to. This will ensure that we have a resolution of
    # at least 0.01 meV
    spectMax = int(5e-6 / grid.dx_unscaled)
    # Mask for collection of spectrum and energy. This will ensure that we only
    # collect from the center of the ring
    mask = x_us ** 2 + y_us ** 2 > (5e-6) ** 2

    # --- Initial wavefunction ---
    # Gaussian in the middle of the trap
    psi0 = UtilityFunctions.GaussianPump(0.2*rMiddle,
                                         0.1, 1, exponent=1.1).\
        scaledFunction(ringParams.charL)

    pump = UtilityFunctions.AnnularPumpFlat(rMiddle, width, P, Pth,
                                            sigma=sigma)
    # Since we used Pth from params, the pump will be in units of
    # L_C^-2 * T_C^-1
    pumpFunction = pump.scaledFunction(ringParams.charL)

    # P0 = np.max(pumpFunction(x, y))
    # print P0
    # stabParam = (P0 / params['Pth']) / ((params['g_R'] * params['gamma_C'])
    #                                     / (params['g_C'] *
    #                                        params['gamma_R']))
    # print("Stability parameter %f (should be > 1)" % stabParam)
    numStabParam = (np.pi * dt) / dx**2

    if numStabParam >= 0.8:
        print("Warning, numerical stability parameter is %.4f \n Should be <1 "
              % numStabParam)
    print("Numerical Stability Parameter %f (should be < 1)" %
          ((np.pi * dt) / dx**2))

    solver = Simulator.GPESolver(ringParams, dt, grid,
                                 pumpFunction=pumpFunction, psiInitial=psi0,
                                 REAL_DTYPE=np.float64,
                                 COMPLEX_DTYPE=np.complex128)
    solver.stepTo(T_MAX, stepsPerObservation=1000, spectStart=T_STABLE,
                  normalized=False, spectLength=spectLength, printTime=True,
                  collectionMask=mask, spectMax=spectMax)
    f = diagnosticPlot(solver)
    f.savefig("PScanSingle " + getPlotName(ringParams, P=P) + ".png", dpi=500)
    # f.savefig("testPlot" + ".png", dpi=600)
    # solver.save("testPlot", P=P)
    solver.save("PScanSingle " + getPlotName(ringParams.getOutputParams(), P=P),
                P=P)
    plt.close(f)

# --- Experiments ---
# runIt(g_C, R, gamma_C, a, width, diffusionLength, rMiddle, 2.5)

# for dl in dLs:
#     runIt(g_C, R, gamma_C, width, dl, rMiddle, 2.5)

# for a in As:
#     runIt(g_C, R, gamma_C, a, width, diffusionLength, rMiddle, 2.1)

for P in Ps:
    runIt(g_C, R, gamma_C, a, width, diffusionLength, rMiddle, P)

# for (g_Ci, Ri) in itertools.product(g_Cs, Rs):
#     # To keep trap depth constant. Can use g_C instead of g_R because g_R is
#     # proportional to g_C
#     P_0 = 1.8
#     P = (P_0 * g_C) / g_Ci
#     runIt(g_Ci, Ri, gamma_C, a, width, diffusionLength, rMiddle, P)
#
