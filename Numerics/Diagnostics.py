import matplotlib.pyplot as plt
import numpy as np
from numpy import fft

def getPlotName(params):
    outParams = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'charT']
    outStrings = ["=".join([outParam, str(params[outParam])])
                  for outParam in outParams]
    return ", ".join(outStrings)


def plot(x, y, solver, pumpFunction):
    """
    Returns a matplotlib figure
    """
    psi = solver.psiFinal
    fig, axes = plt.subplots(3, 2)
    p0 = axes[0, 0].contourf(x, y, np.absolute(psi))
    axes[0, 0].set_title("Real Space Density")
    plt.colorbar(p0, ax=axes[0, 0])

    p1 = axes[0, 1].contourf(x, y, np.absolute(fft.fftshift(fft.fft2(psi))))
    axes[0, 1].set_title("k Space Density")
    plt.colorbar(p1, ax=axes[0, 1])

    p2 = axes[1, 0].contourf(x, y, np.angle(psi))
    axes[1, 0].set_title("Phase")
    plt.colorbar(p2, ax=axes[1, 0])

    N = solver.psiFinal.shape[0]
    axes[1, 1].plot(x[N/2, :], np.absolute(solver.psiFinal[N/2, :]))
    axes[1, 1].set_title("Radial Profile")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_xlabel("x")

    axes[2, 0].plot(solver.energyTimes, solver.energy)
    axes[2, 0].set_title("Energy")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("Energy")

    p4 = axes[2, 1].contourf(x, y, pumpFunction(x, y) / solver.Pth)
    axes[2, 1].set_title("Pump Density (units of Pth)")
    plt.colorbar(p4, ax=axes[2, 1])

    fig.tight_layout()
    return fig

