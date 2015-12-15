import numpy as np
from numpy import fft
import matplotlib.pyplot as plt


def plotLineDensities(grid, solver):
    N = solver.N
    f, (ax1, ax2) = plt.subplots(1, 2)
    phi = fft.fftshift(fft.fft2(solver.psiFinal))
    ax1.plot(1e6 * grid.x_axis_unscaled[N/2:],
             (np.absolute(solver.psiFinal[N/2:, N/2])
              / np.absolute(solver.psiFinal[N/2, N/2])))
    ax1.set_xlabel("r")
    ax1.set_ylabel("p(r)")
    ax2.plot(fft.fftshift(grid.k_axis_scaled)[N/2 - N/8: N/2 + N/8],
             np.absolute(phi[N/2, N/2 - N/8: N/2 + N/8])
             / np.max(np.absolute(phi[N/2, :])))
    ax2.set_xlabel('k')
    ax2.set_ylabel('p(k)')
    return f
