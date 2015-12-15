import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
# Physical Parameters
g = 1.0
g_R = 2.0 * g
# Also denoted gamma_C
gamma = 1.0
gamma_R = (1.0 / 0.1) * gamma
R = 1
P_th = (gamma * gamma_R) / R
P = 1.2 * P_th
psi0 = np.sqrt((P - P_th) / gamma)
# noise = 10e-2 * psi_0 * np.random.normal(size=N)
# Simulation parameters
dt = 0.1
T_MAX = 100
N_TIMESTEPS = (1.0 * T_MAX) / dt
# Number of points. It's best to keep this as a multiple of 2 - see fftw
# documentation
N = 1024
# Spatial extent. The spatial grid will run roughly from - max_XY to max_XY in
# both dimesnions
max_XY = 150
grid_element_size = (2 * max_XY) / (1.0 * N)
x_grid = np.arange(-max_XY, max_XY, grid_element_size)
assert x_grid.shape[0] == N
# In the arange, starts at -N/2 and ends at N/2 - 1
k = ((2 * np.pi) / (2 * max_XY)) * fft.fftshift(np.arange(- N/2, N/2))
K = k ** 2

# Initial wavefunction and exponential factors
psi = psi0 * np.ones(N)
# Start with density equal to steady-state density. Must change for other
# simulations
n = gamma / R * np.ones(N)
currentDensity = np.absolute(psi) ** 2
expFactorPolariton = np.exp(-1j * (n * (1j * R + g_R) + g * currentDensity
                                   - 1j * gamma))
expFactorExciton = np.exp(- (gamma_R + R * currentDensity) * dt)
kineticFactor = np.exp(-1j * K * dt)

# Set up arrays to store the density
density = np.zeros((N, ))

# Run simulation
for step in np.arange(N_TIMESTEPS):
    # Update wavefunction and resovoir, record density
    psi = fft.ifft(kineticFactor * fft.fft(expFactorPolariton * psi))
    n *= (expFactorExciton + P)

    # Update exponential factors and density
    currentDensity = np.absolute(psi) ** 2
    expFactorPolariton = np.exp(-1j * (n * (1j * R + g_R) + g * currentDensity
                                       - 1j * gamma))
    expFactorExciton = np.exp(- (gamma_R + R * currentDensity) * dt)
    # Record density
    density[:, step] = np.copy(currentDensity)
    print("Time: %d" % (step * dt))

time = np.arange(density.shape[1]) * dt
plt.pcolor(density)
plt.show()
