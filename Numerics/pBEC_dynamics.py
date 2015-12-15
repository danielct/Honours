from __future__ import division
import numpy as np
import os
from numpy import fft
import time
start = time.time()
# TODO: Damping is wrong. It is 1.0 in two of the corners.

# Simulation parameters
dt = 0.1
T_MAX = 5
N_TIMESTEPS = (1.0 * T_MAX) / dt
# Number of points. It's best to keep this as a multiple of 2 - see fftw
# documentation
N = 1024
assert N % 2 == 0, "N is not even. This is going to cause all sorts of\
       problems!"
# Spatial extent. The spatial grid will run roughly from - max_XY to max_XY in
# both dimesnions
max_XY = 55
grid_element_size = (2.0 * max_XY) / (N - 1)
# Array of coordinates along the x-axis
x_axis = np.linspace(-max_XY, max_XY, num=N)
x, y = np.meshgrid(x_axis, x_axis)
assert x.shape == (N, N)
# Construct k-space grid
k_axis = ((2 * np.pi) / (2 * max_XY)) * fft.fftshift(np.linspace(- N/2, N/2 - 1,
                                                                 num=N))
k_x, k_y = np.meshgrid(k_axis, k_axis)
K = k_x ** 2 + k_y ** 2

# plt.contourf(np.absolute(psi)) Physical Parameters
g = 0.187
g_R = 2.0 * g
# Also denoted gamma_C
gamma = 1.0
gamma_R = 1.5 * gamma
R = 0.1
Pth = (gamma * gamma_R) / R

# Initial wavefunction and exponential factors
# psi0 = 0.2 * (x ** 2 + y ** 2) ** 0 * np.exp(-0.04 * (x ** 2 + y ** 2))
psi0 = np.random.normal(size=[N, N])
currentDensity = np.absolute(psi0) ** 2
# Pumping
sigma = 15.0
P = 1.5 * (200 * Pth / (sigma) ** 2) * np.exp(-1 / (2 * sigma ** 2)
                                        * (x ** 2 + y ** 2))
n = 0.0 * P
kineticFactorHalf = np.exp(-1.0j * K * dt / 2.0)
# expFactorPolariton = np.exp(-1.0j * (n * (1j * R + g_R) + g * currentDensity
#                                    - 1j * gamma))

# Quadratic Potential
# :potential = 40 * np.exp(- 0.01 * (x ** 2 + 2 * y ** 2))
# Toroidal trap
# potential = -0.001 * np.exp(-0.01 * (x **2 + y ** 2)) * (x **2 + y **2) ** 2
potential = 0.0
# Damping boundaries?
damping = (0.5 * np.tanh(5 * (x + max_XY - 5)) * np.tanh(5 + (y + max_XY - 5))
           + 0.5 * np.tanh(5 * (- x + max_XY - 5)) *
           np.tanh(5 * (- y + max_XY - 5)))
# damping = 1.0
# Set up arrays to store the density
# First two dimensions are spatial, third is time
# density = np.zeros(x.shape + tuple([N_TIMESTEPS]))

# Run simulation
psi = psi0
for step in np.arange(N_TIMESTEPS):
    # Implementing split-step method
    # Update wavefunction and resovoir, record density
    psi = fft.ifft2(kineticFactorHalf * fft.fft2(psi))
    currentDensity = np.absolute(psi) ** 2

    expFactorExciton = np.exp(- (gamma_R + R * currentDensity) * dt)
    n *= expFactorExciton
    n += P * dt

    # expFactorPolariton = np.exp(-1j * (n * (1j * R + g_R) + g * currentDensity
    #                                    - 1j * gamma))
    expFactorPolariton = np.exp((n * (0.5 * R - 1.0j * g_R) - 0.5 * gamma
                                 - 1.0j * g * currentDensity) * dt)

    psi = (fft.ifft2(kineticFactorHalf * fft.fft2(expFactorPolariton * psi))
           * damping)
    # Update exponential factors and density

    # Record density
    print("Time = %f" % (step * dt))

# ding!
end = time.time()
print("%d steps in %f seconds. On average, %f seconds per step"
      % (N_TIMESTEPS, end - start, (end - start) / N_TIMESTEPS))
os.system('play --no-show-progress --null --channels 1 synth %s sine %f'
          % (1, 400))
