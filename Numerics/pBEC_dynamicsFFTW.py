from __future__ import division
import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
import json
import pyfftw
from numpy import fft
import time
import UtilityFunctions
start = time.time()
# TODO: Damping is wrong. It is 1.0 in two of the corners.

# Simulation parameters
dt = 0.1
T_MAX = 50
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

g = 0.187
g_R = 2.0 * g
# Also denoted gamma_C
gamma = 1.0
gamma_R = 1.5 * gamma
R = 0.1
Pth = (gamma * gamma_R) / R

# Initial wavefunction and exponential factors
# psi0 = 0.2 * (x ** 2 + y ** 2) ** 0 * np.exp(-0.04 * (x ** 2 + y ** 2))
# psi0 = np.random.normal(size=[N, N])
psi0 = (np.abs(np.random.normal(size=(N, N), scale=10e-3,
                                loc=10e-2))
        + 1.0j*np.random.normal(size=(N, N), scale=10e-3,
                                loc=10e-2))
currentDensity = np.absolute(psi0) ** 2
# Pumping
sigma = 15.0
# P = 1.5 * (200 * Pth / (sigma) ** 2) * np.exp(-1 / (2 * sigma ** 2)
#                                               * (x ** 2 + y ** 2))
pumpFunction = UtilityFunctions.GaussianPump(sigma, 1.5*200 / sigma**2,
                                             Pth).unscaledFunction()
P = pumpFunction(x, y)
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
# damping = (0.5 * np.tanh(5 * (x + max_XY - 5)) * np.tanh(5 + (y + max_XY - 5))
#            + 0.5 * np.tanh(5 * (- x + max_XY - 5)) *
#            np.tanh(5 * (- y + max_XY - 5)))
dampingFunction = UtilityFunctions.TanhDamping(max_XY).unscaledFunction()
damping = dampingFunction(x, y)
# Set up arrays to store the density
# First two dimensions are spatial, third is time
# density = np.zeros(x.shape + tuple([N_TIMESTEPS]))

# Set up fft and inverse fft
# NOTE: psi must be initialised to psi0 *after* the plan is created. Creation of
# the plan may erase the contents of psi!
#
# When we call fft_object(), psi will be replaced with the fft of psi. The same
# is true for ifft_object
# Optimal alignment for the CPU
if len(sys.argv) > 1:
    f = open(sys.argv[1])
    importStatus = pyfftw.import_wisdom(json.load(f))
    if not np.array(importStatus).all():
        raise IOError("Wisdom not correctly loaded")
if len(sys.argv) > 2:
    N_THREADS = int(sys.argv[2])
else:
    N_THREADS = 2
print("N threads: %d" % N_THREADS)
al = pyfftw.simd_alignment
psi = pyfftw.n_byte_align_empty((N, N), al, 'complex128')
flag = 'FFTW_PATIENT'
fft_object = pyfftw.FFTW(psi, psi, flags=[flag], axes=(0, 1), threads=N_THREADS)
ifft_object = pyfftw.FFTW(psi, psi, flags=[flag], axes=(0, 1),
                          threads=N_THREADS, direction='FFTW_BACKWARD')

# copy psi0 into psi. To be safe about keeping the alignment of psi, set all the
# entries to 1 and then multiply.
psi[:] = 1.0
psi *= psi0

# Check psi is aligned
assert pyfftw.is_n_byte_aligned(psi, al)
print("Planning finished")
# Run simulation
for step in np.arange(N_TIMESTEPS):
    # Implementing split-step method
    # Update wavefunction and resovoir, record density

    # Take fft, multiply by kinetic factor and then take inverse fft.
    fft_object()
    psi *= kineticFactorHalf
    ifft_object()
    # psi = fft.ifft2(kineticFactorHalf * fft.fft2(psi))
    currentDensity = np.absolute(psi) ** 2

    expFactorExciton = np.exp(- (gamma_R + R * currentDensity) * dt)
    n *= expFactorExciton
    n += P * dt

    expFactorPolariton = np.exp((n * (0.5 * R - 1.0j * g_R) - 0.5 * gamma
                                 - 1.0j * g * currentDensity
                                 - 1.0j * potential) * dt)

    # Do the nonlinear update, take FFT, do kinetic update, and then take the
    # inverse fft
    psi *= expFactorPolariton
    fft_object()
    psi *= kineticFactorHalf
    ifft_object()
    psi *= damping
    # psi = (fft.ifft2(kineticFactorHalf * fft.fft2(expFactorPolariton * psi))
    #        * damping)
    print("Time = %f" % (step * dt))
end = time.time()
timePerStep = (end - start) / N_TIMESTEPS
print ("%d threads. On average, %f seconds per timestep"
       % (N_THREADS, timePerStep))
print("%d steps in %f seconds. On average, %f seconds per step"
      % (N_TIMESTEPS, end - start, (end - start) / N_TIMESTEPS))
os.system('play --no-show-progress --null --channels 1 synth %s sine %f'
          % (1, 400))
