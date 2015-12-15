from __future__ import division
import numpy as np
from numpy import fft
import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from skcuda import misc
from pycuda import cumath

REAL_DTYPE = np.float32
COMPLEX_DTYPE = np.complex64
# TODO: Damping is wrong. It is 1.0 in two of the corners.

# Simulation parameters
dt = 0.1
T_MAX = 20
N_TIMESTEPS = int((1.0 * T_MAX) / dt)
N_RUNS = 2
times = np.zeros(N_RUNS)
# Number of points. It's best to keep this as a multiple of 2 - see fftw
# documentation
N = 512
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
psi0 = np.random.normal(size=[N, N]) + 0j
currentDensity = np.absolute(psi0) ** 2
# Pumping
sigma = 15.0
P = 1.5 * (200 * Pth / (sigma) ** 2) * np.exp(-1 / (2 * sigma ** 2)
                                              * (x ** 2 + y ** 2))
n = np.array(0.0 * P)
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

# make GPU versions of all of our arrays.
psi_gpu = gpuarray.to_gpu(psi.astype(COMPLEX_DTYPE))
n_gpu = gpuarray.to_gpu(n.astype(REAL_DTYPE))
kineticFactorHalf_gpu = gpuarray.to_gpu(kineticFactorHalf.astype(COMPLEX_DTYPE))
damping_gpu = gpuarray.to_gpu(damping.astype(REAL_DTYPE))
currentDensity_gpu = gpuarray.to_gpu(currentDensity.astype(REAL_DTYPE))
Pdt_gpu = gpuarray.to_gpu((P*dt).astype(REAL_DTYPE))
gammaRdt_gpu = gpuarray.to_gpu(np.array(gamma_R * dt).astype(REAL_DTYPE))
Rdt_gpu = gpuarray.to_gpu(np.array(R * dt).astype(REAL_DTYPE))
spectrum = gpuarray.to_gpu(np.zeros((N_TIMESTEPS, N)).astype(COMPLEX_DTYPE))
# expFactorExciton_gpu = cumath.exp(-gammaRdt_gpu +
#                                   (Rdt_gpu * currentDensity_gpu))
expFactorPolFirst = (0.5 * R - 1j * g_R) * dt
expFactorPolSecond = -1j * g * dt
expFactorPolThird = -0.5 * gamma * dt
expFactorPolFirst_gpu = gpuarray.to_gpu(np.array((0.5 * R - 1j * g_R) * dt)
                                        .astype(COMPLEX_DTYPE))
expFactorPolSecond_gpu = gpuarray.to_gpu(np.array(-1j * g * dt)
                                         .astype(COMPLEX_DTYPE))
expFactorPolThird_gpu = gpuarray.to_gpu(np.array(-0.5 * gamma * dt)
                                        .astype(COMPLEX_DTYPE))
# make FFT plans
# TODO: Are these 2D????
plan_forward = cu_fft.Plan((N, N), COMPLEX_DTYPE, COMPLEX_DTYPE)
plan_inverse = cu_fft.Plan((N, N), COMPLEX_DTYPE, COMPLEX_DTYPE)


mod = SourceModule("""
     #include <pycuda-complex.hpp>
                   __global__ void modSquared(pycuda::complex<float> *a,
                   float *dest, int N)
        {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         int idy = blockIdx.y * blockDim.y + threadIdx.y;
         int ind = idx + N * idy;

         dest[ind] = a[ind]._M_re * a[ind]._M_re
                   + a[ind]._M_im * a[ind]._M_im;
        }
                   """)
mod2 = SourceModule("""
    #include <pycuda-complex.hpp>

    __device__ pycuda::complex<float> comExp(pycuda::complex<float> z)
    {
     pycuda::complex<float> res;
     float s, c;
     float e = expf(z.real());
     sincosf(z.imag(), &s, &c);
     res._M_re = c * e;
     res._M_im = s * e;
     return res;
    }


    __global__ void test(pycuda::complex<float>  a1,
    pycuda::complex<float> a2, pycuda::complex<float> a3,
    pycuda::complex<float> *dest, float *x1, float *x2, int N)
        {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         int idy = blockIdx.y * blockDim.y + threadIdx.y;
         int ind = idx + N * idy;

         dest[ind] *= comExp(a1 * x1[ind] + a2 * x2[ind] + a3);
        }
                   """)

modSquared = mod.get_function("modSquared")
psiNonlinear = mod2.get_function("test")
modSquared.prepare(["P", "P", "I"])
psiNonlinear.prepare("FFFPPPI")
block = (16, 16, 1)
grid = (64, 64)

for n in np.arange(N_RUNS):
    start = time.time()

    for step in xrange(N_TIMESTEPS):
        # print step
       # Implementing split-step method
       # Update wavefunction and resovoir, record density
        cu_fft.fft(psi_gpu, psi_gpu, plan_forward)
        psi_gpu *= kineticFactorHalf_gpu
        cu_fft.ifft(psi_gpu, psi_gpu, plan_inverse, scale=True)

        # currentDensity_gpu = abs(psi_gpu) ** 2
        # currentDensity_gpu = psi_gpu.real **2 + psi_gpu.imag ** 2
        currentDensity_gpu = (psi_gpu * psi_gpu.conj()).real
        # modSquared.prepared_call(grid, block, psi_gpu.gpudata,
        #                          currentDensity_gpu.gpudata, 1024)
        # n_gpu *= cumath.exp(-gammaRdt_gpu + Rdt_gpu * currentDensity_gpu)
        n_gpu *= cumath.exp(misc.add(- gammaRdt_gpu,
                                     - misc.multiply(Rdt_gpu, currentDensity_gpu)))
        n_gpu += Pdt_gpu
        psi_gpu *= cumath.exp(
            misc.add(
                misc.add(misc.multiply(expFactorPolFirst_gpu, n_gpu),
                         misc.multiply(expFactorPolSecond_gpu, currentDensity_gpu)),
                expFactorPolThird_gpu))

        #  psiNonlinear.prepared_call(grid, block, expFactorPolFirst,
        #                             expFactorPolSecond, expFactorPolThird,
        #                             psi_gpu.gpudata, n_gpu.gpudata,
        #                             currentDensity_gpu.gpudata, 1024)

        cu_fft.fft(psi_gpu, psi_gpu, plan_forward)
        # record spectrum
        drv.memcpy_dtod(spectrum[step, :].gpudata, psi_gpu[N//2, :].gpudata,
                        psi_gpu[N//2, :].nbytes)
        psi_gpu *= kineticFactorHalf_gpu
        cu_fft.ifft(psi_gpu, psi_gpu, plan_inverse, scale=True)

        psi_gpu *= damping_gpu
        n_gpu *= damping_gpu

       # # Record density
       # print("Time = %f" % (step * dt))

    # for step in np.arange(N_TIMESTEPS):
    #     cu_fft.fft(psi_gpu, psi_gpu, plan_forward)
    #     cu_fft.ifft(psi_gpu, psi_gpu, plan_inverse, scale=True)
    #     cu_fft.fft(psi_gpu, psi_gpu, plan_forward)
    #     cu_fft.ifft(psi_gpu, psi_gpu, plan_inverse, scale=True)

    # ding!
    end = time.time()
    times[n] = (end - start) / N_TIMESTEPS

print("%f +/- %f seconds per step" % (times.mean(), times.std()))
# print("%d steps in %f seconds. On average, %f seconds per step"
#       % (N_TIMESTEPS, end - start, (end - start) / N_TIMESTEPS))
# os.system('play --no-show-progress --null --channels 1 synth %s sine %f'
#           % (1, 400))

psiFinal = psi_gpu.get()
nFinal = n_gpu.get()
