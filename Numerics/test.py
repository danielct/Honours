import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from jinja2 import Template

tpl = Template("""
    #include <pycuda-complex.hpp>

    __device__ pycuda::complex<float> comexp(pycuda::complex<float> z)
    {
     pycuda::complex<float> res;
     float s, c;
     float e = expf(z.real());
     sincosf(z.imag(), &s, &c);
     res._m_re = c * e;
     res._m_im = s * e;
     return res;
    }


    __global__ void test(pycuda::complex<float> *dest, float *x1, float *x2, int n)
        {
         pycuda::complex<float> a1

         int idx = blockidx.x * blockdim.x + threadidx.x;
         int idy = blockidx.y * blockdim.y + threadidx.y;
         int ind = idx + n * idy;

         dest[ind] = comexp(a1 * x1[ind] + a2 * x2[ind] + a3);
        }
""")

mod = SourceModule("""
  __global__ void test(int N, float *a, float *dest)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = idx + N * idy;
    dest[ind] = 2 * a[ind];
  }
  """)

# mod = SourceModule("""
#   __global__ void test(float *a)
#   {
#     int idx = threadIdx.x + threadIdx.y*4;
#     a[idx] *= 2;
#   }
#   """)

fun = mod.get_function("test")
# Prepare the function to be called with an integer and two pointers
# This is required because we can't pass an integer directly to a
# kernel!
fun.prepare("IPP")

# Usage:
# grid = (, )
# block = (, )
# fun.prepared_call(grid, block, *args)

# Test a module that will return the mod square of a complex array

mod = SourceModule("""
     #include <pycuda-complex.hpp>
                   __global__ void test(pycuda::complex<float> *a,
                   float *dest, int N)
        {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         int idy = blockIdx.y * blockDim.y + threadIdx.y;
         int ind = idx + N * idy;

         dest[ind] = a[ind]._M_re * a[ind]._M_re + a[ind]._M_im * a[ind]._M_im;

         //dest[ind] = a[ind].real() * a[ind].real()
         //          + a[ind].imag() * a[ind].imag();
        }
                   """)

mod = SourceModule("""
    #include <pycuda-complex.hpp>

    __device__ pycuda::complex<float> comexp(pycuda::complex<float> z)
    {
     pycuda::complex<float> res;
     float s, c;
     float e = expf(z.real());
     sincosf(z.imag(), &s, &c);
     res._m_re = c * e;
     res._m_im = s * e;
     return res;
    }


    __global__ void test(pycuda::complex<float>  a1,
    pycuda::complex<float> a2, pycuda::complex<float> a3,
    pycuda::complex<float> *dest, float *x1, float *x2, int n)
        {
         int idx = blockidx.x * blockdim.x + threadidx.x;
         int idy = blockidx.y * blockdim.y + threadidx.y;
         int ind = idx + n * idy;

         dest[ind] = comexp(a1 * x1[ind] + a2 * x2[ind] + a3);
        }
                   """)


fun = mod.get_function("test")
