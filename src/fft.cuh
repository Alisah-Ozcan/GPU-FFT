
#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "common.cuh"
#include "complex.cuh"
#include "cuda_runtime.h"

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#ifndef NTT_FFT_CORE_H
#define NTT_FFT_CORE_H

#if MAX_LOG2_RINGSIZE <= 32
typedef unsigned location_t;
#else
typedef unsigned long long location_t;
#endif

enum type
{
    FORWARD,
    INVERSE
};

struct ntt_configuration
{
    int n_power;
    type ntt_type;
    bool zero_padding;
    COMPLEX mod_inverse;
    cudaStream_t stream;
};

__device__ void CooleyTukeyUnit(COMPLEX& U, COMPLEX& V, COMPLEX& root);

__device__ void GentlemanSandeUnit(COMPLEX& U, COMPLEX& V, COMPLEX& root);

__global__ void ForwardCore(COMPLEX* polynomial, COMPLEX* root_of_unity_table,
                            int logm, int outer_iteration_count, int N_power,
                            bool zero_padding, bool not_last_kernel);

__global__ void InverseCore(COMPLEX* polynomial,
                            COMPLEX* inverse_root_of_unity_table, int logm,
                            int k, int outer_iteration_count, int N_power,
                            COMPLEX n_inverse, bool last_kernel, bool NTT_mult,
                            int offset2);

__host__ void GPU_NTT(COMPLEX* device_inout, COMPLEX* root_of_unity_table,
                      ntt_configuration cfg, int batch_size,
                      bool multiplication);

__global__ void GPU_ACTIVITY(unsigned long long* output,
                             unsigned long long fix_num);
__host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                unsigned long long fix_num);

#endif  // NTT_FFT_CORE_H
