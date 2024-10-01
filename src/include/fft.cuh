// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef FFT_CORE_H
#define FFT_CORE_H

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "common_fft.cuh"
#include "fft_cpu.cuh"
#include "cuda_runtime.h"

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

namespace fft
{
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

    struct fft_configuration
    {
        int n_power;
        type ntt_type;
        ReductionPolynomial reduction_poly;
        bool zero_padding;
        COMPLEX mod_inverse;
        cudaStream_t stream;
    };

    __device__ void CooleyTukeyUnit(COMPLEX& U, COMPLEX& V, COMPLEX& root);

    __device__ void GentlemanSandeUnit(COMPLEX& U, COMPLEX& V, COMPLEX& root);

    __global__ void ForwardCore(COMPLEX* polynomial,
                                COMPLEX* root_of_unity_table, int logm,
                                int outer_iteration_count, int N_power,
                                bool zero_padding, bool not_last_kernel,
                                bool reduction_poly_check);

    __global__ void InverseCore(COMPLEX* polynomial,
                                COMPLEX* inverse_root_of_unity_table, int logm,
                                int k, int outer_iteration_count, int N_power,
                                COMPLEX n_inverse, bool last_kernel,
                                bool NTT_mult, int offset2,
                                bool reduction_poly_check);

    __host__ void GPU_FFT(COMPLEX* device_inout, COMPLEX* root_of_unity_table,
                          fft_configuration cfg, int batch_size,
                          bool multiplication);

    __global__ void GPU_ACTIVITY(unsigned long long* output,
                                 unsigned long long fix_num);
    __host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                    unsigned long long fix_num);

} // namespace fft
#endif // FFT_CORE_H
