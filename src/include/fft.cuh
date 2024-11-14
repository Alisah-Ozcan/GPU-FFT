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

    __global__ void ForwardCore(FIXED_POINT* input, COMPLEX* polynomial,
                                COMPLEX* root_of_unity_table, int logm,
                                int outer_iteration_count, int N_power,
                                bool zero_padding, bool not_last_kernel,
                                bool reduction_poly_check);

    __global__ void InverseCore(FIXED_POINT* output, COMPLEX* polynomial,
                                COMPLEX* inverse_root_of_unity_table, int logm,
                                int k, int outer_iteration_count, int N_power,
                                COMPLEX n_inverse, bool last_kernel,
                                bool NTT_mult, int offset2,
                                bool reduction_poly_check);

    // Complex to Complex
    __host__ void GPU_FFT(COMPLEX* device_inout, COMPLEX* root_of_unity_table,
                          fft_configuration cfg, int batch_size,
                          bool multiplication);

    // FFT: Real to Complex
    // IFFT: Complex to Real
    __host__ void GPU_FFT(FIXED_POINT* device_fix_inout,
                          COMPLEX* device_complex_inout,
                          COMPLEX* root_of_unity_table, fft_configuration cfg,
                          int batch_size, bool multiplication);

    ////////////////////////////////////////////
    ////////////////////////////////////////////

    //      ---- Special FFT for HE encoding ----
    /**
     * FFT-like algorithm for evaluating SF_l
     *
     * Input:
     *   - l > 1 (a power of 2 integer)
     *   - z ∈ C^l
     *   - Ψ: A precomputed table of complex 4l-roots of unities
     *        where Ψ[j] = exp(πij/2l), 0 ≤ j < 4l.
     *
     * Output:
     *   - w = SF_l * z
     *
     * Steps:
     *
     * 1. Initialize w with the values of z
     *    w = z;
     *
     * 2. Perform bit reversal on w with length l
     *    bitReverse(w, l);
     *
     * 3. Iteratively compute the FFT-like transformation
     *    for (m = 2; m <= l; m = 2 * m) {
     *        // Outer loop to increase the sub-array size
     *        for (i = 0; i < l; i += m) {
     *            // Inner loop to process each sub-array
     *            for (j = 0; j < m / 2; j++) {
     *                // 4. Compute index for the twiddle factor
     *                k = (5^j mod 4m) * (l / m);
     *
     *                // 5. Load values from w to U and V
     *                U = w[i + j];
     *                V = w[i + j + m / 2];
     *
     *                // 6. Multiply V by the twiddle factor Ψ[k]
     *                V = V * Ψ[k];
     *
     *                // 7. Update w with computed values
     *                w[i + j] = U + V;
     *                w[i + j + m / 2] = U - V;
     *            }
     *        }
     *    }
     *
     * 8. Return the transformed array w
     *    return w;
     */

    __global__ void Special_ForwardCore(COMPLEX* polynomial,
                                        COMPLEX* inverse_root_of_unity_table,
                                        int logm, int k,
                                        int outer_iteration_count, int N_power);

    __global__ void Special_InverseCore(COMPLEX* polynomial,
                                        COMPLEX* root_of_unity_table, int logm,
                                        int outer_iteration_count, int N_power,
                                        COMPLEX n_inverse,
                                        bool not_last_kernel);

    __host__ void GPU_Special_FFT(COMPLEX* device_inout,
                                  COMPLEX* root_of_unity_table,
                                  fft_configuration cfg, int batch_size);

    ////////////////////////////////////////////
    ////////////////////////////////////////////

    __global__ void GPU_ACTIVITY(unsigned long long* output,
                                 unsigned long long fix_num);
    __host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                    unsigned long long fix_num);

} // namespace fft
#endif // FFT_CORE_H
