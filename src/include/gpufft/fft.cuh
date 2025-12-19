// Copyright 2023-2025 Alişah Özcan
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
#include <functional>
#include <unordered_map>

#include "common_fft.cuh"
#include "fft_cpu.cuh"
#include "cuda_runtime.h"

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

namespace gpufft
{
    typedef unsigned location_t;
    /*
    #if MAX_LOG2_RINGSIZE <= 32
        typedef unsigned location_t;
    #else
        typedef unsigned long long location_t;
    #endif
    */
    enum type
    {
        FORWARD,
        INVERSE
    };

    template <typename T> struct fft_configuration
    {
        int n_power;
        type fft_type;
        ReductionPolynomial reduction_poly;
        bool zero_padding;
        COMPLEX<T> mod_inverse;
        cudaStream_t stream;
    };

    struct KernelConfig
    {
        int griddim_x;
        int griddim_y;
        int blockdim_x;
        int blockdim_y;
        size_t shared_memory;

        int shared_index;
        int logm;
        int k;
        int outer_iteration_count;

        bool not_last_kernel;
    };

    template <typename T>
    __device__ __forceinline__ void CooleyTukeyUnit(COMPLEX<T>& U, COMPLEX<T>& V,
                                    COMPLEX<T>& root)
    {
        COMPLEX<T> u_ = U;
        COMPLEX<T> v_ = V * root;

        U = u_ + v_;
        V = u_ - v_;
    }

    template <typename T>
    __device__ __forceinline__ void GentlemanSandeUnit(COMPLEX<T>& U, COMPLEX<T>& V,
                                       COMPLEX<T>& root)
    {
        COMPLEX<T> u_ = U;
        COMPLEX<T> v_ = V;

        U = (u_ + v_);
        V = (u_ - v_) * root;
    }

    template <typename T>
    __global__ void
    ForwardCore(COMPLEX<T>* polynomial, COMPLEX<T>* root_of_unity_table,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool zero_padding, bool not_last_kernel,
                bool reduction_poly_check);

    template <typename T>
    __global__ void
    InverseCore(COMPLEX<T>* polynomial, COMPLEX<T>* inverse_root_of_unity_table,
                int shared_index, int logm, int k, int outer_iteration_count,
                int N_power, COMPLEX<T> n_inverse, bool last_kernel,
                bool NTT_mult, int offset2, bool reduction_poly_check);

    template <typename T>
    __global__ void
    ForwardCore(T* input, COMPLEX<T>* polynomial,
                COMPLEX<T>* root_of_unity_table, int shared_index, int logm,
                int outer_iteration_count, int N_power, bool zero_padding,
                bool not_last_kernel, bool reduction_poly_check);

    template <typename T>
    __global__ void
    InverseCore(T* output, COMPLEX<T>* polynomial,
                COMPLEX<T>* inverse_root_of_unity_table, int shared_index,
                int logm, int k, int outer_iteration_count, int N_power,
                COMPLEX<T> n_inverse, bool last_kernel, bool NTT_mult,
                int offset2, bool reduction_poly_check);

    // COMPLEX<T> to COMPLEX<T>
    template <typename T>
    __host__ void
    GPU_FFT(COMPLEX<T>* device_inout, COMPLEX<T>* root_of_unity_table,
            fft_configuration<T> cfg, int batch_size, bool multiplication);

    // FFT: Real to COMPLEX<T>
    // IFFT: COMPLEX<T> to Real
    template <typename T>
    __host__ void
    GPU_FFT(T* device_fix_inout, COMPLEX<T>* device_Complex64_inout,
            COMPLEX<T>* root_of_unity_table, fft_configuration<T> cfg,
            int batch_size, bool multiplication);

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

    template <typename T>
    __global__ void Special_ForwardCore(COMPLEX<T>* polynomial,
                                        COMPLEX<T>* inverse_root_of_unity_table,
                                        int shared_index, int logm, int k,
                                        int outer_iteration_count, int N_power);

    template <typename T>
    __global__ void
    Special_InverseCore(COMPLEX<T>* polynomial, COMPLEX<T>* root_of_unity_table,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, COMPLEX<T> n_inverse,
                        bool not_last_kernel);
    template <typename T>
    __host__ void GPU_Special_FFT(COMPLEX<T>* device_inout,
                                  COMPLEX<T>* root_of_unity_table,
                                  fft_configuration<T> cfg, int batch_size);

    //      ---- FFNT ----
    /**
     * Efficient Negacyclic Convolution over R
     * Paper: https://eprint.iacr.org/2021/480
     *
     * Input:
     *   - f, g ∈ R^N for some even integer N.
     *
     * Precomputation:
     *   - Compute ω = exp(2πi / N) for j = -N/2 + 1, …, N/2 - 1.
     *
     * Output:
     *   - h ∈ R^N, where h is the negacyclic convolution of f and g (i.e., h =
     * f * g).
     *
     * Steps:
     *
     * 1. For j = 0 to (N/2 - 1):
     *      - Compute f'_j = f_j + i * f_(j + N/2)   // fold
     *      - Compute g'_j = g_j + i * g_(j + N/2)   // fold
     *
     * 2. For j = N/2 to (N - 1):
     *      - Compute f'_j = f_(j - N/2) - i * f_j
     *      - Compute g'_j = g_(j - N/2) - i * g_j
     *
     * 3. Compute F = F_(N/2)(f')     // Forward FFT of f' of size N/2
     *
     * 4. Compute G = F_(N/2)(g')     // Forward FFT of g' of size N/2
     *
     * 5. Compute H = F * G           // Pointwise multiplication
     *
     * 6. Compute h' = F_(N/2)^(-1)(H) // Inverse FFT of H of size N/2
     *
     * 7. For j = 0 to (N/2 - 1):
     *      - Set h'_j       = Re(h'_j)
     *      - Set h'_(j+N/2) = Im(h'_j)
     *
     * 8. Unfold the result to obtain h:
     *      - h_j       = Re(S(h'_j))
     *      - h_(j+N/2) = S(h'_(j+N/2))
     *
     * 9. Return h.
     */

    template <typename T>
    __global__ void
    ForwardFFNTCoreUnique(T* polynomial_in, COMPLEX<T>* polynomial_out,
                          COMPLEX<T>* twist_table,
                          COMPLEX<T>* root_of_unity_table, int shared_index,
                          int logm, int outer_iteration_count, int N_power);

    template <typename T>
    __global__ void
    ForwardFFNTCoreRegular(COMPLEX<T>* polynomial,
                           COMPLEX<T>* root_of_unity_table, int shared_index,
                           int logm, int outer_iteration_count, int N_power,
                           bool zero_padding, bool not_last_kernel);

    template <typename T>
    __global__ void InverseFFNTCoreRegular(
        COMPLEX<T>* polynomial, COMPLEX<T>* inverse_root_of_unity_table,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, bool NTT_mult, int offset2);

    template <typename T>
    __global__ void InverseFFNTCoreUnique(
        COMPLEX<T>* polynomial_in, T* polynomial_out, COMPLEX<T>* untwist_table,
        COMPLEX<T>* inverse_root_of_unity_table, int shared_index, int logm,
        int k, int outer_iteration_count, int N_power, COMPLEX<T> n_inverse);

    template <typename T>
    __host__ void
    GPU_FFNT(T* device_inout, COMPLEX<T>* device_temp, COMPLEX<T>* twist_table,
             COMPLEX<T>* root_of_unity_table, fft_configuration<T> cfg,
             int batch_size, bool multiplication);

    ///////////////////////////////////////////////////
    ///////////////////////////////////////////////////

    // FFT Kernel Parameters
    template <typename T> auto CreateForwardFFTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {12,
             {{8, 1, 64, 4, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 3, true},
              {1, 8, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 9, false}}},
            {13,
             {{16, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {1, 16, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 9, false}}},
            {14,
             {{32, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {1, 32, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 9, false}}},
            {15,
             {{64, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {1, 64, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 9, false}}},
            {16,
             {{128, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 7, true},
              {1, 128, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 7, 0, 9, false}}},
            {17,
             {{256, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {16, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 4, true},
              {1, 256, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 8, 0, 9, false}}},
            {18,
             {{512, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {32, 16, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true},
              {1, 512, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 9, 0, 9, false}}},
            {19,
             {{1024, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {32, 32, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 5, true},
              {1, 1024, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 10, 0, 9, false}}},
            {20,
             {{2048, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {64, 32, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true},
              {1, 2048, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 11, 0, 9, false}}},
            {21,
             {{4096, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {64, 64, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 6, true},
              {1, 4096, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 12, 0, 9, false}}},
            {22,
             {{8192, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {128, 64, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true},
              {1, 8192, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 13, 0, 9, false}}},
            {23,
             {{16384, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 7, true},
              {128, 128, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 7, 0, 7, true},
              {1, 16384, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 14, 0, 9,
               false}}},
            {24,
             {{32768, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {2048, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 4, true},
              {128, 256, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 8, 0, 7, true},
              {1, 32768, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 15, 0, 9,
               false}}}};
    }

    template <typename T> auto CreateInverseFFTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {12,
             {{1, 8, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 11, 3, 9, false},
              {8, 1, 64, 4, 512 * sizeof(COMPLEX<T>), 8, 2, 0, 3, true}}},
            {13,
             {{1, 16, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 12, 4, 9, false},
              {16, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {14,
             {{1, 32, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 13, 5, 9, false},
              {32, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {15,
             {{1, 64, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 14, 6, 9, false},
              {64, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {16,
             {{1, 128, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 15, 7, 9, false},
              {128, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true}}},
            {17,
             {{1, 256, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 16, 8, 9, false},
              {16, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 7, 4, 4, false},
              {256, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {18,
             {{1, 512, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 17, 9, 9, false},
              {32, 16, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 8, 4, 5, false},
              {512, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {19,
             {{1, 1024, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 18, 10, 9, false},
              {32, 32, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 9, 5, 5, false},
              {1024, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {20,
             {{1, 2048, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 19, 11, 9, false},
              {64, 32, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 10, 5, 6, false},
              {2048, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {21,
             {{1, 4096, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 20, 12, 9, false},
              {64, 64, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 11, 6, 6, false},
              {4096, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {22,
             {{1, 8192, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 21, 13, 9, false},
              {128, 64, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 12, 6, 7, false},
              {8192, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {23,
             {{1, 16384, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 22, 14, 9, false},
              {128, 128, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 13, 7, 7, false},
              {16384, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true}}},
            {24,
             {{1, 32768, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 23, 15, 9, false},
              {128, 256, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 14, 8, 7, false},
              {2048, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 7, 4, 4, false},
              {32768, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}}};
    }

    // Special FFT Kernel Parameters
    template <typename T> auto CreateForwardSpecialFFTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {11,
             {{1, 4, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 10, 2, 9, false},
              {4, 1, 128, 2, 512 * sizeof(COMPLEX<T>), 8, 1, 0, 2, true}}},
            {12,
             {{1, 8, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 11, 3, 9, false},
              {8, 1, 64, 4, 512 * sizeof(COMPLEX<T>), 8, 2, 0, 3, true}}},
            {13,
             {{1, 16, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 12, 4, 9, false},
              {16, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {14,
             {{1, 32, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 13, 5, 9, false},
              {32, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {15,
             {{1, 64, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 14, 6, 9, false},
              {64, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {16,
             {{1, 128, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 15, 7, 9, false},
              {128, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true}}},
            {17,
             {{1, 256, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 16, 8, 9, false},
              {16, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 7, 4, 4, false},
              {256, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {18,
             {{1, 512, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 17, 9, 9, false},
              {32, 16, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 8, 4, 5, false},
              {512, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {19,
             {{1, 1024, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 18, 10, 9, false},
              {32, 32, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 9, 5, 5, false},
              {1024, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {20,
             {{1, 2048, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 19, 11, 9, false},
              {64, 32, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 10, 5, 6, false},
              {2048, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {21,
             {{1, 4096, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 20, 12, 9, false},
              {64, 64, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 11, 6, 6, false},
              {4096, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {22,
             {{1, 8192, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 21, 13, 9, false},
              {128, 64, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 12, 6, 7, false},
              {8192, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {23,
             {{1, 16384, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 22, 14, 9, false},
              {128, 128, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 13, 7, 7, false},
              {16384, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true}}},
            {24,
             {{1, 32768, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 23, 15, 9, false},
              {128, 256, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 14, 8, 7, false},
              {2048, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 7, 4, 4, false},
              {32768, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}}};
    }

    template <typename T> auto CreateInverseSpecialFFTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {11,
             {{4, 1, 128, 2, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 2, true},
              {1, 4, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 2, 0, 9, false}}},
            {12,
             {{8, 1, 64, 4, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 3, true},
              {1, 8, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 9, false}}},
            {13,
             {{16, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {1, 16, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 9, false}}},
            {14,
             {{32, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {1, 32, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 9, false}}},
            {15,
             {{64, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {1, 64, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 9, false}}},
            {16,
             {{128, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 7, true},
              {1, 128, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 7, 0, 9, false}}},
            {17,
             {{256, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {16, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 4, true},
              {1, 256, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 8, 0, 9, false}}},
            {18,
             {{512, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {32, 16, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true},
              {1, 512, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 9, 0, 9, false}}},
            {19,
             {{1024, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {32, 32, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 5, true},
              {1, 1024, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 10, 0, 9, false}}},
            {20,
             {{2048, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {64, 32, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true},
              {1, 2048, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 11, 0, 9, false}}},
            {21,
             {{4096, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {64, 64, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 6, true},
              {1, 4096, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 12, 0, 9, false}}},
            {22,
             {{8192, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {128, 64, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true},
              {1, 8192, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 13, 0, 9, false}}},
            {23,
             {{16384, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 7, true},
              {128, 128, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 7, 0, 7, true},
              {1, 16384, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 14, 0, 9,
               false}}},
            {24,
             {{32768, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {2048, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 4, true},
              {128, 256, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 8, 0, 7, true},
              {1, 32768, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 15, 0, 9,
               false}}}};
    }

    // FFNT Kernel Parameters
    template <typename T> auto CreateForwardFFNTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {12,
             {{4, 1, 128, 2, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 2, false},
              {1, 4, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 2, 0, 9, true}}},
            {13,
             {{8, 1, 64, 4, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 3, true},
              {1, 8, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 9, false}}},
            {14,
             {{16, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {1, 16, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 9, false}}},
            {15,
             {{32, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {1, 32, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 9, false}}},
            {16,
             {{64, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {1, 64, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 9, false}}},
            {17,
             {{128, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 7, true},
              {1, 128, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 7, 0, 9, false}}},
            {18,
             {{256, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {16, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 4, true},
              {1, 256, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 8, 0, 9, false}}},
            {19,
             {{512, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 4, true},
              {32, 16, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true},
              {1, 512, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 9, 0, 9, false}}},
            {20,
             {{1024, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {32, 32, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 5, true},
              {1, 1024, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 10, 0, 9, false}}},
            {21,
             {{2048, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 5, true},
              {64, 32, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true},
              {1, 2048, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 11, 0, 9, false}}},
            {22,
             {{4096, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {64, 64, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 6, true},
              {1, 4096, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 12, 0, 9, false}}},
            {23,
             {{8192, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 6, true},
              {128, 64, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true},
              {1, 8192, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 13, 0, 9, false}}},
            {24,
             {{16384, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 0, 0, 7, true},
              {128, 128, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 7, 0, 7, true},
              {1, 16384, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 14, 0, 9,
               false}}}};
    }

    template <typename T> auto CreateInverseFFNTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {12,
             {{1, 4, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 10, 2, 9, false},
              {4, 1, 128, 2, 512 * sizeof(COMPLEX<T>), 8, 1, 0, 2, true}}},
            {13,
             {{1, 8, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 11, 3, 9, false},
              {8, 1, 64, 4, 512 * sizeof(COMPLEX<T>), 8, 2, 0, 3, true}}},
            {14,
             {{1, 16, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 12, 4, 9, false},
              {16, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {15,
             {{1, 32, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 13, 5, 9, false},
              {32, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {16,
             {{1, 64, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 14, 6, 9, false},
              {64, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {17,
             {{1, 128, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 15, 7, 9, false},
              {128, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true}}},
            {18,
             {{1, 256, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 16, 8, 9, false},
              {16, 16, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 7, 4, 4, false},
              {256, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {19,
             {{1, 512, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 17, 9, 9, false},
              {32, 16, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 8, 4, 5, false},
              {512, 1, 32, 8, 512 * sizeof(COMPLEX<T>), 8, 3, 0, 4, true}}},
            {20,
             {{1, 1024, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 18, 10, 9, false},
              {32, 32, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 9, 5, 5, false},
              {1024, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {21,
             {{1, 2048, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 19, 11, 9, false},
              {64, 32, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 10, 5, 6, false},
              {2048, 1, 16, 16, 512 * sizeof(COMPLEX<T>), 8, 4, 0, 5, true}}},
            {22,
             {{1, 4096, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 20, 12, 9, false},
              {64, 64, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 11, 6, 6, false},
              {4096, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {23,
             {{1, 8192, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 21, 13, 9, false},
              {128, 64, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 12, 6, 7, false},
              {8192, 1, 8, 32, 512 * sizeof(COMPLEX<T>), 8, 5, 0, 6, true}}},
            {24,
             {{1, 16384, 256, 1, 512 * sizeof(COMPLEX<T>), 8, 22, 14, 9, false},
              {128, 128, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 13, 7, 7, false},
              {16384, 1, 4, 64, 512 * sizeof(COMPLEX<T>), 8, 6, 0, 7, true}}}};
    }

} // namespace gpufft
#endif // FFT_CORE_H
