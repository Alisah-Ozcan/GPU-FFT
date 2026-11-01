// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef FFT_CORE_H
#define FFT_CORE_H

#include "common_fft.cuh"
#include "complex.cuh"
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
        COMPLEX<T> mod_inverse;
        cudaStream_t stream;
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
                int N_power, bool not_last_kernel, bool reduction_poly_check);

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
                int outer_iteration_count, int N_power, bool not_last_kernel,
                bool reduction_poly_check);

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
    __global__ void ForwardFFNTCoreRegular(COMPLEX<T>* polynomial,
                                           COMPLEX<T>* root_of_unity_table,
                                           int shared_index, int logm,
                                           int outer_iteration_count,
                                           int N_power, bool not_last_kernel);

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

} // namespace gpufft
#endif // FFT_CORE_H
