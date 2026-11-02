// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef GPUFFT_TWIDDLES_H
#define GPUFFT_TWIDDLES_H

#include "complex.cuh"
#include "fft.cuh"
#include "fft_device.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// Root tables of GPU_FFT, GPU_FFNT and GPU_Special_FFT, built on the host
// (make_*_host) or by a kernel in device memory (make_*). Every entry is
// exp(pi i x) for a rational x; the host computes it in double precision and
// rounds it to T, the kernel uses sincospi, so both agree up to rounding.
// The inverse tables hold the conjugates.

namespace gpufft
{
    // The order of the entries of a root table.
    enum class TwiddleOrder
    {
        // Entry i: x = bitrev(i) / 2^log_size, bitrev over log_size bits.
        BitReversed,
        // Entry i: x = i / 2^log_size.
        Natural,
        // Entry m / 2 + j for m = 2, 4, ...: x = (5^j mod 4m) / (2m);
        // entry 0 is zero.
        Special
    };

    // 5^exponent mod modulus, for a power of two modulus.
    __host__ __device__ inline unsigned
    power_of_five(unsigned exponent, unsigned long long modulus)
    {
        unsigned long long result = 1;
        unsigned long long base = 5 % modulus;
        while (exponent > 0)
        {
            if (exponent & 1)
            {
                result = (result * base) & (modulus - 1);
            }
            base = (base * base) & (modulus - 1);
            exponent >>= 1;
        }
        return unsigned(result & (modulus - 1));
    }

    // x of entry i is numerator / denominator.
    __host__ __device__ inline void twiddle_fraction(TwiddleOrder order, int i,
                                                     int log_size,
                                                     double& numerator,
                                                     double& denominator)
    {
        if (order == TwiddleOrder::Special)
        {
            int log_m = 0;
            while ((1 << log_m) <= i)
            {
                log_m++;
            }
            const unsigned m = 1u << log_m;
            numerator = power_of_five(unsigned(i) - m / 2, 4 * m);
            denominator = 2.0 * m;
            return;
        }
        int index = i;
        if (order == TwiddleOrder::BitReversed)
        {
            index = 0;
            for (int b = 0; b < log_size; b++)
            {
                index |= ((i >> b) & 1) << (log_size - 1 - b);
            }
        }
        numerator = index;
        denominator = double(1 << log_size);
    }

    template <typename T>
    __global__ void TwiddleTableKernel(COMPLEX<T>* table, int count,
                                       TwiddleOrder order, int log_size,
                                       bool inverse)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
        {
            return;
        }
        if (order == TwiddleOrder::Special && i == 0)
        {
            table[i] = COMPLEX<T>(0.0, 0.0);
            return;
        }
        double numerator;
        double denominator;
        twiddle_fraction(order, i, log_size, numerator, denominator);
        double sine;
        double cosine;
        sincospi(numerator / denominator, &sine, &cosine);
        table[i] = COMPLEX<T>(cosine, inverse ? -sine : sine);
    }

    template <typename T>
    std::vector<COMPLEX<T>> make_twiddle_table_host(int count,
                                                    TwiddleOrder order,
                                                    int log_size, bool inverse)
    {
        std::vector<COMPLEX<T>> table(count);
        for (int i = 0; i < count; i++)
        {
            if (order == TwiddleOrder::Special && i == 0)
            {
                table[i] = COMPLEX<T>(0.0, 0.0);
                continue;
            }
            double numerator;
            double denominator;
            twiddle_fraction(order, i, log_size, numerator, denominator);
            const double angle = M_PI * numerator / denominator;
            const double sine = std::sin(angle);
            table[i] = COMPLEX<T>(std::cos(angle), inverse ? -sine : sine);
        }
        return table;
    }

    template <typename T>
    void make_twiddle_table(COMPLEX<T>* table, int count, TwiddleOrder order,
                            int log_size, bool inverse, cudaStream_t stream)
    {
        constexpr int threads = 256;
        TwiddleTableKernel<<<(count + threads - 1) / threads, threads, 0,
                             stream>>>(table, count, order, log_size, inverse);
        GPUFFT_CUDA_CHECK(cudaGetLastError());
    }

    inline void check_twiddle_log_size(int log_size, int min_log_size,
                                       const char* name)
    {
        if (log_size < min_log_size || log_size > kMaxLogN)
        {
            throw std::invalid_argument(std::string(name) + " must be in [" +
                                        std::to_string(min_log_size) + ", " +
                                        std::to_string(kMaxLogN) + "].");
        }
    }

    // ---------------- GPU_FFT ----------------

    // Entries of the root table of GPU_FFT for size 2^n_power: 2^n_power for
    // X^N + 1, 2^(n_power - 1) for X^N - 1.
    inline int fft_twiddle_table_size(int n_power, ReductionPolynomial poly)
    {
        return (poly == X_N_plus) ? (1 << n_power) : (1 << (n_power - 1));
    }

    // The root table of GPU_FFT: the powers of exp(2 pi i / N) (X^N - 1) or
    // of exp(pi i / N) (X^N + 1) in bit-reversed order, conjugated for the
    // inverse.
    template <typename T>
    std::vector<COMPLEX<T>> make_fft_twiddles_host(int n_power,
                                                   ReductionPolynomial poly,
                                                   type direction)
    {
        check_twiddle_log_size(n_power, 1, "n_power");
        const int size = fft_twiddle_table_size(n_power, poly);
        return make_twiddle_table_host<T>(size, TwiddleOrder::BitReversed,
                                          log2_of_power_of_two(size),
                                          direction == INVERSE);
    }

    // The same table, written by a kernel on stream to table, which holds
    // fft_twiddle_table_size(n_power, poly) entries in device memory.
    template <typename T>
    void make_fft_twiddles(COMPLEX<T>* table, int n_power,
                           ReductionPolynomial poly, type direction,
                           cudaStream_t stream = 0)
    {
        check_twiddle_log_size(n_power, 1, "n_power");
        const int size = fft_twiddle_table_size(n_power, poly);
        make_twiddle_table(table, size, TwiddleOrder::BitReversed,
                           log2_of_power_of_two(size), direction == INVERSE,
                           stream);
    }

    // ---------------- GPU_FFNT ----------------

    // The root table of GPU_FFNT for 2^n_power real values, the X^N - 1
    // table of an FFT of size 2^(n_power - 1): 2^(n_power - 2) entries.
    template <typename T>
    std::vector<COMPLEX<T>> make_ffnt_twiddles_host(int n_power, type direction)
    {
        check_twiddle_log_size(n_power, 2, "n_power");
        return make_fft_twiddles_host<T>(n_power - 1, X_N_minus, direction);
    }

    template <typename T>
    void make_ffnt_twiddles(COMPLEX<T>* table, int n_power, type direction,
                            cudaStream_t stream = 0)
    {
        check_twiddle_log_size(n_power, 2, "n_power");
        make_fft_twiddles(table, n_power - 1, X_N_minus, direction, stream);
    }

    // The twist table (forward) or untwist table (inverse) of GPU_FFNT for
    // 2^n_power real values: 2^(n_power - 1) entries, entry j is
    // exp(pi i j / 2^n_power), conjugated for the untwist table.
    template <typename T>
    std::vector<COMPLEX<T>> make_ffnt_twist_host(int n_power, type direction)
    {
        check_twiddle_log_size(n_power, 2, "n_power");
        return make_twiddle_table_host<T>(1 << (n_power - 1),
                                          TwiddleOrder::Natural, n_power,
                                          direction == INVERSE);
    }

    template <typename T>
    void make_ffnt_twist(COMPLEX<T>* table, int n_power, type direction,
                         cudaStream_t stream = 0)
    {
        check_twiddle_log_size(n_power, 2, "n_power");
        make_twiddle_table(table, 1 << (n_power - 1), TwiddleOrder::Natural,
                           n_power, direction == INVERSE, stream);
    }

    // ---------------- GPU_Special_FFT ----------------

    // The root table of GPU_Special_FFT for 2^log_slots slots (the CKKS
    // canonical embedding): 2^log_slots entries; entry m / 2 + j of the
    // level with sub-arrays of m values is exp(pi i (5^j mod 4m) / (2m)),
    // conjugated for the inverse, and entry 0 is zero. The entries do not
    // depend on log_slots, so the table also serves fewer slots.
    template <typename T>
    std::vector<COMPLEX<T>> make_special_fft_twiddles_host(int log_slots,
                                                           type direction)
    {
        check_twiddle_log_size(log_slots, 1, "log_slots");
        return make_twiddle_table_host<T>(1 << log_slots, TwiddleOrder::Special,
                                          log_slots, direction == INVERSE);
    }

    template <typename T>
    void make_special_fft_twiddles(COMPLEX<T>* table, int log_slots,
                                   type direction, cudaStream_t stream = 0)
    {
        check_twiddle_log_size(log_slots, 1, "log_slots");
        make_twiddle_table(table, 1 << log_slots, TwiddleOrder::Special,
                           log_slots, direction == INVERSE, stream);
    }

} // namespace gpufft

#endif // GPUFFT_TWIDDLES_H
