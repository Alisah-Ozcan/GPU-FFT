// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef GPUFFT_FFT_DEVICE_H
#define GPUFFT_FFT_DEVICE_H

#include "complex.cuh"
#include "fft.cuh"

// Building blocks of the FFT kernels. A stage loads two values per thread
// from global to shared memory, runs a group of butterfly levels in shared
// memory and stores the two values back. The kernels in fft.cu combine one
// load, one execute and one store step.

namespace gpufft
{
    // ---------------- Stage plan ----------------

    // Launch parameters of one stage kernel.
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

        // The inverse plans store the last_kernel flag here.
        bool not_last_kernel;
    };

    constexpr int kStageThreads = 256;

    // Sizes up to 2^kLowRingMaxLogN run in a single low ring kernel (see
    // LowRingThread); 2^10 runs in a single stage kernel of 512 threads.
    constexpr int kLowRingMaxLogN = 9;

    constexpr int kMaxStages = 5;

    // Threads per stage kernel of an FFT of size 2^log_n.
    constexpr int default_stage_threads(int log_n)
    {
        return (log_n == 10) ? 512 : kStageThreads;
    }

    // The kernels of one transform, in launch order, with the grid axes
    // flag of each kernel.
    struct StagePlan
    {
        int count;
        KernelConfig stage[kMaxStages];
        bool swap_grid_axes[kMaxStages];
    };

    // The kernels of an FFT of size 2^log_n > 2^kLowRingMaxLogN. Every kernel
    // has threads = 2^s threads (2^s <= 2^log_n / 2, default_stage_threads if
    // threads is 0) and each thread keeps two values in shared memory. The last
    // forward kernel runs s + 1 levels with block (2^s, 1). The levels before
    // it are split as evenly as possible, smaller ones first, over as few
    // kernels as possible with at most s - 1 levels each; a kernel with L
    // levels has block (2^(s - L + 1), 2^(L - 1)), so its rows read at least
    // four adjacent values. A kernel swaps the grid axes when its 2^logm groups
    // do not fit grid y. The inverse kernels mirror the forward ones. The
    // Special FFT runs the inverse plan forward and the forward plan inverse;
    // the FFNT of 2^n real values runs the plans of 2^(n - 1).
    template <typename T>
    constexpr StagePlan stage_plan(type direction, int log_n, int threads = 0)
    {
        if (threads == 0)
        {
            threads = default_stage_threads(log_n);
        }

        int s = 0;
        while ((1 << s) < threads)
        {
            s++;
        }

        StagePlan plan = {};
        const int early_levels = log_n - (s + 1);
        const int early_count = (early_levels + s - 2) / (s - 1);
        plan.count = early_count + 1;

        KernelConfig forward[kMaxStages] = {};
        bool swap[kMaxStages] = {};
        int logm = 0;
        for (int i = 0; i < plan.count; i++)
        {
            const bool last = (i == early_count);
            const int levels =
                last ? s + 1
                     : early_levels / early_count +
                           (i >= early_count - early_levels % early_count);
            const int block_y = last ? 1 : 1 << (levels - 1);
            const int block_x = threads / block_y;
            const long long columns =
                last ? 1 : (1LL << (log_n - logm - levels)) / block_x;
            const long long groups = 1LL << logm;
            swap[i] = groups > 65535;
            forward[i] = KernelConfig{int(swap[i] ? groups : columns),
                                      int(swap[i] ? columns : groups),
                                      block_x,
                                      block_y,
                                      size_t(2 * threads) * sizeof(COMPLEX<T>),
                                      s,
                                      logm,
                                      0,
                                      levels,
                                      !last};
            logm += levels;
        }

        for (int i = 0; i < plan.count; i++)
        {
            const int f = (direction == FORWARD) ? i : plan.count - 1 - i;
            plan.swap_grid_axes[i] = swap[f];
            if (direction == FORWARD)
            {
                plan.stage[i] = forward[f];
            }
            else
            {
                const KernelConfig& c = forward[f];
                plan.stage[i] =
                    KernelConfig{c.griddim_x,
                                 c.griddim_y,
                                 c.blockdim_x,
                                 c.blockdim_y,
                                 c.shared_memory,
                                 c.shared_index,
                                 c.logm + c.outer_iteration_count - 1,
                                 c.logm,
                                 c.outer_iteration_count,
                                 i + 1 == plan.count};
            }
        }
        return plan;
    }

    // Per-thread state of one stage: thread (x, y) owns the shared memory
    // entries shared_address and shared_address + blockDim.x * blockDim.y.
    // In global memory they are at local_address and local_address + offset
    // of the polynomial (local_address does not include the polynomial of
    // the block, see poly_address).
    struct StageThread
    {
        location_t offset;
        location_t local_address;
        location_t omega_address;
        location_t shared_address;
    };

    // The distance between the two values of a thread is
    // 2^(N_power - level - 1), with level = logm for the forward and k for
    // the inverse transform.
    __device__ __forceinline__ StageThread
    StageThreadState(int level, int outer_iteration_count, int N_power)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;

        const location_t offset = 1 << (N_power - level - 1);

        const location_t local_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset);

        const location_t omega_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        const location_t shared_address = (idx_x + (idx_y * blockDim.x));

        return StageThread{offset, local_address, omega_address,
                           shared_address};
    }

    // Global address of the thread's first value in the polynomial of the
    // block, for polynomials of 2^log_size values.
    __device__ __forceinline__ location_t
    poly_address(const StageThread& thread, int log_size)
    {
        return thread.local_address + (location_t) (blockIdx.z << log_size);
    }

    template <typename T> __device__ __forceinline__ COMPLEX<T>* stage_shared()
    {
        extern __shared__ char shared_memory_typed[];
        return reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);
    }

    // The low ring kernels transform blockDim.y polynomials of 2^N_power
    // values per block; thread (x, y) owns values x and x + N/2 of polynomial
    // blockIdx.x * blockDim.y + y, which are entries shared_address0 and
    // shared_address1 of row y in shared memory. Rows of polynomials beyond
    // total_batch read polynomial 0 and store nothing.
    struct LowRingThread
    {
        bool active;
        location_t global_address;
        location_t half_n;
        int shared_address0;
        int shared_address1;
    };

    __device__ __forceinline__ LowRingThread LowRingThreadState(int N_power,
                                                                int total_batch)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int batch_index = (blockIdx.x * blockDim.y) + idx_y;
        const bool active = (batch_index < total_batch);

        const int half_n = 1 << (N_power - 1);
        const int shared_address0 = (idx_y << N_power) + idx_x;
        const location_t global_address =
            idx_x + ((location_t) (active ? batch_index : 0) << N_power);

        return LowRingThread{active, global_address, (location_t) half_n,
                             shared_address0, shared_address0 + half_n};
    }

    // The state of the stage execute steps in a low ring kernel. With shared
    // address x + y * N/2, the butterflies of thread (x, y) stay in row y
    // (entries y * N to y * N + N - 1), since no butterfly distance exceeds
    // N/2; the roots depend on x only.
    __device__ __forceinline__ StageThread LowRingExecuteThread()
    {
        return StageThread{0, 0, location_t(threadIdx.x),
                           location_t(threadIdx.x + threadIdx.y * blockDim.x)};
    }

    // ---------------- Load ----------------

    template <typename T>
    __device__ __forceinline__ void
    StageLoad(const StageThread& thread, COMPLEX<T>* shared_memory,
              const COMPLEX<T>* polynomial, location_t global_address)
    {
        shared_memory[thread.shared_address] = polynomial[global_address];
        shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] =
            polynomial[global_address + thread.offset];
    }

    // Real input, the imaginary parts are zero.
    template <typename T>
    __device__ __forceinline__ void
    StageLoad(const StageThread& thread, COMPLEX<T>* shared_memory,
              const T* input, location_t global_address)
    {
        shared_memory[thread.shared_address] =
            COMPLEX<T>(input[global_address]);
        shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] =
            COMPLEX<T>(input[global_address + thread.offset]);
    }

    // Pointwise product with the second operand at polynomial + offset2.
    template <typename T>
    __device__ __forceinline__ void
    StageLoadProduct(const StageThread& thread, COMPLEX<T>* shared_memory,
                     const COMPLEX<T>* polynomial, location_t global_address,
                     int offset2)
    {
        COMPLEX<T> a0 = polynomial[global_address];
        COMPLEX<T> a1 = polynomial[global_address + thread.offset];

        COMPLEX<T> b0 = polynomial[global_address + offset2];
        COMPLEX<T> b1 = polynomial[global_address + thread.offset + offset2];

        shared_memory[thread.shared_address] = a0 * b0;
        shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] =
            a1 * b1;
    }

    // FFNT input: fold f_j + i f_(j + n/2) of 2^(N_power + 1) real values,
    // then twist.
    template <typename T>
    __device__ __forceinline__ void
    StageLoadFoldTwist(const StageThread& thread, COMPLEX<T>* shared_memory,
                       const T* polynomial_in, location_t global_address_in,
                       const COMPLEX<T>* twist_table, int N_power)
    {
        T real_part1 = polynomial_in[global_address_in];
        T imaginary_part1 = polynomial_in[global_address_in + (1 << N_power)];
        COMPLEX<T> part1(real_part1, imaginary_part1); // fold
        COMPLEX<T> twist_part1 = twist_table[thread.local_address];
        part1 = part1 * twist_part1; // twist

        T real_part2 = polynomial_in[global_address_in + thread.offset];
        T imaginary_part2 =
            polynomial_in[global_address_in + thread.offset + (1 << N_power)];
        COMPLEX<T> part2(real_part2, imaginary_part2); // fold
        COMPLEX<T> twist_part2 =
            twist_table[thread.local_address + thread.offset];
        part2 = part2 * twist_part2; // twist

        shared_memory[thread.shared_address] = part1;
        shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] =
            part2;
    }

    template <typename T>
    __device__ __forceinline__ void LowRingLoad(const LowRingThread& thread,
                                                COMPLEX<T>* shared_memory,
                                                const COMPLEX<T>* polynomial)
    {
        shared_memory[thread.shared_address0] =
            polynomial[thread.global_address];
        shared_memory[thread.shared_address1] =
            polynomial[thread.global_address + thread.half_n];
    }

    // Real input, the imaginary parts are zero.
    template <typename T>
    __device__ __forceinline__ void LowRingLoad(const LowRingThread& thread,
                                                COMPLEX<T>* shared_memory,
                                                const T* input)
    {
        shared_memory[thread.shared_address0] =
            COMPLEX<T>(input[thread.global_address]);
        shared_memory[thread.shared_address1] =
            COMPLEX<T>(input[thread.global_address + thread.half_n]);
    }

    // Pointwise product with the second operand at polynomial + offset2.
    template <typename T>
    __device__ __forceinline__ void
    LowRingLoadProduct(const LowRingThread& thread, COMPLEX<T>* shared_memory,
                       const COMPLEX<T>* polynomial, int offset2)
    {
        COMPLEX<T> a0 = polynomial[thread.global_address];
        COMPLEX<T> a1 = polynomial[thread.global_address + thread.half_n];

        COMPLEX<T> b0 = polynomial[thread.global_address + offset2];
        COMPLEX<T> b1 =
            polynomial[thread.global_address + thread.half_n + offset2];

        shared_memory[thread.shared_address0] = a0 * b0;
        shared_memory[thread.shared_address1] = a1 * b1;
    }

    // ---------------- Execute ----------------

    struct CooleyTukey
    {
        template <typename T>
        __device__ __forceinline__ static void
        apply(COMPLEX<T>& U, COMPLEX<T>& V, COMPLEX<T>& root)
        {
            CooleyTukeyUnit(U, V, root);
        }
    };

    struct GentlemanSande
    {
        template <typename T>
        __device__ __forceinline__ static void
        apply(COMPLEX<T>& U, COMPLEX<T>& V, COMPLEX<T>& root)
        {
            GentlemanSandeUnit(U, V, root);
        }
    };

    // Root index of the butterflies of one level. Increasing is the
    // direction of the butterfly distance from level to level.

    // X^N - 1: roots of the merge FFT.
    template <bool Increasing> struct CyclicRoots
    {
        int t_2;

        __device__ __forceinline__ location_t index(location_t omega) const
        {
            return omega >> t_2;
        }

        __device__ __forceinline__ void advance()
        {
            if constexpr (Increasing)
                t_2 += 1;
            else
                t_2 -= 1;
        }
    };

    // X^N - 1 or X^N + 1, chosen at run time.
    template <bool Increasing> struct PolyRoots
    {
        bool cyclic;
        int t_2;
        location_t m;

        __device__ __forceinline__ location_t index(location_t omega) const
        {
            if (cyclic)
            { // X_N_minus
                return (omega >> t_2);
            }
            else
            { // X_N_plus
                return m + (omega >> t_2);
            }
        }

        __device__ __forceinline__ void advance()
        {
            if constexpr (Increasing)
            {
                t_2 += 1;
                m >>= 1;
            }
            else
            {
                t_2 -= 1;
                m <<= 1;
            }
        }
    };

    // Special FFT: entry m + j of the table for butterfly j of a sub-array
    // of 2m values.
    template <bool Increasing> struct SpecialRoots
    {
        location_t m;

        __device__ __forceinline__ location_t index(location_t omega) const
        {
            return m + (omega & (m - 1));
        }

        __device__ __forceinline__ void advance()
        {
            if constexpr (Increasing)
                m <<= 1;
            else
                m >>= 1;
        }
    };

    // Levels with decreasing butterfly distance 2^t_start, 2^(t_start - 1),
    // ... In the last stage the final warp_levels levels (distance at most
    // 32) stay within a warp: only the sync_levels levels before them end
    // with a block barrier. Ends with a block barrier in the last stage, so
    // the result can be read by any thread of the block.
    template <typename Butterfly, typename T, typename Roots>
    __device__ __forceinline__ void
    DecreasingStageExecute(const StageThread& thread, COMPLEX<T>* shared_memory,
                           COMPLEX<T>* root_table, Roots roots, int t_start,
                           int loops, bool not_last_kernel, int sync_levels,
                           int warp_levels = 6)
    {
        const location_t shared_addresss = thread.shared_address;
        const location_t omega_addresss = thread.omega_address;

        int t_ = t_start;
        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
        if (not_last_kernel)
        {
#pragma unroll
            for (int lp = 0; lp < loops; lp++)
            {
                current_root_index = roots.index(omega_addresss);

                Butterfly::apply(shared_memory[in_shared_address],
                                 shared_memory[in_shared_address + t],
                                 root_table[current_root_index]);

                t = t >> 1;
                roots.advance();
                t_ -= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < sync_levels; lp++)
            {
                current_root_index = roots.index(omega_addresss);

                Butterfly::apply(shared_memory[in_shared_address],
                                 shared_memory[in_shared_address + t],
                                 root_table[current_root_index]);

                t = t >> 1;
                roots.advance();
                t_ -= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }

#pragma unroll
            for (int lp = 0; lp < warp_levels; lp++)
            {
                current_root_index = roots.index(omega_addresss);

                Butterfly::apply(shared_memory[in_shared_address],
                                 shared_memory[in_shared_address + t],
                                 root_table[current_root_index]);

                t = t >> 1;
                roots.advance();
                t_ -= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncwarp();
            }
            __syncthreads();
        }
    }

    // Levels with increasing butterfly distance 2^t_start, 2^(t_start + 1),
    // ... Every level starts with a block barrier; ends with a block barrier.
    template <typename Butterfly, typename T, typename Roots>
    __device__ __forceinline__ void
    IncreasingStageExecute(const StageThread& thread, COMPLEX<T>* shared_memory,
                           COMPLEX<T>* root_table, Roots roots, int t_start,
                           int loops)
    {
        const location_t shared_addresss = thread.shared_address;
        const location_t omega_addresss = thread.omega_address;

        int t_ = t_start;
        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();

            current_root_index = roots.index(omega_addresss);

            Butterfly::apply(shared_memory[in_shared_address],
                             shared_memory[in_shared_address + t],
                             root_table[current_root_index]);

            t = t << 1;
            roots.advance();
            t_ += 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    // Forward FFT of every row of a low ring kernel. The first level only
    // reads the thread's own entries, so the load needs no barrier; rows
    // narrower than a warp run all levels within the warp. Ends with a block
    // barrier.
    template <int LogN, typename T>
    __device__ __forceinline__ void
    ForwardLowRingExecute(COMPLEX<T>* shared_memory,
                          COMPLEX<T>* root_of_unity_table,
                          bool reduction_poly_check)
    {
        DecreasingStageExecute<CooleyTukey>(
            LowRingExecuteThread(), shared_memory, root_of_unity_table,
            PolyRoots<false>{reduction_poly_check, LogN - 1, 1}, LogN - 1, LogN,
            false, (LogN > 6) ? LogN - 6 : 0, (LogN > 6) ? 6 : LogN);
    }

    // Inverse FFT of every row of a low ring kernel, without the scaling by
    // n_inverse. Ends with a block barrier.
    template <int LogN, typename T>
    __device__ __forceinline__ void
    InverseLowRingExecute(COMPLEX<T>* shared_memory,
                          COMPLEX<T>* inverse_root_of_unity_table,
                          bool reduction_poly_check)
    {
        IncreasingStageExecute<GentlemanSande>(
            LowRingExecuteThread(), shared_memory, inverse_root_of_unity_table,
            PolyRoots<true>{reduction_poly_check, 0,
                            (location_t) 1 << (LogN - 1)},
            0, LogN);
    }

    // ---------------- Store ----------------

    template <typename T>
    __device__ __forceinline__ void
    StageStore(const StageThread& thread, const COMPLEX<T>* shared_memory,
               COMPLEX<T>* polynomial, location_t global_address)
    {
        polynomial[global_address] = shared_memory[thread.shared_address];
        polynomial[global_address + thread.offset] =
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)];
    }

    template <typename T>
    __device__ __forceinline__ void
    StageStoreScaled(const StageThread& thread, const COMPLEX<T>* shared_memory,
                     COMPLEX<T>* polynomial, location_t global_address,
                     COMPLEX<T> n_inverse)
    {
        polynomial[global_address] =
            shared_memory[thread.shared_address] * n_inverse;
        polynomial[global_address + thread.offset] =
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] *
            n_inverse;
    }

    // Real output: the real parts.
    template <typename T>
    __device__ __forceinline__ void
    StageStore(const StageThread& thread, const COMPLEX<T>* shared_memory,
               T* output, location_t global_address)
    {
        output[global_address] = shared_memory[thread.shared_address].real();
        output[global_address + thread.offset] =
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)]
                .real();
    }

    template <typename T>
    __device__ __forceinline__ void
    StageStoreScaled(const StageThread& thread, const COMPLEX<T>* shared_memory,
                     T* output, location_t global_address, COMPLEX<T> n_inverse)
    {
        output[global_address] =
            (shared_memory[thread.shared_address] * n_inverse).real();
        output[global_address + thread.offset] =
            (shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] *
             n_inverse)
                .real();
    }

    template <typename T>
    __device__ __forceinline__ void
    LowRingStore(const LowRingThread& thread, const COMPLEX<T>* shared_memory,
                 COMPLEX<T>* polynomial)
    {
        if (thread.active)
        {
            polynomial[thread.global_address] =
                shared_memory[thread.shared_address0];
            polynomial[thread.global_address + thread.half_n] =
                shared_memory[thread.shared_address1];
        }
    }

    template <typename T>
    __device__ __forceinline__ void
    LowRingStoreScaled(const LowRingThread& thread,
                       const COMPLEX<T>* shared_memory, COMPLEX<T>* polynomial,
                       COMPLEX<T> n_inverse)
    {
        if (thread.active)
        {
            polynomial[thread.global_address] =
                shared_memory[thread.shared_address0] * n_inverse;
            polynomial[thread.global_address + thread.half_n] =
                shared_memory[thread.shared_address1] * n_inverse;
        }
    }

    // Real output: the real parts.
    template <typename T>
    __device__ __forceinline__ void
    LowRingStore(const LowRingThread& thread, const COMPLEX<T>* shared_memory,
                 T* output)
    {
        if (thread.active)
        {
            output[thread.global_address] =
                shared_memory[thread.shared_address0].real();
            output[thread.global_address + thread.half_n] =
                shared_memory[thread.shared_address1].real();
        }
    }

    template <typename T>
    __device__ __forceinline__ void
    LowRingStoreScaled(const LowRingThread& thread,
                       const COMPLEX<T>* shared_memory, T* output,
                       COMPLEX<T> n_inverse)
    {
        if (thread.active)
        {
            output[thread.global_address] =
                (shared_memory[thread.shared_address0] * n_inverse).real();
            output[thread.global_address + thread.half_n] =
                (shared_memory[thread.shared_address1] * n_inverse).real();
        }
    }

    // FFNT output: times n_inverse, untwist, then unfold to 2^(N_power + 1)
    // real values.
    template <typename T>
    __device__ __forceinline__ void StageStoreUntwistUnfold(
        const StageThread& thread, const COMPLEX<T>* shared_memory,
        T* polynomial_out, location_t global_address_out,
        const COMPLEX<T>* untwist_table, COMPLEX<T> n_inverse, int N_power)
    {
        COMPLEX<T> part1 = shared_memory[thread.shared_address] * n_inverse;
        COMPLEX<T> part2 =
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] *
            n_inverse;

        COMPLEX<T> untwist_part1 = untwist_table[thread.local_address];
        part1 = part1 * untwist_part1; // untwist

        COMPLEX<T> untwist_part2 =
            untwist_table[thread.local_address + thread.offset];
        part2 = part2 * untwist_part2; // untwist

        // unfold
        polynomial_out[global_address_out] = part1.real();
        polynomial_out[global_address_out + (1 << N_power)] = part1.imag();

        polynomial_out[global_address_out + thread.offset] = part2.real();
        polynomial_out[global_address_out + (1 << N_power) + thread.offset] =
            part2.imag();
    }

} // namespace gpufft
#endif // GPUFFT_FFT_DEVICE_H
