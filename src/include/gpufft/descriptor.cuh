// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef GPUFFT_DESCRIPTOR_H
#define GPUFFT_DESCRIPTOR_H

#include "gpufft/config.cuh"
#include "complex.cuh"
#include "fft.cuh"
#include "fft_device.cuh"
#include "twiddles.cuh"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gpufft
{
    // Compile-time description of an FFT, for use inside user kernels. A
    // description is the sum of operators, each fixing one property:
    //
    //   using FwdFFT = decltype(Size<4096>() + Direction<FORWARD>() +
    //                           Precision<Float64>() +
    //                           ReductionPoly<X_N_minus>() + Block());
    //
    // Every operator may appear at most once.

    enum class FFTOperator
    {
        Size,
        Direction,
        Precision,
        ReductionPoly,
        Block,
        BlockDim,
        BatchesPerBlock,
        Transform
    };

    template <FFTOperator Op> struct FFTOperatorBase
    {
        static constexpr FFTOperator op = Op;
    };

    // Size N: the number of complex values of one polynomial (for the
    // special FFT the number of slots, for the FFNT the number of real
    // values).
    template <int N> struct Size : FFTOperatorBase<FFTOperator::Size>
    {
        static_assert(is_power_of_two(N) && N >= 2 && N <= (1 << kMaxLogN),
                      "Size must be a power of two in [2, 2^28]");
        static constexpr int value = N;
        static constexpr int log_value = log2_of_power_of_two(N);
    };

    template <type D> struct Direction : FFTOperatorBase<FFTOperator::Direction>
    {
        static_assert(D == FORWARD || D == INVERSE,
                      "Direction must be FORWARD or INVERSE");
        static constexpr type value = D;
    };

    // Type of the real and imaginary parts: Float32 or Float64.
    template <typename T>
    struct Precision : FFTOperatorBase<FFTOperator::Precision>
    {
        static_assert(std::is_same<T, Float32>::value ||
                          std::is_same<T, Float64>::value,
                      "Precision must be Float32 or Float64");
        using real_type = T;
    };

    // X_N_minus: X^N - 1 (cyclic), X_N_plus: X^N + 1 (negacyclic).
    template <ReductionPolynomial R>
    struct ReductionPoly : FFTOperatorBase<FFTOperator::ReductionPoly>
    {
        static_assert(R == X_N_minus || R == X_N_plus,
                      "ReductionPoly must be X_N_minus or X_N_plus");
        static constexpr ReductionPolynomial value = R;
    };

    // The FFT is computed by a thread block.
    struct Block : FFTOperatorBase<FFTOperator::Block>
    {
    };

    template <int X, int Y = 1, int Z = 1>
    struct BlockDim : FFTOperatorBase<FFTOperator::BlockDim>
    {
        static_assert(X >= 1 && Y >= 1 && Z >= 1 && X * Y * Z <= 1024,
                      "BlockDim must have between 1 and 1024 threads");
        static constexpr int x = X;
        static constexpr int y = Y;
        static constexpr int z = Z;
    };

    template <int B>
    struct BatchesPerBlock : FFTOperatorBase<FFTOperator::BatchesPerBlock>
    {
        static_assert(B >= 1, "BatchesPerBlock must be at least 1");
        static constexpr int value = B;
    };

    // The transform a description computes: the FFT of GPU_FFT (the
    // default), the special FFT of GPU_Special_FFT or the FFNT of GPU_FFNT.
    enum class TransformType
    {
        FFT,
        Special,
        FFNT
    };

    template <TransformType X>
    struct Transform : FFTOperatorBase<FFTOperator::Transform>
    {
        static_assert(X == TransformType::FFT || X == TransformType::Special ||
                          X == TransformType::FFNT,
                      "Transform must be TransformType::FFT, Special or FFNT");
        static constexpr TransformType value = X;
    };

    template <typename... Operators> struct FFTDescription;

    template <typename T, typename = void>
    struct is_fft_operator : std::false_type
    {
    };

    template <typename T>
    struct is_fft_operator<T, std::void_t<decltype(T::op)>>
        : std::is_same<std::remove_cv_t<decltype(T::op)>, FFTOperator>
    {
    };

    template <typename T> struct is_fft_description : std::false_type
    {
    };

    template <typename... Operators>
    struct is_fft_description<FFTDescription<Operators...>> : std::true_type
    {
    };

    template <FFTOperator Op, typename Description> struct has_operator;

    template <FFTOperator Op, typename... Operators>
    struct has_operator<Op, FFTDescription<Operators...>>
        : std::bool_constant<((Operators::op == Op) || ...)>
    {
    };

    // The first operator of kind Op in Operators, or Default if there is
    // none.
    template <FFTOperator Op, typename Default, typename... Operators>
    struct find_operator
    {
        using type = Default;
    };

    template <FFTOperator Op, typename Default, typename First,
              typename... Rest>
    struct find_operator<Op, Default, First, Rest...>
    {
        using type = std::conditional_t<
            First::op == Op, First,
            typename find_operator<Op, Default, Rest...>::type>;
    };

    template <FFTOperator Op, typename Description, typename Default>
    struct get_or_default;

    template <FFTOperator Op, typename... Operators, typename Default>
    struct get_or_default<Op, FFTDescription<Operators...>, Default>
    {
        using type = typename find_operator<Op, Default, Operators...>::type;
    };

    template <FFTOperator Op, typename Description, typename Default>
    using get_or_default_t =
        typename get_or_default<Op, Description, Default>::type;

    // The kernel's dynamic shared memory as an array of T. Kernels that use
    // the device API launch with the description's shared_memory_size.
    template <typename T> __device__ __forceinline__ T* dynamic_shared_memory()
    {
        extern __shared__ __align__(16) unsigned char gpufft_shared_memory[];
        return reinterpret_cast<T*>(gpufft_shared_memory);
    }

    // The shared memory entries of one thread: entry k is
    // shared_memory[position[k]] and belongs at data[index[k]], where data is
    // the array load reads and store writes. load fills exactly these entries
    // and they keep their positions until store. In a last single kernel
    // block that is not full, the threads of the missing polynomials have
    // valid == false.
    struct ThreadEntries
    {
        static constexpr int count = 2;
        bool valid;
        int position[count];
        size_t index[count];
    };

    template <typename T, typename V>
    constexpr bool is_complex_or_real =
        std::is_same<std::remove_const_t<V>, COMPLEX<T>>::value ||
        std::is_same<std::remove_const_t<V>, T>::value;

    // The direction of the stage plan a transform runs: the special FFT runs
    // the inverse plan forward and the forward plan inverse (see stage_plan).
    constexpr type plan_direction(TransformType transform, type direction)
    {
        if (transform != TransformType::Special)
        {
            return direction;
        }
        return (direction == FORWARD) ? INVERSE : FORWARD;
    }

    // The butterflies of a stage kernel with the launch parameters
    // Kernel::config, the same as the host API kernels. The FFT of the FFNT
    // is cyclic. Ends with a block barrier.
    template <typename Kernel, type D, int LogN, TransformType X, typename T>
    __device__ __forceinline__ void
    stage_execute(const StageThread& thread, COMPLEX<T>* shared_memory,
                  const COMPLEX<T>* roots, bool x_n_minus)
    {
        constexpr KernelConfig config = Kernel::config;
        if constexpr (X == TransformType::Special && D == FORWARD)
        {
            IncreasingStageExecute<CooleyTukey>(
                thread, shared_memory, roots,
                SpecialRoots<true>{(location_t) 1 << (LogN - config.logm - 1)},
                (config.shared_index + 1) - config.outer_iteration_count,
                config.outer_iteration_count);
        }
        else if constexpr (X == TransformType::Special)
        {
            DecreasingStageExecute<GentlemanSande>(
                thread, shared_memory, roots,
                SpecialRoots<false>{(location_t) 1 << (LogN - config.logm - 1)},
                config.shared_index, config.outer_iteration_count,
                config.not_last_kernel, config.shared_index - 5);
        }
        else if constexpr (D == FORWARD)
        {
            DecreasingStageExecute<CooleyTukey>(
                thread, shared_memory, roots,
                PolyRoots<false>{x_n_minus, LogN - config.logm - 1,
                                 (location_t) 1 << config.logm},
                config.shared_index, config.outer_iteration_count,
                config.not_last_kernel, config.shared_index - 5);
        }
        else
        {
            IncreasingStageExecute<GentlemanSande>(
                thread, shared_memory, roots,
                PolyRoots<true>{x_n_minus, LogN - config.logm - 1,
                                (location_t) 1 << config.logm},
                (config.shared_index + 1) - config.outer_iteration_count,
                config.outer_iteration_count);
        }
    }

    // The butterflies of a low ring kernel. Ends with a block barrier.
    template <type D, int LogN, TransformType X, typename T>
    __device__ __forceinline__ void low_ring_execute(COMPLEX<T>* shared_memory,
                                                     const COMPLEX<T>* roots,
                                                     bool x_n_minus)
    {
        if constexpr (X == TransformType::Special && D == FORWARD)
        {
            SpecialForwardLowRingExecute<LogN>(shared_memory, roots);
        }
        else if constexpr (X == TransformType::Special)
        {
            SpecialInverseLowRingExecute<LogN>(shared_memory, roots);
        }
        else if constexpr (D == FORWARD)
        {
            ForwardLowRingExecute<LogN>(shared_memory, roots, x_n_minus);
        }
        else
        {
            InverseLowRingExecute<LogN>(shared_memory, roots, x_n_minus);
        }
    }

    // Launch parameters and device functions of kernel I of an FFT of size
    // 2^LogN (for the FFNT, the FFT of the 2^(LogN + 1) real values). Sizes
    // up to 2^kLowRingMaxLogN use the low ring kernel, which transforms B
    // polynomials per block (use the single kernel API for them); larger
    // ones use one or more stage kernels. A stage kernel transforms part of
    // polynomial blockIdx.z of data: launch kernel I with grid(batch),
    // block_dim and shared_memory_size, in order I = 0, 1, ..., num_stages -
    // 1. Kernel 0 reads the input; every kernel writes the output array,
    // which the later kernels read and write in place. Inside kernel I:
    //
    //   FFT::stage<I>().load(shared_memory, data);
    //   FFT::stage<I>().execute(shared_memory, roots);
    //   FFT::stage<I>().store(shared_memory, out);
    //
    // The last inverse kernel stores with store(shared_memory, out,
    // n_inverse); the first forward FFNT kernel loads with
    // load(shared_memory, data, twist) and the last inverse one stores with
    // store(shared_memory, out, untwist, n_inverse).
    //
    // for_each gives each entry with its position in data, for element-wise
    // work inside a stage kernel. The last forward stage and the first
    // inverse stage of a size have the same launch parameters and keep the
    // same entries in shared memory, so one kernel can run the last forward
    // execute, element-wise work and the first inverse execute without
    // storing and loading in between.
    //
    // Threads is the thread count of BlockDim<Threads>, or 0 for the default
    // (see stage_plan).
    template <typename T, type D, int LogN, int I, int B, bool HasReductionPoly,
              ReductionPolynomial R, int Threads = 0,
              TransformType X = TransformType::FFT>
    struct FFTStage
    {
        static constexpr StagePlan plan =
            stage_plan<T>(plan_direction(X, D), LogN, Threads);

        // The stage runs a forward plan (see plan_direction).
        static constexpr bool forward_plan = plan_direction(X, D) == FORWARD;

        static_assert(I >= 0 && I < plan.count, "Stage index out of range");

        static constexpr KernelConfig config = plan.stage[I];

        static constexpr bool low_ring = (LogN <= kLowRingMaxLogN);

        static constexpr dim3 block_dim =
            low_ring ? dim3(1 << (LogN - 1), B)
                     : dim3(config.blockdim_x, config.blockdim_y);

        static constexpr size_t shared_memory_size =
            low_ring ? (size_t(B) << LogN) * sizeof(COMPLEX<T>)
                     : config.shared_memory;

        static constexpr dim3 grid(int batch)
        {
            return low_ring ? dim3((batch + B - 1) / B)
                            : dim3(config.griddim_x, config.griddim_y, batch);
        }

        static constexpr int index = I;

        static constexpr bool last = (I + 1 == plan.count);

        // The kernel swaps the roles of blockIdx.x and blockIdx.y, because
        // its groups do not fit grid y; grid() already accounts for it.
        static constexpr bool swap_grid_axes = plan.swap_grid_axes[I];

        // The kernel's dynamic shared memory (shared_memory_size bytes).
        __device__ __forceinline__ static COMPLEX<T>* shared_memory()
        {
            return dynamic_shared_memory<COMPLEX<T>>();
        }

        // data holds complex values, or real values with zero imaginary
        // parts.
        template <typename TIn>
        __device__ __forceinline__ void load(COMPLEX<T>* shared_memory,
                                             const TIn* data) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(is_complex_or_real<T, TIn>,
                          "The input must be complex or real values of the "
                          "description's Precision");
            static_assert(
                X != TransformType::FFNT ||
                    std::is_same<std::remove_const_t<TIn>, COMPLEX<T>>::value,
                "The FFNT reads its real values with "
                "load(shared_memory, data, twist)");
            const StageThread state = thread();
            StageLoad(state, shared_memory, data, poly_address(state, LogN));
        }

        // The first forward FFNT stage: folds the 2N real values of the
        // polynomial (N = 2^LogN) to f_j + i f_(j + N) and multiplies them by
        // the twist table.
        __device__ __forceinline__ void load(COMPLEX<T>* shared_memory,
                                             const T* data,
                                             const COMPLEX<T>* twist) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^10");
            static_assert(X == TransformType::FFNT && D == FORWARD && I == 0,
                          "Only the first forward FFNT stage folds and "
                          "twists");
            const StageThread state = thread();
            StageLoadFoldTwist(state, shared_memory, data,
                               poly_address(state, LogN + 1), twist, LogN);
        }

        __device__ __forceinline__ void execute(COMPLEX<T>* shared_memory,
                                                const COMPLEX<T>* roots) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(HasReductionPoly,
                          "execute needs the ReductionPoly operator");
            stage_execute<FFTStage, D, LogN, X>(thread(), shared_memory, roots,
                                                R == X_N_minus);
        }

        // This thread's two entries (see ThreadEntries).
        __device__ __forceinline__ ThreadEntries entries() const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            const StageThread state = thread();
            const int position = int(state.shared_address);
            const size_t index = poly_address(state, LogN);
            return ThreadEntries{
                true,
                {position, position + int(blockDim.x * blockDim.y)},
                {index, index + state.offset}};
        }

        // Calls f(value, index) for the entries of this thread (see
        // entries); index is the entry's position in data. Between load and
        // store an entry keeps its position, so after the last forward
        // execute value is element index of the host API output. Ends with a
        // block barrier.
        template <typename F>
        __device__ __forceinline__ void for_each(COMPLEX<T>* shared_memory,
                                                 F f) const
        {
            const ThreadEntries thread_entries = entries();
            for (int k = 0; k < ThreadEntries::count; k++)
            {
                f(shared_memory[thread_entries.position[k]],
                  thread_entries.index[k]);
            }
            __syncthreads();
        }

        __device__ __forceinline__ void store(COMPLEX<T>* shared_memory,
                                              COMPLEX<T>* data) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(D == FORWARD || !last,
                          "The last inverse store needs n_inverse");
            const StageThread state = thread();
            StageStore(state, shared_memory, data, poly_address(state, LogN));
        }

        // Writes the values times n_inverse, as complex values or their real
        // parts.
        template <typename TOut>
        __device__ __forceinline__ void
        store(COMPLEX<T>* shared_memory, TOut* data, COMPLEX<T> n_inverse) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(D == INVERSE && last,
                          "Only the last inverse store takes n_inverse");
            static_assert(X != TransformType::FFNT,
                          "The last inverse FFNT store needs the untwist "
                          "table");
            static_assert(is_complex_or_real<T, TOut>,
                          "The output must be complex or real values of the "
                          "description's Precision");
            const StageThread state = thread();
            StageStoreScaled(state, shared_memory, data,
                             poly_address(state, LogN), n_inverse);
        }

        // The last inverse FFNT stage: multiplies by n_inverse and the
        // untwist table and unfolds to the 2N real values of the polynomial.
        __device__ __forceinline__ void store(COMPLEX<T>* shared_memory,
                                              T* data,
                                              const COMPLEX<T>* untwist,
                                              COMPLEX<T> n_inverse) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^10");
            static_assert(X == TransformType::FFNT && D == INVERSE && last,
                          "Only the last inverse FFNT stage untwists and "
                          "unfolds");
            const StageThread state = thread();
            StageStoreUntwistUnfold(state, shared_memory, data,
                                    poly_address(state, LogN + 1), untwist,
                                    n_inverse, LogN);
        }

      private:
        __device__ __forceinline__ static StageThread thread()
        {
            return StageThreadState<swap_grid_axes>(
                forward_plan ? config.logm : config.k,
                config.outer_iteration_count, LogN);
        }
    };

    // Launch parameters and device functions of an FFT that runs in a single
    // kernel (sizes up to 2^10, and 2^11 with BlockDim<1024>; for the FFNT
    // twice these numbers of real values). Block b transforms the B
    // polynomials that start at data + b * B * N; launch it with
    // grid(batch), block_dim and shared_memory_size. Inside the kernel:
    //
    //   FFT().load(shared_memory, data);
    //   FFT().execute(shared_memory, roots);
    //   ... optional work on shared_memory ...
    //   FFT().store(shared_memory, data);              // forward
    //   FFT().store(shared_memory, data, n_inverse);   // inverse
    //
    // The FFNT loads with load(shared_memory, data, twist) (forward) and
    // stores with store(shared_memory, data, untwist, n_inverse) (inverse).
    //
    // num_polys (at most B) is the number of polynomials of the block, for a
    // last block that is not full. execute ends with a block barrier; code
    // that writes shared memory between execute and store must end with
    // __syncthreads(). The inverse execute leaves N * IFFT(x) in shared
    // memory; store multiplies by n_inverse, usually n_inv().
    template <typename T, type D, int LogN, int B, bool HasReductionPoly,
              ReductionPolynomial R, int Threads = 0,
              TransformType X = TransformType::FFT>
    struct FFTSingleKernel
    {
        static constexpr KernelConfig config =
            stage_plan<T>(plan_direction(X, D), LogN, Threads).stage[0];

        // The kernel runs a forward plan (see plan_direction).
        static constexpr bool forward_plan = plan_direction(X, D) == FORWARD;

        static constexpr bool low_ring = (LogN <= kLowRingMaxLogN);

        static constexpr dim3 block_dim =
            low_ring ? dim3(1 << (LogN - 1), B)
                     : dim3(config.blockdim_x, config.blockdim_y);

        static constexpr size_t shared_memory_size =
            low_ring ? (size_t(B) << LogN) * sizeof(COMPLEX<T>)
                     : config.shared_memory;

        static constexpr dim3 grid(int batch)
        {
            return dim3((batch + B - 1) / B);
        }

        // The kernel's dynamic shared memory (shared_memory_size bytes).
        __device__ __forceinline__ static COMPLEX<T>* shared_memory()
        {
            return dynamic_shared_memory<COMPLEX<T>>();
        }

        // The block's first polynomial in an array of polynomials:
        // data + blockIdx.x * B * N, where N is the number of values of a
        // polynomial in data (for the FFNT 2^(LogN + 1) real or 2^LogN
        // complex values).
        template <typename U>
        __device__ __forceinline__ static U* block_data(U* data)
        {
            constexpr int log_values =
                (X == TransformType::FFNT &&
                 std::is_same<std::remove_const_t<U>, T>::value)
                    ? LogN + 1
                    : LogN;
            return data + ((size_t(blockIdx.x) * B) << log_values);
        }

        // Polynomials of the block out of batch: B, or fewer in a last block
        // that is not full.
        __device__ __forceinline__ static int num_polys(int batch)
        {
            return min(B, batch - int(blockIdx.x) * B);
        }

        // data holds complex values, or real values with zero imaginary
        // parts.
        template <typename TIn>
        __device__ __forceinline__ void load(COMPLEX<T>* shared_memory,
                                             const TIn* data,
                                             int num_polys = B) const
        {
            static_assert(is_complex_or_real<T, TIn>,
                          "The input must be complex or real values of the "
                          "description's Precision");
            static_assert(
                X != TransformType::FFNT ||
                    std::is_same<std::remove_const_t<TIn>, COMPLEX<T>>::value,
                "The FFNT reads its real values with "
                "load(shared_memory, data, twist)");
            if constexpr (low_ring)
            {
                LowRingLoad(low_ring_thread(num_polys), shared_memory, data);
            }
            else
            {
                const StageThread state = stage_thread();
                StageLoad(state, shared_memory, data, state.local_address);
            }
        }

        // The forward FFNT: folds the 2N real values of each polynomial
        // (N = 2^LogN) to f_j + i f_(j + N) and multiplies them by the twist
        // table.
        __device__ __forceinline__ void load(COMPLEX<T>* shared_memory,
                                             const T* data,
                                             const COMPLEX<T>* twist,
                                             int num_polys = B) const
        {
            static_assert(X == TransformType::FFNT && D == FORWARD,
                          "Only the forward FFNT folds and twists");
            if constexpr (low_ring)
            {
                LowRingLoadFoldTwist(low_ring_thread(num_polys), shared_memory,
                                     data, real_address(num_polys), twist,
                                     LogN);
            }
            else
            {
                const StageThread state = stage_thread();
                StageLoadFoldTwist(state, shared_memory, data,
                                   state.local_address, twist, LogN);
            }
        }

        __device__ __forceinline__ void execute(COMPLEX<T>* shared_memory,
                                                const COMPLEX<T>* roots) const
        {
            static_assert(HasReductionPoly,
                          "execute needs the ReductionPoly operator");
            if constexpr (low_ring)
            {
                low_ring_execute<D, LogN, X>(shared_memory, roots,
                                             R == X_N_minus);
            }
            else
            {
                stage_execute<FFTSingleKernel, D, LogN, X>(
                    stage_thread(), shared_memory, roots, R == X_N_minus);
            }
        }

        // This thread's two entries (see ThreadEntries). Shared memory holds
        // the block's polynomials in the order of data, so position and
        // index are equal for the low ring kernel.
        __device__ __forceinline__ ThreadEntries
        entries(int num_polys = B) const
        {
            if constexpr (low_ring)
            {
                const LowRingThread state = low_ring_thread(num_polys);
                return ThreadEntries{
                    state.active,
                    {state.shared_address0, state.shared_address1},
                    {size_t(state.shared_address0),
                     size_t(state.shared_address1)}};
            }
            else
            {
                const StageThread state = stage_thread();
                const int position = int(state.shared_address);
                return ThreadEntries{
                    true,
                    {position, position + int(blockDim.x * blockDim.y)},
                    {size_t(state.local_address),
                     size_t(state.local_address + state.offset)}};
            }
        }

        // Calls f(value, index) for the entries of the block's num_polys
        // polynomials, each by the thread that owns it (see entries); index
        // is the entry's position in data, the array load reads and store
        // writes (for the FFNT the complex values). Ends with a block
        // barrier.
        template <typename F>
        __device__ __forceinline__ void for_each(COMPLEX<T>* shared_memory, F f,
                                                 int num_polys = B) const
        {
            const ThreadEntries thread_entries = entries(num_polys);
            if (thread_entries.valid)
            {
                for (int k = 0; k < ThreadEntries::count; k++)
                {
                    f(shared_memory[thread_entries.position[k]],
                      thread_entries.index[k]);
                }
            }
            __syncthreads();
        }

        __device__ __forceinline__ void store(COMPLEX<T>* shared_memory,
                                              COMPLEX<T>* data,
                                              int num_polys = B) const
        {
            static_assert(D == FORWARD, "The inverse store needs n_inverse");
            if constexpr (low_ring)
            {
                LowRingStore(low_ring_thread(num_polys), shared_memory, data);
            }
            else
            {
                const StageThread state = stage_thread();
                StageStore(state, shared_memory, data, state.local_address);
            }
        }

        // Writes the values times n_inverse, as complex values or their real
        // parts.
        template <typename TOut>
        __device__ __forceinline__ void store(COMPLEX<T>* shared_memory,
                                              TOut* data, COMPLEX<T> n_inverse,
                                              int num_polys = B) const
        {
            static_assert(D == INVERSE, "Only the inverse store takes "
                                        "n_inverse");
            static_assert(X != TransformType::FFNT,
                          "The inverse FFNT store needs the untwist table");
            static_assert(is_complex_or_real<T, TOut>,
                          "The output must be complex or real values of the "
                          "description's Precision");
            if constexpr (low_ring)
            {
                LowRingStoreScaled(low_ring_thread(num_polys), shared_memory,
                                   data, n_inverse);
            }
            else
            {
                const StageThread state = stage_thread();
                StageStoreScaled(state, shared_memory, data,
                                 state.local_address, n_inverse);
            }
        }

        // The inverse FFNT: multiplies by n_inverse and the untwist table and
        // unfolds to the 2N real values of each polynomial.
        __device__ __forceinline__ void
        store(COMPLEX<T>* shared_memory, T* data, const COMPLEX<T>* untwist,
              COMPLEX<T> n_inverse, int num_polys = B) const
        {
            static_assert(X == TransformType::FFNT && D == INVERSE,
                          "Only the inverse FFNT untwists and unfolds");
            if constexpr (low_ring)
            {
                LowRingStoreUntwistUnfold(
                    low_ring_thread(num_polys), shared_memory, data,
                    real_address(num_polys), untwist, n_inverse, LogN);
            }
            else
            {
                const StageThread state = stage_thread();
                StageStoreUntwistUnfold(state, shared_memory, data,
                                        state.local_address, untwist, n_inverse,
                                        LogN);
            }
        }

      private:
        // Thread (x, y) owns values x and x + N/2 of polynomial y of the
        // block. Rows y >= num_polys read polynomial 0 and store nothing.
        __device__ __forceinline__ static LowRingThread
        low_ring_thread(int num_polys)
        {
            const int idx_x = threadIdx.x;
            const int idx_y = threadIdx.y;
            const bool active = (idx_y < num_polys);
            const int half_n = 1 << (LogN - 1);
            const int shared_address0 = (idx_y << LogN) + idx_x;
            return LowRingThread{
                active, location_t(idx_x + ((active ? idx_y : 0) << LogN)),
                location_t(half_n), shared_address0, shared_address0 + half_n};
        }

        // The address of the thread's first FFNT real value f_x in the
        // block's real data (2^(LogN + 1) values per polynomial).
        __device__ __forceinline__ static location_t real_address(int num_polys)
        {
            const int idx_y = threadIdx.y;
            return location_t(threadIdx.x +
                              ((idx_y < num_polys ? idx_y : 0) << (LogN + 1)));
        }

        // The single stage kernel transforms one polynomial per block, as a
        // stage kernel with block (0, 0).
        __device__ __forceinline__ static StageThread stage_thread()
        {
            return StageThreadState(forward_plan ? config.logm : config.k,
                                    config.outer_iteration_count, LogN,
                                    threadIdx.x, threadIdx.y, 0, 0);
        }
    };

    // Root table of a transform of size N, built on the host or by a kernel
    // (see twiddles.cuh); the inverse tables hold the conjugates.
    // Descriptions of an FFT without ReductionPoly have no root table.
    template <typename T, int N, type D, bool HasReductionPoly,
              ReductionPolynomial R, TransformType X = TransformType::FFT>
    struct FFTRootTable
    {
    };

    // The FFT: the powers of exp(2 pi i / N) (X^N - 1) or of exp(pi i / N)
    // (X^N + 1) in bit-reversed order.
    template <typename T, int N, type D, ReductionPolynomial R>
    struct FFTRootTable<T, N, D, true, R, TransformType::FFT>
    {
        // Entries the kernels read: N / 2 for X^N - 1, N for X^N + 1.
        static constexpr int twiddle_table_size = (R == X_N_plus) ? N : N / 2;

        // The table that execute and GPU_FFT take, computed in double
        // precision and rounded to Precision.
        static std::vector<COMPLEX<T>> make_twiddles_host()
        {
            return make_fft_twiddles_host<T>(log2_of_power_of_two(N), R, D);
        }

        // The same table, written by a kernel to table, which holds
        // twiddle_table_size entries in device memory. The values agree
        // with make_twiddles_host up to rounding. The kernel runs on stream
        // and table is ready when it finishes.
        static void make_twiddles(COMPLEX<T>* table, cudaStream_t stream = 0)
        {
            make_fft_twiddles(table, log2_of_power_of_two(N), R, D, stream);
        }
    };

    // The special FFT of N slots (see make_special_fft_twiddles_host).
    template <typename T, int N, type D, ReductionPolynomial R>
    struct FFTRootTable<T, N, D, true, R, TransformType::Special>
    {
        static constexpr int twiddle_table_size = N;

        static std::vector<COMPLEX<T>> make_twiddles_host()
        {
            return make_special_fft_twiddles_host<T>(log2_of_power_of_two(N),
                                                     D);
        }

        static void make_twiddles(COMPLEX<T>* table, cudaStream_t stream = 0)
        {
            make_special_fft_twiddles(table, log2_of_power_of_two(N), D,
                                      stream);
        }
    };

    // The FFNT of N real values: the root table of its FFT of size N / 2
    // (X^N - 1) and the twist (forward) or untwist (inverse) table.
    template <typename T, int N, type D, ReductionPolynomial R>
    struct FFTRootTable<T, N, D, true, R, TransformType::FFNT>
    {
        static constexpr int twiddle_table_size = N / 4;

        static constexpr int twist_table_size = N / 2;

        static std::vector<COMPLEX<T>> make_twiddles_host()
        {
            return make_ffnt_twiddles_host<T>(log2_of_power_of_two(N), D);
        }

        static void make_twiddles(COMPLEX<T>* table, cudaStream_t stream = 0)
        {
            make_ffnt_twiddles(table, log2_of_power_of_two(N), D, stream);
        }

        static std::vector<COMPLEX<T>> make_twist_host()
        {
            return make_ffnt_twist_host<T>(log2_of_power_of_two(N), D);
        }

        static void make_twist(COMPLEX<T>* table, cudaStream_t stream = 0)
        {
            make_ffnt_twist(table, log2_of_power_of_two(N), D, stream);
        }
    };

    // The kernels of FFT::launch: the whole transform in a single kernel
    // (Kernel is an FFTSingleKernel) or one stage of it (Stage is an
    // FFTStage).
    template <typename Kernel, type D, typename T, typename TIn, typename TOut>
    __global__ void fft_launch_kernel(const TIn* in, TOut* out,
                                      const COMPLEX<T>* roots,
                                      COMPLEX<T> n_inverse, int batch)
    {
        COMPLEX<T>* shared_memory = Kernel::shared_memory();
        const int num_polys = Kernel::num_polys(batch);
        Kernel().load(shared_memory, Kernel::block_data(in), num_polys);
        Kernel().execute(shared_memory, roots);
        if constexpr (D == FORWARD)
        {
            Kernel().store(shared_memory, Kernel::block_data(out), num_polys);
        }
        else
        {
            Kernel().store(shared_memory, Kernel::block_data(out), n_inverse,
                           num_polys);
        }
    }

    template <typename Stage, type D, typename T, typename TIn, typename TOut>
    __global__ void fft_launch_stage_kernel(const TIn* in, TOut* out,
                                            const COMPLEX<T>* roots,
                                            COMPLEX<T> n_inverse)
    {
        COMPLEX<T>* shared_memory = Stage::shared_memory();
        Stage().load(shared_memory, in);
        Stage().execute(shared_memory, roots);
        if constexpr (D == INVERSE && Stage::last)
        {
            Stage().store(shared_memory, out, n_inverse);
        }
        else
        {
            Stage().store(shared_memory, out);
        }
    }

    // The same for the FFNT, with the twist (forward) or untwist (inverse)
    // table.
    template <typename Kernel, type D, typename T, typename TIn, typename TOut>
    __global__ void
    ffnt_launch_kernel(const TIn* in, TOut* out, const COMPLEX<T>* roots,
                       const COMPLEX<T>* twist, COMPLEX<T> n_inverse, int batch)
    {
        COMPLEX<T>* shared_memory = Kernel::shared_memory();
        const int num_polys = Kernel::num_polys(batch);
        if constexpr (D == FORWARD)
        {
            Kernel().load(shared_memory, Kernel::block_data(in), twist,
                          num_polys);
            Kernel().execute(shared_memory, roots);
            Kernel().store(shared_memory, Kernel::block_data(out), num_polys);
        }
        else
        {
            Kernel().load(shared_memory, Kernel::block_data(in), num_polys);
            Kernel().execute(shared_memory, roots);
            Kernel().store(shared_memory, Kernel::block_data(out), twist,
                           n_inverse, num_polys);
        }
    }

    template <typename Stage, type D, typename T, typename TIn, typename TOut>
    __global__ void
    ffnt_launch_stage_kernel(const TIn* in, TOut* out, const COMPLEX<T>* roots,
                             const COMPLEX<T>* twist, COMPLEX<T> n_inverse)
    {
        COMPLEX<T>* shared_memory = Stage::shared_memory();
        if constexpr (D == FORWARD && Stage::index == 0)
        {
            Stage().load(shared_memory, in, twist);
        }
        else
        {
            Stage().load(shared_memory, in);
        }
        Stage().execute(shared_memory, roots);
        if constexpr (D == INVERSE && Stage::last)
        {
            Stage().store(shared_memory, out, twist, n_inverse);
        }
        else
        {
            Stage().store(shared_memory, out);
        }
    }

    // Values of a description that has Size, Direction and Precision.
    template <typename... Operators> struct FFTDescriptionValues
    {
      private:
        template <FFTOperator Op>
        using get = typename find_operator<Op, void, Operators...>::type;

        using batches_op = get<FFTOperator::BatchesPerBlock>;
        using block_dim_op = get<FFTOperator::BlockDim>;
        using reduction_op = get<FFTOperator::ReductionPoly>;
        using transform_op = get<FFTOperator::Transform>;

        static constexpr bool has_batches = !std::is_void<batches_op>::value;
        static constexpr bool has_block_dim =
            !std::is_void<block_dim_op>::value;
        static constexpr bool has_reduction_poly =
            !std::is_void<reduction_op>::value;

        // True if Op is absent or is BlockDim<x, y> with x == half_size.
        template <typename Op>
        static constexpr bool block_dim_is_rows(int half_size)
        {
            if constexpr (std::is_void<Op>::value)
            {
                return true;
            }
            else
            {
                return Op::x == half_size && Op::z == 1;
            }
        }

        template <typename Op> static constexpr int block_dim_y()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return 0;
            }
            else
            {
                return Op::y;
            }
        }

        // The thread count of BlockDim<T>, or 0 if Op is absent or has more
        // than one dimension.
        template <typename Op> static constexpr int block_dim_threads()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return 0;
            }
            else
            {
                return (Op::y == 1 && Op::z == 1) ? Op::x : 0;
            }
        }

        template <typename Op> static constexpr int batches_value()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return 0;
            }
            else
            {
                return Op::value;
            }
        }

        template <typename Op>
        static constexpr ReductionPolynomial reduction_value()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return X_N_minus;
            }
            else
            {
                return Op::value;
            }
        }

        template <typename Op> static constexpr TransformType transform_value()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return TransformType::FFT;
            }
            else
            {
                return Op::value;
            }
        }

      public:
        using real_type = typename get<FFTOperator::Precision>::real_type;
        using value_type = COMPLEX<real_type>;

        static constexpr int size = get<FFTOperator::Size>::value;
        static constexpr int log_size = get<FFTOperator::Size>::log_value;
        static constexpr type direction = get<FFTOperator::Direction>::value;
        static constexpr TransformType transform =
            transform_value<transform_op>();

      private:
        static constexpr bool ffnt = (transform == TransformType::FFNT);

        static_assert(!ffnt || size >= 4, "The FFNT needs Size at least 4");
        static_assert(transform == TransformType::FFT || !has_reduction_poly,
                      "ReductionPoly is only for TransformType::FFT");

        // The kernels run an FFT of 2^fft_log_size values: Size, or Size / 2
        // for the FFNT.
        static constexpr int fft_log_size = ffnt ? log_size - 1 : log_size;
        static constexpr int fft_size = 1 << fft_log_size;

        // The special FFT and the FFNT always have a root table; the FFT of
        // the FFNT is cyclic.
        static constexpr bool has_roots =
            has_reduction_poly || transform != TransformType::FFT;
        static constexpr ReductionPolynomial roots_poly =
            reduction_value<reduction_op>();

      public:
        // 1 / N for the FFT of N values (N = Size, or Size / 2 for the
        // FFNT), the n_inverse of the inverse store.
        __host__ __device__ static value_type n_inv()
        {
            return value_type(real_type(1) / real_type(fft_size), real_type(0));
        }

      private:
        static constexpr int
            threads_op = fft_log_size > kLowRingMaxLogN
                             ? block_dim_threads<block_dim_op>()
                             : 0;

        static_assert(fft_log_size <= kLowRingMaxLogN || !has_batches,
                      "BatchesPerBlock can only be set for an FFT size up to "
                      "2^9 (Size up to 2^10 for the FFNT)");
        static_assert(fft_log_size <= kLowRingMaxLogN || !has_block_dim ||
                          (threads_op >= 128 && is_power_of_two(threads_op)),
                      "For an FFT size above 2^9, BlockDim must be BlockDim<T> "
                      "with T = 128, 256, 512 or 1024 threads");
        static_assert(threads_op <= fft_size / 2,
                      "BlockDim<T> needs T at most half the FFT size (Size / "
                      "2, or Size / 4 for the FFNT)");

      public:
        // Threads per kernel set with BlockDim<T> for an FFT size above 2^9,
        // or 0 for the default (see stage_plan).
        static constexpr int block_threads =
            (threads_op >= 128 && is_power_of_two(threads_op) &&
             threads_op <= fft_size / 2)
                ? threads_op
                : 0;

      private:
        static constexpr StagePlan plan = stage_plan<real_type>(
            plan_direction(transform, direction), fft_log_size, block_threads);

      public:
        static constexpr int num_stages =
            (fft_log_size <= kLowRingMaxLogN) ? 1 : plan.count;

        // Calls f(std::integral_constant<int, I>()) for I = 0, 1, ...,
        // num_stages - 1 in order, e.g. to launch the stage kernels:
        //   FFT::for_each_stage([&](auto I) { my_stage<I><<<...>>>(...); });
        template <typename F> static void for_each_stage(F f)
        {
            call_for_stages(f, std::make_integer_sequence<int, num_stages>());
        }

        // Polynomials per block. Only the low ring kernel takes more than
        // one; by default it has 256 threads.
        static constexpr int batches_per_block =
            has_batches     ? batches_value<batches_op>()
            : has_block_dim ? block_dim_y<block_dim_op>()
            : fft_log_size <= kLowRingMaxLogN
                ? 1 << (kLowRingMaxLogN - fft_log_size)
                : 1;

        static_assert(fft_log_size > kLowRingMaxLogN ||
                          block_dim_is_rows<block_dim_op>(fft_size / 2),
                      "BlockDim must be (N / 2, BatchesPerBlock) for the FFT "
                      "size N (Size, or Size / 2 for the FFNT)");
        static_assert(fft_log_size > kLowRingMaxLogN || !has_block_dim ||
                          !has_batches ||
                          block_dim_y<block_dim_op>() ==
                              batches_value<batches_op>(),
                      "BlockDim must be (N / 2, BatchesPerBlock) for the FFT "
                      "size N (Size, or Size / 2 for the FFNT)");
        static_assert(fft_log_size > kLowRingMaxLogN ||
                          (fft_size / 2) * batches_per_block <= 1024,
                      "N / 2 * BatchesPerBlock must be at most 1024 threads "
                      "for the FFT size N");

        template <int I>
        using stage =
            FFTStage<real_type, direction, fft_log_size, I, batches_per_block,
                     has_roots, roots_poly, block_threads, transform>;

        using single_kernel =
            FFTSingleKernel<real_type, direction, fft_log_size,
                            batches_per_block, has_roots, roots_poly,
                            block_threads, transform>;

        using root_table = FFTRootTable<real_type, size, direction, has_roots,
                                        roots_poly, transform>;

        // Transforms batch polynomials on stream with the description's
        // kernels, the same result as GPU_FFT or GPU_Special_FFT: in holds
        // the polynomials one after another, out (which may be in) receives
        // the result. For the FFT the forward input may be real (zero
        // imaginary parts) and the inverse output may be real (the real
        // parts). roots is the root table of the description (see
        // make_twiddles); the inverse multiplies by n_inv(). An inverse with
        // real output and more than one stage keeps the intermediate values
        // in in, like GPU_FFT.
        template <typename TIn, typename TOut>
        static void launch(TIn* in, TOut* out, const value_type* roots,
                           int batch, cudaStream_t stream = 0)
        {
            using In = std::remove_const_t<TIn>;
            static_assert(has_roots, "launch needs the ReductionPoly operator");
            static_assert(!ffnt, "The FFNT launch takes the twist table: "
                                 "launch(in, out, roots, twist, batch)");
            static_assert(std::is_same<In, value_type>::value ||
                              (direction == FORWARD &&
                               transform == TransformType::FFT &&
                               std::is_same<In, real_type>::value),
                          "The input must be complex values of the "
                          "description's Precision, or real values for the "
                          "forward FFT");
            static_assert(std::is_same<TOut, value_type>::value ||
                              (direction == INVERSE &&
                               transform == TransformType::FFT &&
                               std::is_same<TOut, real_type>::value),
                          "The output must be complex values of the "
                          "description's Precision, or real values for the "
                          "inverse FFT");
            static_assert(std::is_same<TOut, value_type>::value ||
                              num_stages == 1 || !std::is_const<TIn>::value,
                          "An inverse FFT with real output and several stages "
                          "writes intermediate values to in");

            check_batch(batch);
            const value_type n_inverse =
                (direction == INVERSE) ? n_inv() : value_type();
            if constexpr (num_stages == 1)
            {
                fft_launch_kernel<single_kernel, direction, real_type>
                    <<<single_kernel::grid(batch), single_kernel::block_dim,
                       single_kernel::shared_memory_size, stream>>>(
                        in, out, roots, n_inverse, batch);
                GPUFFT_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                // Stage 0 reads in; the stages write the work array, which
                // the later stages read and write in place, and the last one
                // writes out.
                value_type* work;
                if constexpr (std::is_same<TOut, value_type>::value)
                {
                    work = out;
                }
                else
                {
                    work = in;
                }
                for_each_stage(
                    [&](auto I)
                    {
                        if constexpr (I == 0)
                        {
                            launch_stage<I>(in, work, roots, n_inverse, batch,
                                            stream);
                        }
                        else if constexpr (I + 1 == num_stages)
                        {
                            launch_stage<I>(work, out, roots, n_inverse, batch,
                                            stream);
                        }
                        else
                        {
                            launch_stage<I>(work, work, roots, n_inverse, batch,
                                            stream);
                        }
                    });
            }
        }

        // The FFNT of batch polynomials of Size real values, the same result
        // as GPU_FFNT: the forward transform reads real values from in and
        // writes Size / 2 complex values per polynomial to out, the inverse
        // the other way round. roots and twist are the root table and the
        // twist table of the description (see make_twiddles and
        // make_twist); the inverse multiplies by n_inv(). An inverse with
        // more than one stage keeps the intermediate values in in, like
        // GPU_FFNT.
        template <typename TIn, typename TOut>
        static void launch(TIn* in, TOut* out, const value_type* roots,
                           const value_type* twist, int batch,
                           cudaStream_t stream = 0)
        {
            using In = std::remove_const_t<TIn>;
            static_assert(ffnt, "Only the FFNT launch takes a twist table");
            static_assert(direction == FORWARD
                              ? (std::is_same<In, real_type>::value &&
                                 std::is_same<TOut, value_type>::value)
                              : (std::is_same<In, value_type>::value &&
                                 std::is_same<TOut, real_type>::value),
                          "The forward FFNT reads real values and writes "
                          "complex values of the description's Precision, the "
                          "inverse FFNT the other way round");
            static_assert(direction == FORWARD || num_stages == 1 ||
                              !std::is_const<TIn>::value,
                          "An inverse FFNT with several stages writes "
                          "intermediate values to in");

            check_batch(batch);
            const value_type n_inverse =
                (direction == INVERSE) ? n_inv() : value_type();
            if constexpr (num_stages == 1)
            {
                ffnt_launch_kernel<single_kernel, direction, real_type>
                    <<<single_kernel::grid(batch), single_kernel::block_dim,
                       single_kernel::shared_memory_size, stream>>>(
                        in, out, roots, twist, n_inverse, batch);
                GPUFFT_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                // The complex values stay in out (forward) or in (inverse).
                value_type* work;
                if constexpr (direction == FORWARD)
                {
                    work = out;
                }
                else
                {
                    work = in;
                }
                for_each_stage(
                    [&](auto I)
                    {
                        if constexpr (I == 0)
                        {
                            launch_ffnt_stage<I>(in, work, roots, twist,
                                                 n_inverse, batch, stream);
                        }
                        else if constexpr (I + 1 == num_stages)
                        {
                            launch_ffnt_stage<I>(work, out, roots, twist,
                                                 n_inverse, batch, stream);
                        }
                        else
                        {
                            launch_ffnt_stage<I>(work, work, roots, twist,
                                                 n_inverse, batch, stream);
                        }
                    });
            }
        }

      private:
        // The stage kernels take the batch as the third grid dimension.
        static void check_batch(int batch)
        {
            if (batch <= 0 ||
                (static_cast<long long>(batch) << log_size) >= (1LL << 31) ||
                (num_stages > 1 && batch > 65535))
            {
                throw std::invalid_argument(
                    "batch must be positive with batch * Size below 2^31, "
                    "and at most 65535 for more than one stage.");
            }
        }

        template <int I, typename TIn, typename TOut>
        static void launch_stage(const TIn* in, TOut* out,
                                 const value_type* roots, value_type n_inverse,
                                 int batch, cudaStream_t stream)
        {
            using Stage = stage<I>;
            fft_launch_stage_kernel<Stage, direction, real_type>
                <<<Stage::grid(batch), Stage::block_dim,
                   Stage::shared_memory_size, stream>>>(in, out, roots,
                                                        n_inverse);
            GPUFFT_CUDA_CHECK(cudaGetLastError());
        }

        template <int I, typename TIn, typename TOut>
        static void
        launch_ffnt_stage(const TIn* in, TOut* out, const value_type* roots,
                          const value_type* twist, value_type n_inverse,
                          int batch, cudaStream_t stream)
        {
            using Stage = stage<I>;
            ffnt_launch_stage_kernel<Stage, direction, real_type>
                <<<Stage::grid(batch), Stage::block_dim,
                   Stage::shared_memory_size, stream>>>(in, out, roots, twist,
                                                        n_inverse);
            GPUFFT_CUDA_CHECK(cudaGetLastError());
        }

        template <typename F, int... I>
        static void call_for_stages(F f, std::integer_sequence<int, I...>)
        {
            (f(std::integral_constant<int, I>()), ...);
        }
    };

    struct FFTNoMembers
    {
    };

    template <typename... Operators>
    constexpr bool has_required_operators =
        ((Operators::op == FFTOperator::Size) || ...) &&
        ((Operators::op == FFTOperator::Direction) || ...) &&
        ((Operators::op == FFTOperator::Precision) || ...);

    template <bool Complete, typename... Operators>
    struct FFTDescriptionMembers : FFTNoMembers
    {
    };

    // A description with a single kernel also has that kernel's launch
    // parameters and device functions (see FFTSingleKernel); one with a root
    // table (ReductionPoly, or the special FFT or the FFNT) also has the root
    // table helpers (see FFTRootTable).
    template <typename... Operators>
    struct FFTDescriptionMembers<true, Operators...>
        : FFTDescriptionValues<Operators...>,
          FFTDescriptionValues<Operators...>::root_table,
          std::conditional_t<
              FFTDescriptionValues<Operators...>::num_stages == 1,
              typename FFTDescriptionValues<Operators...>::single_kernel,
              FFTNoMembers>
    {
    };

    // Descriptions without Size, Direction and Precision (for example the
    // partial sums while a description is built) have no members.
    template <typename... Operators>
    struct FFTDescription
        : FFTDescriptionMembers<has_required_operators<Operators...>,
                                Operators...>
    {
    };

    template <typename... Operators>
    constexpr bool has_unique_operators(FFTDescription<Operators...>)
    {
        constexpr FFTOperator ops[] = {Operators::op..., FFTOperator::Size};
        for (int i = 0; i < int(sizeof...(Operators)); i++)
        {
            for (int j = i + 1; j < int(sizeof...(Operators)); j++)
            {
                if (ops[i] == ops[j])
                {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T> constexpr auto as_fft_description(T)
    {
        if constexpr (is_fft_description<T>::value)
        {
            return T{};
        }
        else
        {
            return FFTDescription<T>{};
        }
    }

    template <typename... Left, typename... Right>
    constexpr auto concat_fft_descriptions(FFTDescription<Left...>,
                                           FFTDescription<Right...>)
    {
        return FFTDescription<Left..., Right...>{};
    }

    // Operators and descriptions add up to a description. The return type is
    // deduced, so the check below also runs inside decltype.
    template <
        typename L, typename R,
        std::enable_if_t<
            (is_fft_operator<L>::value || is_fft_description<L>::value) &&
                (is_fft_operator<R>::value || is_fft_description<R>::value),
            int> = 0>
    constexpr auto operator+(L, R)
    {
        constexpr auto result = concat_fft_descriptions(
            as_fft_description(L{}), as_fft_description(R{}));
        static_assert(has_unique_operators(result),
                      "Each FFT operator may appear at most once");
        return result;
    }

} // namespace gpufft

#endif // GPUFFT_DESCRIPTOR_H
