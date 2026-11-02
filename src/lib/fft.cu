// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "gpufft/fft.cuh"
#include "gpufft/fft_device.cuh"

// The host API runs every transform with kernels that take the size as a
// template parameter and read their launch parameters from the stage plan at
// compile time.

namespace gpufft
{
    static constexpr int kMinLogN = 1;
    static constexpr int kMinSpecialLogN = 11;
    static constexpr int kMinFFNTLogN = 12;

    static void check_n_power(int n_power, int min_log_n)
    {
        if (n_power < min_log_n || n_power > kMaxLogN)
        {
            throw std::invalid_argument("Invalid n_power range!");
        }
    }

    // The batch is the third grid dimension of the stage kernels (the low
    // ring kernels spread it over grid x), and kernels compute global memory
    // offsets (including the second operand of a multiplication) with 32-bit
    // integers.
    static void check_batch_size(int n_power, int batch_size)
    {
        if (batch_size <= 0 ||
            (n_power > kLowRingMaxLogN && batch_size > 65535))
        {
            throw std::invalid_argument("Invalid batch_size!");
        }
        if ((static_cast<long long>(batch_size) << n_power) >= (1LL << 31))
        {
            throw std::invalid_argument(
                "batch_size * ring size must be less than 2^31!");
        }
    }

    // Launch parameters of kernel I of a transform of size 2^LogN.
    template <typename T, type D, int LogN, int I> struct StageConfig
    {
        static constexpr StagePlan plan = stage_plan<T>(D, LogN);

        static_assert(I >= 0 && I < plan.count, "Stage index out of range");

        static constexpr KernelConfig config = plan.stage[I];

        static constexpr bool swap_grid_axes = plan.swap_grid_axes[I];
    };

    // ---------------- Kernels ----------------

    // Forward FFT of size 2^LogN <= 2^kLowRingMaxLogN in one kernel, of
    // blockDim.y polynomials per block (see LowRingThread): reads
    // polynomial_in (complex or real) and writes polynomial_out.
    template <int LogN, typename In, typename T>
    __global__ void ForwardLowRing(const In* polynomial_in,
                                   COMPLEX<T>* polynomial_out,
                                   COMPLEX<T>* root_of_unity_table,
                                   bool reduction_poly_check, int batch_size)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const LowRingThread thread = LowRingThreadState(LogN, batch_size);

        LowRingLoad(thread, shared_memory, polynomial_in);
        ForwardLowRingExecute<LogN>(shared_memory, root_of_unity_table,
                                    reduction_poly_check);
        LowRingStore(thread, shared_memory, polynomial_out);
    }

    // Inverse FFT of size 2^LogN <= 2^kLowRingMaxLogN in one kernel: reads
    // polynomial_in, times the second operand at polynomial_in + offset2 if
    // multiplication is set, and writes polynomial_out (complex or real)
    // times n_inverse.
    template <int LogN, typename T, typename Out>
    __global__ void
    InverseLowRing(const COMPLEX<T>* polynomial_in, Out* polynomial_out,
                   COMPLEX<T>* inverse_root_of_unity_table,
                   COMPLEX<T> n_inverse, bool multiplication, int offset2,
                   bool reduction_poly_check, int batch_size)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const LowRingThread thread = LowRingThreadState(LogN, batch_size);

        if (multiplication)
            LowRingLoadProduct(thread, shared_memory, polynomial_in, offset2);
        else
            LowRingLoad(thread, shared_memory, polynomial_in);
        InverseLowRingExecute<LogN>(shared_memory, inverse_root_of_unity_table,
                                    reduction_poly_check);
        LowRingStoreScaled(thread, shared_memory, polynomial_out, n_inverse);
    }

    // Stage I of a forward FFT: reads polynomial_in (complex or real) and
    // writes polynomial_out.
    template <int LogN, int I, typename In, typename T>
    __global__ void
    ForwardStage(const In* polynomial_in, COMPLEX<T>* polynomial_out,
                 COMPLEX<T>* root_of_unity_table, bool reduction_poly_check)
    {
        using Config = StageConfig<T, FORWARD, LogN, I>;
        constexpr KernelConfig k = Config::config;

        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread = StageThreadState<Config::swap_grid_axes>(
            k.logm, k.outer_iteration_count, LogN);
        const location_t global_address = poly_address(thread, LogN);

        StageLoad(thread, shared_memory, polynomial_in, global_address);
        DecreasingStageExecute<CooleyTukey>(
            thread, shared_memory, root_of_unity_table,
            PolyRoots<false>{reduction_poly_check, LogN - k.logm - 1,
                             (location_t) 1 << k.logm},
            k.shared_index, k.outer_iteration_count, k.not_last_kernel,
            k.shared_index - 5);
        StageStore(thread, shared_memory, polynomial_out, global_address);
    }

    // Stage I of an inverse FFT: reads polynomial_in, in the first stage
    // times the second operand at polynomial_in + offset2 if multiplication
    // is set, and writes polynomial_out (complex or real), in the last stage
    // times n_inverse.
    template <int LogN, int I, typename T, typename Out>
    __global__ void
    InverseStage(const COMPLEX<T>* polynomial_in, Out* polynomial_out,
                 COMPLEX<T>* inverse_root_of_unity_table, COMPLEX<T> n_inverse,
                 bool multiplication, int offset2, bool reduction_poly_check)
    {
        using Config = StageConfig<T, INVERSE, LogN, I>;
        constexpr KernelConfig k = Config::config;

        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread = StageThreadState<Config::swap_grid_axes>(
            k.k, k.outer_iteration_count, LogN);
        const location_t global_address = poly_address(thread, LogN);

        if (I == 0 && multiplication)
            StageLoadProduct(thread, shared_memory, polynomial_in,
                             global_address, offset2);
        else
            StageLoad(thread, shared_memory, polynomial_in, global_address);
        IncreasingStageExecute<GentlemanSande>(
            thread, shared_memory, inverse_root_of_unity_table,
            PolyRoots<true>{reduction_poly_check, LogN - k.logm - 1,
                            (location_t) 1 << k.logm},
            (k.shared_index + 1) - k.outer_iteration_count,
            k.outer_iteration_count);
        // The inverse plan stores the last_kernel flag in not_last_kernel.
        if constexpr (k.not_last_kernel)
            StageStoreScaled(thread, shared_memory, polynomial_out,
                             global_address, n_inverse);
        else
            StageStore(thread, shared_memory, polynomial_out, global_address);
    }

    // Stage I of a forward Special FFT, which runs the inverse plan.
    template <int LogN, int I, typename T>
    __global__ void SpecialForwardStage(COMPLEX<T>* polynomial,
                                        COMPLEX<T>* inverse_root_of_unity_table)
    {
        using Config = StageConfig<T, INVERSE, LogN, I>;
        constexpr KernelConfig k = Config::config;

        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread = StageThreadState<Config::swap_grid_axes>(
            k.k, k.outer_iteration_count, LogN);
        const location_t global_address = poly_address(thread, LogN);

        StageLoad(thread, shared_memory, polynomial, global_address);
        IncreasingStageExecute<CooleyTukey>(
            thread, shared_memory, inverse_root_of_unity_table,
            SpecialRoots<true>{(location_t) 1 << (LogN - k.logm - 1)},
            (k.shared_index + 1) - k.outer_iteration_count,
            k.outer_iteration_count);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    // Stage I of an inverse Special FFT, which runs the forward plan; the
    // last stage multiplies by n_inverse.
    template <int LogN, int I, typename T>
    __global__ void SpecialInverseStage(COMPLEX<T>* polynomial,
                                        COMPLEX<T>* root_of_unity_table,
                                        COMPLEX<T> n_inverse)
    {
        using Config = StageConfig<T, FORWARD, LogN, I>;
        constexpr KernelConfig k = Config::config;

        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread = StageThreadState<Config::swap_grid_axes>(
            k.logm, k.outer_iteration_count, LogN);
        const location_t global_address = poly_address(thread, LogN);

        StageLoad(thread, shared_memory, polynomial, global_address);
        DecreasingStageExecute<GentlemanSande>(
            thread, shared_memory, root_of_unity_table,
            SpecialRoots<false>{(location_t) 1 << (LogN - k.logm - 1)},
            k.shared_index, k.outer_iteration_count, k.not_last_kernel,
            k.shared_index - 5);
        if constexpr (k.not_last_kernel)
            StageStore(thread, shared_memory, polynomial, global_address);
        else
            StageStoreScaled(thread, shared_memory, polynomial, global_address,
                             n_inverse);
    }

    // Stage I of a forward FFNT of 2^(LogN + 1) real values, an FFT of size
    // 2^LogN: the first stage folds and twists polynomial_in, every stage
    // writes polynomial and the later stages read it in place.
    template <int LogN, int I, typename T>
    __global__ void FFNTForwardStage(const T* polynomial_in,
                                     COMPLEX<T>* polynomial,
                                     const COMPLEX<T>* twist_table,
                                     COMPLEX<T>* root_of_unity_table)
    {
        using Config = StageConfig<T, FORWARD, LogN, I>;
        constexpr KernelConfig k = Config::config;

        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread = StageThreadState<Config::swap_grid_axes>(
            k.logm, k.outer_iteration_count, LogN);
        const location_t global_address = poly_address(thread, LogN);

        if constexpr (I == 0)
            StageLoadFoldTwist(thread, shared_memory, polynomial_in,
                               poly_address(thread, LogN + 1), twist_table,
                               LogN);
        else
            StageLoad(thread, shared_memory, polynomial, global_address);
        DecreasingStageExecute<CooleyTukey>(
            thread, shared_memory, root_of_unity_table,
            CyclicRoots<false>{LogN - k.logm - 1}, k.shared_index,
            k.outer_iteration_count, k.not_last_kernel, k.shared_index - 5);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    // Stage I of an inverse FFNT: every stage reads polynomial, the first
    // one times the second operand at polynomial + offset2 if multiplication
    // is set; the last stage multiplies by n_inverse, untwists and unfolds
    // into polynomial_out, the others write polynomial in place.
    template <int LogN, int I, typename T>
    __global__ void FFNTInverseStage(COMPLEX<T>* polynomial, T* polynomial_out,
                                     const COMPLEX<T>* untwist_table,
                                     COMPLEX<T>* inverse_root_of_unity_table,
                                     COMPLEX<T> n_inverse, bool multiplication,
                                     int offset2)
    {
        using Config = StageConfig<T, INVERSE, LogN, I>;
        constexpr KernelConfig k = Config::config;

        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread = StageThreadState<Config::swap_grid_axes>(
            k.k, k.outer_iteration_count, LogN);
        const location_t global_address = poly_address(thread, LogN);

        if (I == 0 && multiplication)
            StageLoadProduct(thread, shared_memory, polynomial, global_address,
                             offset2);
        else
            StageLoad(thread, shared_memory, polynomial, global_address);
        IncreasingStageExecute<GentlemanSande>(
            thread, shared_memory, inverse_root_of_unity_table,
            CyclicRoots<true>{LogN - k.logm - 1},
            (k.shared_index + 1) - k.outer_iteration_count,
            k.outer_iteration_count);
        if constexpr (k.not_last_kernel)
            StageStoreUntwistUnfold(thread, shared_memory, polynomial_out,
                                    poly_address(thread, LogN + 1),
                                    untwist_table, n_inverse, LogN);
        else
            StageStore(thread, shared_memory, polynomial, global_address);
    }

    // ---------------- Launch ----------------

    // Calls f(std::integral_constant<int, n_power>()) if n_power is in
    // [LogN, MaxLogN]; the host API checks the range before.
    template <int LogN, int MaxLogN, typename F>
    void dispatch_log_n(int n_power, F&& f)
    {
        if constexpr (LogN <= MaxLogN)
        {
            if (n_power == LogN)
            {
                f(std::integral_constant<int, LogN>());
            }
            else
            {
                dispatch_log_n<LogN + 1, MaxLogN>(n_power, f);
            }
        }
    }

    template <typename F, int... I>
    void for_each_stage(F&& f, std::integer_sequence<int, I...>)
    {
        (f(std::integral_constant<int, I>()), ...);
    }

    // Calls f(std::integral_constant<int, I>()) for the stages of the plan
    // of direction D and size 2^LogN, in launch order.
    template <typename T, type D, int LogN, typename F>
    void for_each_stage(F&& f)
    {
        for_each_stage(
            f, std::make_integer_sequence<int, stage_plan<T>(D, LogN).count>());
    }

    // Launches kernel with the launch parameters of stage I of the plan of
    // direction D and size 2^LogN; blockIdx.z is the polynomial.
    template <typename T, type D, int LogN, int I, typename Kernel,
              typename... Args>
    void launch_stage(Kernel kernel, int batch_size, cudaStream_t stream,
                      Args... args)
    {
        constexpr KernelConfig k = StageConfig<T, D, LogN, I>::config;

        kernel<<<dim3(k.griddim_x, k.griddim_y, batch_size),
                 dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
            args...);
        GPUFFT_CUDA_CHECK(cudaGetLastError());
    }

    // Launches the low ring kernel of size 2^LogN: 256 threads, which
    // transform 2^(kLowRingMaxLogN - LogN) polynomials per block.
    template <typename T, int LogN, typename Kernel, typename... Args>
    void launch_low_ring(Kernel kernel, int batch_size, cudaStream_t stream,
                         Args... args)
    {
        constexpr int polys_per_block = 1 << (kLowRingMaxLogN - LogN);

        kernel<<<(batch_size + polys_per_block - 1) / polys_per_block,
                 dim3(1 << (LogN - 1), polys_per_block),
                 (size_t(polys_per_block) << LogN) * sizeof(COMPLEX<T>),
                 stream>>>(args...);
        GPUFFT_CUDA_CHECK(cudaGetLastError());
    }

    // Small sizes run in the low ring kernel. Otherwise the first stage reads
    // polynomial_in; every stage writes polynomial_out and the later stages
    // read it in place.
    template <int LogN, typename In, typename T>
    void forward_stages(const In* polynomial_in, COMPLEX<T>* polynomial_out,
                        COMPLEX<T>* root_of_unity_table,
                        bool reduction_poly_check, int batch_size,
                        cudaStream_t stream)
    {
        if constexpr (LogN <= kLowRingMaxLogN)
        {
            launch_low_ring<T, LogN>(ForwardLowRing<LogN, In, T>, batch_size,
                                     stream, polynomial_in, polynomial_out,
                                     root_of_unity_table, reduction_poly_check,
                                     batch_size);
        }
        else
        {
            for_each_stage<T, FORWARD, LogN>(
                [&](auto stage)
                {
                    constexpr int I = decltype(stage)::value;
                    if constexpr (I == 0)
                        launch_stage<T, FORWARD, LogN, I>(
                            ForwardStage<LogN, I, In, T>, batch_size, stream,
                            polynomial_in, polynomial_out, root_of_unity_table,
                            reduction_poly_check);
                    else
                        launch_stage<T, FORWARD, LogN, I>(
                            ForwardStage<LogN, I, COMPLEX<T>, T>, batch_size,
                            stream, polynomial_out, polynomial_out,
                            root_of_unity_table, reduction_poly_check);
                });
        }
    }

    // Small sizes run in the low ring kernel. Otherwise every stage reads
    // polynomial_in; the last one writes polynomial_out, the others write
    // polynomial_in in place.
    template <int LogN, typename T, typename Out>
    void inverse_stages(COMPLEX<T>* polynomial_in, Out* polynomial_out,
                        COMPLEX<T>* inverse_root_of_unity_table,
                        COMPLEX<T> n_inverse, bool multiplication, int offset2,
                        bool reduction_poly_check, int batch_size,
                        cudaStream_t stream)
    {
        if constexpr (LogN <= kLowRingMaxLogN)
        {
            launch_low_ring<T, LogN>(
                InverseLowRing<LogN, T, Out>, batch_size, stream, polynomial_in,
                polynomial_out, inverse_root_of_unity_table, n_inverse,
                multiplication, offset2, reduction_poly_check, batch_size);
        }
        else
        {
            constexpr int count = stage_plan<T>(INVERSE, LogN).count;

            for_each_stage<T, INVERSE, LogN>(
                [&](auto stage)
                {
                    constexpr int I = decltype(stage)::value;
                    if constexpr (I + 1 == count)
                        launch_stage<T, INVERSE, LogN, I>(
                            InverseStage<LogN, I, T, Out>, batch_size, stream,
                            polynomial_in, polynomial_out,
                            inverse_root_of_unity_table, n_inverse,
                            multiplication, offset2, reduction_poly_check);
                    else
                        launch_stage<T, INVERSE, LogN, I>(
                            InverseStage<LogN, I, T, COMPLEX<T>>, batch_size,
                            stream, polynomial_in, polynomial_in,
                            inverse_root_of_unity_table, n_inverse,
                            multiplication, offset2, reduction_poly_check);
                });
        }
    }

    // ---------------- Host API ----------------

    template <typename T>
    __host__ void
    GPU_FFT(COMPLEX<T>* device_inout, COMPLEX<T>* root_of_unity_table,
            fft_configuration<T> cfg, int batch_size, bool multiplication)
    {
        check_n_power(cfg.n_power, kMinLogN);
        check_batch_size(cfg.n_power, batch_size);

        const bool x_n_minus =
            (cfg.reduction_poly == ReductionPolynomial::X_N_minus);
        dispatch_log_n<kMinLogN, kMaxLogN>(
            cfg.n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if (cfg.fft_type == FORWARD)
                    forward_stages<LogN>(device_inout, device_inout,
                                         root_of_unity_table, x_n_minus,
                                         batch_size, cfg.stream);
                else if (cfg.fft_type == INVERSE)
                    inverse_stages<LogN>(device_inout, device_inout,
                                         root_of_unity_table, cfg.mod_inverse,
                                         multiplication,
                                         batch_size << cfg.n_power, x_n_minus,
                                         batch_size, cfg.stream);
            });
    }

    template <typename T>
    __host__ void
    GPU_FFT(T* device_fix_inout, COMPLEX<T>* device_Complex64_inout,
            COMPLEX<T>* root_of_unity_table, fft_configuration<T> cfg,
            int batch_size, bool multiplication)
    {
        check_n_power(cfg.n_power, kMinLogN);
        check_batch_size(cfg.n_power, batch_size);

        const bool x_n_minus =
            (cfg.reduction_poly == ReductionPolynomial::X_N_minus);
        dispatch_log_n<kMinLogN, kMaxLogN>(
            cfg.n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if (cfg.fft_type == FORWARD)
                    forward_stages<LogN>(
                        device_fix_inout, device_Complex64_inout,
                        root_of_unity_table, x_n_minus, batch_size, cfg.stream);
                else if (cfg.fft_type == INVERSE)
                    inverse_stages<LogN>(device_Complex64_inout,
                                         device_fix_inout, root_of_unity_table,
                                         cfg.mod_inverse, multiplication,
                                         batch_size << cfg.n_power, x_n_minus,
                                         batch_size, cfg.stream);
            });
    }

    template <typename T>
    __host__ void GPU_Special_FFT(COMPLEX<T>* device_inout,
                                  COMPLEX<T>* root_of_unity_table,
                                  fft_configuration<T> cfg, int batch_size)
    {
        check_n_power(cfg.n_power, kMinSpecialLogN);
        check_batch_size(cfg.n_power, batch_size);

        dispatch_log_n<kMinSpecialLogN, kMaxLogN>(
            cfg.n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if (cfg.fft_type == FORWARD)
                    for_each_stage<T, INVERSE, LogN>(
                        [&](auto stage)
                        {
                            constexpr int I = decltype(stage)::value;
                            launch_stage<T, INVERSE, LogN, I>(
                                SpecialForwardStage<LogN, I, T>, batch_size,
                                cfg.stream, device_inout, root_of_unity_table);
                        });
                else if (cfg.fft_type == INVERSE)
                    for_each_stage<T, FORWARD, LogN>(
                        [&](auto stage)
                        {
                            constexpr int I = decltype(stage)::value;
                            launch_stage<T, FORWARD, LogN, I>(
                                SpecialInverseStage<LogN, I, T>, batch_size,
                                cfg.stream, device_inout, root_of_unity_table,
                                cfg.mod_inverse);
                        });
            });
    }

    template <typename T>
    __host__ void
    GPU_FFNT(T* device_inout, COMPLEX<T>* device_temp, COMPLEX<T>* twist_table,
             COMPLEX<T>* root_of_unity_table, fft_configuration<T> cfg,
             int batch_size, bool multiplication)
    {
        check_n_power(cfg.n_power, kMinFFNTLogN);
        check_batch_size(cfg.n_power, batch_size);

        // The FFT of the folded values has size 2^(n_power - 1).
        dispatch_log_n<kMinFFNTLogN - 1, kMaxLogN - 1>(
            cfg.n_power - 1,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if (cfg.fft_type == FORWARD)
                    for_each_stage<T, FORWARD, LogN>(
                        [&](auto stage)
                        {
                            constexpr int I = decltype(stage)::value;
                            launch_stage<T, FORWARD, LogN, I>(
                                FFNTForwardStage<LogN, I, T>, batch_size,
                                cfg.stream, device_inout, device_temp,
                                twist_table, root_of_unity_table);
                        });
                else if (cfg.fft_type == INVERSE)
                    for_each_stage<T, INVERSE, LogN>(
                        [&](auto stage)
                        {
                            constexpr int I = decltype(stage)::value;
                            launch_stage<T, INVERSE, LogN, I>(
                                FFNTInverseStage<LogN, I, T>, batch_size,
                                cfg.stream, device_temp, device_inout,
                                twist_table, root_of_unity_table,
                                cfg.mod_inverse, multiplication,
                                batch_size << LogN);
                        });
            });
    }

    template __host__ void GPU_FFT(Complex32* device_inout,
                                   Complex32* root_of_unity_table,
                                   fft_configuration<Float32> cfg,
                                   int batch_size, bool multiplication);

    template __host__ void GPU_FFT(Complex64* device_inout,
                                   Complex64* root_of_unity_table,
                                   fft_configuration<Float64> cfg,
                                   int batch_size, bool multiplication);

    template __host__ void GPU_FFT(Float32* device_fix_inout,
                                   Complex32* device_Complex64_inout,
                                   Complex32* root_of_unity_table,
                                   fft_configuration<Float32> cfg,
                                   int batch_size, bool multiplication);

    template __host__ void GPU_FFT(Float64* device_fix_inout,
                                   Complex64* device_Complex64_inout,
                                   Complex64* root_of_unity_table,
                                   fft_configuration<Float64> cfg,
                                   int batch_size, bool multiplication);

    template __host__ void GPU_Special_FFT(Complex32* device_inout,
                                           Complex32* root_of_unity_table,
                                           fft_configuration<Float32> cfg,
                                           int batch_size);

    template __host__ void GPU_Special_FFT(Complex64* device_inout,
                                           Complex64* root_of_unity_table,
                                           fft_configuration<Float64> cfg,
                                           int batch_size);

    template __host__ void GPU_FFNT(Float32* device_inout,
                                    Complex32* device_temp,
                                    Complex32* twist_table,
                                    Complex32* root_of_unity_table,
                                    fft_configuration<Float32> cfg,
                                    int batch_size, bool multiplication);

    template __host__ void GPU_FFNT(Float64* device_inout,
                                    Complex64* device_temp,
                                    Complex64* twist_table,
                                    Complex64* root_of_unity_table,
                                    fft_configuration<Float64> cfg,
                                    int batch_size, bool multiplication);

} // namespace gpufft
