// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include <stdexcept>

#include "gpufft/fft.cuh"
#include "gpufft/fft_device.cuh"

namespace gpufft
{
    // The batch is the third grid dimension, and kernels compute global
    // memory offsets (including the second operand of a multiplication) with
    // 32-bit integers.
    static void check_batch_size(int n_power, int batch_size)
    {
        if (batch_size <= 0 || batch_size > 65535)
        {
            throw std::invalid_argument("Invalid batch_size!");
        }
        if ((static_cast<long long>(batch_size) << n_power) >= (1LL << 31))
        {
            throw std::invalid_argument(
                "batch_size * ring size must be less than 2^31!");
        }
    }

    template <typename T>
    __global__ void
    ForwardCore(COMPLEX<T>* polynomial, COMPLEX<T>* root_of_unity_table,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel, bool reduction_poly_check)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(logm, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        StageLoad(thread, shared_memory, polynomial, global_address);
        DecreasingStageExecute<CooleyTukey>(
            thread, shared_memory, root_of_unity_table,
            PolyRoots<false>{reduction_poly_check, N_power - logm - 1,
                             (location_t) 1 << logm},
            shared_index, outer_iteration_count, not_last_kernel,
            shared_index - 5);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    template <typename T>
    __global__ void
    InverseCore(COMPLEX<T>* polynomial, COMPLEX<T>* inverse_root_of_unity_table,
                int shared_index, int logm, int k, int outer_iteration_count,
                int N_power, COMPLEX<T> n_inverse, bool last_kernel,
                bool NTT_mult, int offset2, bool reduction_poly_check)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(k, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        if (NTT_mult)
            StageLoadProduct(thread, shared_memory, polynomial, global_address,
                             offset2);
        else
            StageLoad(thread, shared_memory, polynomial, global_address);
        IncreasingStageExecute<GentlemanSande>(
            thread, shared_memory, inverse_root_of_unity_table,
            PolyRoots<true>{reduction_poly_check, N_power - logm - 1,
                            (location_t) 1 << logm},
            (shared_index + 1) - outer_iteration_count, outer_iteration_count);
        if (last_kernel)
            StageStoreScaled(thread, shared_memory, polynomial, global_address,
                             n_inverse);
        else
            StageStore(thread, shared_memory, polynomial, global_address);
    }

    template <typename T>
    __global__ void ForwardCore(T* input, COMPLEX<T>* polynomial,
                                COMPLEX<T>* root_of_unity_table,
                                int shared_index, int logm,
                                int outer_iteration_count, int N_power,
                                bool not_last_kernel, bool reduction_poly_check)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(logm, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        StageLoad(thread, shared_memory, input, global_address);
        DecreasingStageExecute<CooleyTukey>(
            thread, shared_memory, root_of_unity_table,
            PolyRoots<false>{reduction_poly_check, N_power - logm - 1,
                             (location_t) 1 << logm},
            shared_index, outer_iteration_count, not_last_kernel,
            shared_index - 5);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    template <typename T>
    __global__ void
    InverseCore(T* output, COMPLEX<T>* polynomial,
                COMPLEX<T>* inverse_root_of_unity_table, int shared_index,
                int logm, int k, int outer_iteration_count, int N_power,
                COMPLEX<T> n_inverse, bool last_kernel, bool NTT_mult,
                int offset2, bool reduction_poly_check)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(k, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        if (NTT_mult)
            StageLoadProduct(thread, shared_memory, polynomial, global_address,
                             offset2);
        else
            StageLoad(thread, shared_memory, polynomial, global_address);
        IncreasingStageExecute<GentlemanSande>(
            thread, shared_memory, inverse_root_of_unity_table,
            PolyRoots<true>{reduction_poly_check, N_power - logm - 1,
                            (location_t) 1 << logm},
            (shared_index + 1) - outer_iteration_count, outer_iteration_count);
        if (last_kernel)
            StageStoreScaled(thread, shared_memory, output, global_address,
                             n_inverse);
        else
            StageStore(thread, shared_memory, output, global_address);
    }

    template <typename T>
    __host__ void
    GPU_FFT(COMPLEX<T>* device_inout, COMPLEX<T>* root_of_unity_table,
            fft_configuration<T> cfg, int batch_size, bool multiplication)
    {
        if ((cfg.n_power <= 11 || cfg.n_power >= 25))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        check_batch_size(cfg.n_power, batch_size);

        const StagePlan plan = stage_plan<T>(cfg.fft_type, cfg.n_power);

        switch (cfg.fft_type)
        {
            case FORWARD:
                for (int i = 0; i < plan.count; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    ForwardCore<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_inout, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }
                break;
            case INVERSE:
            {
                auto& current_kernel_params = plan.stage[0];
                InverseCore<<<dim3(current_kernel_params.griddim_x,
                                   current_kernel_params.griddim_y, batch_size),
                              dim3(current_kernel_params.blockdim_x,
                                   current_kernel_params.blockdim_y),
                              current_kernel_params.shared_memory,
                              cfg.stream>>>(
                    device_inout, root_of_unity_table,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    multiplication, (batch_size << cfg.n_power),
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUFFT_CUDA_CHECK(cudaGetLastError());

                for (int i = 1; i < plan.count; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    InverseCore<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_inout, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm, current_kernel_params.k,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, cfg.mod_inverse,
                        current_kernel_params.not_last_kernel, false,
                        (batch_size << cfg.n_power),
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }
            }
            break;

            default:
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_FFT(T* device_fix_inout, COMPLEX<T>* device_Complex64_inout,
            COMPLEX<T>* root_of_unity_table, fft_configuration<T> cfg,
            int batch_size, bool multiplication)
    {
        if ((cfg.n_power <= 11 || cfg.n_power >= 25))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        check_batch_size(cfg.n_power, batch_size);

        const StagePlan plan = stage_plan<T>(cfg.fft_type, cfg.n_power);

        switch (cfg.fft_type)
        {
            case FORWARD:
            {
                auto& current_kernel_params = plan.stage[0];
                ForwardCore<<<dim3(current_kernel_params.griddim_x,
                                   current_kernel_params.griddim_y, batch_size),
                              dim3(current_kernel_params.blockdim_x,
                                   current_kernel_params.blockdim_y),
                              current_kernel_params.shared_memory,
                              cfg.stream>>>(
                    device_fix_inout, device_Complex64_inout,
                    root_of_unity_table, current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUFFT_CUDA_CHECK(cudaGetLastError());

                for (int i = 1; i < plan.count; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    ForwardCore<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_Complex64_inout, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }
            }
            break;
            case INVERSE:
            {
                auto& current_kernel_params = plan.stage[0];
                InverseCore<<<dim3(current_kernel_params.griddim_x,
                                   current_kernel_params.griddim_y, batch_size),
                              dim3(current_kernel_params.blockdim_x,
                                   current_kernel_params.blockdim_y),
                              current_kernel_params.shared_memory,
                              cfg.stream>>>(
                    device_Complex64_inout, root_of_unity_table,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    multiplication, (batch_size << cfg.n_power),
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUFFT_CUDA_CHECK(cudaGetLastError());

                for (int i = 1; i < plan.count; i++)
                {
                    if (i == (plan.count - 1))
                    {
                        auto& current_kernel_params = plan.stage[i];
                        InverseCore<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_fix_inout, device_Complex64_inout,
                            root_of_unity_table,
                            current_kernel_params.shared_index,
                            current_kernel_params.logm, current_kernel_params.k,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power, cfg.mod_inverse,
                            current_kernel_params.not_last_kernel, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        GPUFFT_CUDA_CHECK(cudaGetLastError());
                    }
                    else
                    {
                        auto& current_kernel_params = plan.stage[i];
                        InverseCore<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_Complex64_inout, root_of_unity_table,
                            current_kernel_params.shared_index,
                            current_kernel_params.logm, current_kernel_params.k,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power, cfg.mod_inverse,
                            current_kernel_params.not_last_kernel, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        GPUFFT_CUDA_CHECK(cudaGetLastError());
                    }
                }
            }
            break;

            default:
                break;
        }
    }

    template <typename T>
    __global__ void
    Special_InverseCore(COMPLEX<T>* polynomial, COMPLEX<T>* root_of_unity_table,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, COMPLEX<T> n_inverse, bool not_last_kernel)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(logm, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        StageLoad(thread, shared_memory, polynomial, global_address);
        DecreasingStageExecute<GentlemanSande>(
            thread, shared_memory, root_of_unity_table,
            SpecialRoots<false>{(location_t) 1 << (N_power - logm - 1)}, 8,
            outer_iteration_count, not_last_kernel, 3);
        if (not_last_kernel)
            StageStore(thread, shared_memory, polynomial, global_address);
        else
            StageStoreScaled(thread, shared_memory, polynomial, global_address,
                             n_inverse);
    }

    template <typename T>
    __global__ void Special_ForwardCore(COMPLEX<T>* polynomial,
                                        COMPLEX<T>* inverse_root_of_unity_table,
                                        int shared_index, int logm, int k,
                                        int outer_iteration_count, int N_power)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(k, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        StageLoad(thread, shared_memory, polynomial, global_address);
        IncreasingStageExecute<CooleyTukey>(
            thread, shared_memory, inverse_root_of_unity_table,
            SpecialRoots<true>{(location_t) 1 << (N_power - logm - 1)},
            (shared_index + 1) - outer_iteration_count, outer_iteration_count);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    template <typename T>
    __host__ void GPU_Special_FFT(COMPLEX<T>* device_inout,
                                  COMPLEX<T>* root_of_unity_table,
                                  fft_configuration<T> cfg, int batch_size)
    {
        if ((cfg.n_power <= 10 || cfg.n_power >= 25))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        check_batch_size(cfg.n_power, batch_size);

        // The Special FFT runs the inverse plan forward and the forward plan
        // inverse.
        const StagePlan plan = stage_plan<T>(
            (cfg.fft_type == FORWARD) ? INVERSE : FORWARD, cfg.n_power);

        switch (cfg.fft_type)
        {
            case FORWARD:
                for (int i = 0; i < plan.count; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    Special_ForwardCore<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_inout, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm, current_kernel_params.k,
                        current_kernel_params.outer_iteration_count, cfg.n_power

                    );
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }
                break;
            case INVERSE:
            {
                for (int i = 0; i < plan.count - 1; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    Special_InverseCore<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_inout, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, cfg.mod_inverse, true);
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }

                auto& current_kernel_params = plan.stage[plan.count - 1];
                Special_InverseCore<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_inout, root_of_unity_table,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, false);
                GPUFFT_CUDA_CHECK(cudaGetLastError());
            }
            break;

            default:
                break;
        }
    }

    /////////////////////////////////////////////
    /////////////////////////////////////////////
    /////////////////////////////////////////////

    template <typename T>
    __global__ void
    ForwardFFNTCoreUnique(T* polynomial_in, COMPLEX<T>* polynomial_out,
                          COMPLEX<T>* twist_table,
                          COMPLEX<T>* root_of_unity_table, int shared_index,
                          int logm, int outer_iteration_count, int N_power)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(logm, outer_iteration_count, N_power);

        StageLoadFoldTwist(thread, shared_memory, polynomial_in,
                           poly_address(thread, N_power + 1), twist_table,
                           N_power);
        DecreasingStageExecute<CooleyTukey>(
            thread, shared_memory, root_of_unity_table,
            CyclicRoots<false>{N_power - logm - 1}, shared_index,
            outer_iteration_count, true, 0);
        __syncthreads();
        StageStore(thread, shared_memory, polynomial_out,
                   poly_address(thread, N_power));
    }

    template <typename T>
    __global__ void ForwardFFNTCoreRegular(COMPLEX<T>* polynomial,
                                           COMPLEX<T>* root_of_unity_table,
                                           int shared_index, int logm,
                                           int outer_iteration_count,
                                           int N_power, bool not_last_kernel)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(logm, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        StageLoad(thread, shared_memory, polynomial, global_address);
        DecreasingStageExecute<CooleyTukey>(
            thread, shared_memory, root_of_unity_table,
            CyclicRoots<false>{N_power - logm - 1}, shared_index,
            outer_iteration_count, not_last_kernel, 3);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    template <typename T>
    __global__ void InverseFFNTCoreRegular(
        COMPLEX<T>* polynomial, COMPLEX<T>* inverse_root_of_unity_table,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, bool NTT_mult, int offset2)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(k, outer_iteration_count, N_power);
        const location_t global_address = poly_address(thread, N_power);

        if (NTT_mult)
            StageLoadProduct(thread, shared_memory, polynomial, global_address,
                             offset2);
        else
            StageLoad(thread, shared_memory, polynomial, global_address);
        IncreasingStageExecute<GentlemanSande>(
            thread, shared_memory, inverse_root_of_unity_table,
            CyclicRoots<true>{N_power - logm - 1},
            (shared_index + 1) - outer_iteration_count, outer_iteration_count);
        StageStore(thread, shared_memory, polynomial, global_address);
    }

    template <typename T>
    __global__ void InverseFFNTCoreUnique(
        COMPLEX<T>* polynomial_in, T* polynomial_out, COMPLEX<T>* untwist_table,
        COMPLEX<T>* inverse_root_of_unity_table, int shared_index, int logm,
        int k, int outer_iteration_count, int N_power, COMPLEX<T> n_inverse)
    {
        COMPLEX<T>* shared_memory = stage_shared<T>();
        const StageThread thread =
            StageThreadState(k, outer_iteration_count, N_power);

        StageLoad(thread, shared_memory, polynomial_in,
                  poly_address(thread, N_power));
        IncreasingStageExecute<GentlemanSande>(
            thread, shared_memory, inverse_root_of_unity_table,
            CyclicRoots<true>{N_power - logm - 1},
            (shared_index + 1) - outer_iteration_count, outer_iteration_count);
        StageStoreUntwistUnfold(thread, shared_memory, polynomial_out,
                                poly_address(thread, N_power + 1),
                                untwist_table, n_inverse, N_power);
    }

    template <typename T>
    __host__ void
    GPU_FFNT(T* device_inout, COMPLEX<T>* device_temp, COMPLEX<T>* twist_table,
             COMPLEX<T>* root_of_unity_table, fft_configuration<T> cfg,
             int batch_size, bool multiplication)
    {
        if ((cfg.n_power <= 11 || cfg.n_power >= 25))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        check_batch_size(cfg.n_power, batch_size);

        const StagePlan plan = stage_plan<T>(cfg.fft_type, cfg.n_power - 1);

        switch (cfg.fft_type)
        {
            case FORWARD:
            {
                auto& current_kernel_params = plan.stage[0];
                ForwardFFNTCoreUnique<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_inout, device_temp, twist_table, root_of_unity_table,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count,
                    cfg.n_power - 1);
                GPUFFT_CUDA_CHECK(cudaGetLastError());

                for (int i = 1; i < plan.count - 1; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    ForwardFFNTCoreRegular<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_temp, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power - 1, true);
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }

                auto& last_kernel_params = plan.stage[plan.count - 1];
                ForwardFFNTCoreRegular<<<
                    dim3(last_kernel_params.griddim_x,
                         last_kernel_params.griddim_y, batch_size),
                    dim3(last_kernel_params.blockdim_x,
                         last_kernel_params.blockdim_y),
                    last_kernel_params.shared_memory, cfg.stream>>>(
                    device_temp, root_of_unity_table,
                    last_kernel_params.shared_index, last_kernel_params.logm,
                    last_kernel_params.outer_iteration_count, cfg.n_power - 1,
                    false);
                GPUFFT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            case INVERSE:
            {
                auto& current_kernel_params = plan.stage[0];
                InverseFFNTCoreRegular<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_temp, root_of_unity_table,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count,
                    cfg.n_power - 1, multiplication,
                    (batch_size << (cfg.n_power - 1)));
                GPUFFT_CUDA_CHECK(cudaGetLastError());

                for (int i = 1; i < plan.count - 1; i++)
                {
                    auto& current_kernel_params = plan.stage[i];
                    InverseFFNTCoreRegular<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_temp, root_of_unity_table,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm, current_kernel_params.k,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power - 1, false,
                        (batch_size << (cfg.n_power - 1)));
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }

                auto& last_kernel_params = plan.stage[plan.count - 1];
                InverseFFNTCoreUnique<<<
                    dim3(last_kernel_params.griddim_x,
                         last_kernel_params.griddim_y, batch_size),
                    dim3(last_kernel_params.blockdim_x,
                         last_kernel_params.blockdim_y),
                    last_kernel_params.shared_memory, cfg.stream>>>(
                    device_temp, device_inout, twist_table, root_of_unity_table,
                    last_kernel_params.shared_index, last_kernel_params.logm,
                    last_kernel_params.k,
                    last_kernel_params.outer_iteration_count, cfg.n_power - 1,
                    cfg.mod_inverse);
                GPUFFT_CUDA_CHECK(cudaGetLastError());
            }
            break;

            default:
                break;
        }
    }

    /////////////////////////////////////////////
    /////////////////////////////////////////////
    /////////////////////////////////////////////

    template __global__ void
    ForwardCore(Complex32* polynomial, Complex32* root_of_unity_table,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel, bool reduction_poly_check);

    template __global__ void
    ForwardCore(Complex64* polynomial, Complex64* root_of_unity_table,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel, bool reduction_poly_check);

    template __global__ void
    InverseCore(Complex32* polynomial, Complex32* inverse_root_of_unity_table,
                int shared_index, int logm, int k, int outer_iteration_count,
                int N_power, Complex32 n_inverse, bool last_kernel,
                bool NTT_mult, int offset2, bool reduction_poly_check);

    template __global__ void
    InverseCore(Complex64* polynomial, Complex64* inverse_root_of_unity_table,
                int shared_index, int logm, int k, int outer_iteration_count,
                int N_power, Complex64 n_inverse, bool last_kernel,
                bool NTT_mult, int offset2, bool reduction_poly_check);

    template __global__ void ForwardCore(Float32* input, Complex32* polynomial,
                                         Complex32* root_of_unity_table,
                                         int shared_index, int logm,
                                         int outer_iteration_count, int N_power,
                                         bool not_last_kernel,
                                         bool reduction_poly_check);

    template __global__ void ForwardCore(Float64* input, Complex64* polynomial,
                                         Complex64* root_of_unity_table,
                                         int shared_index, int logm,
                                         int outer_iteration_count, int N_power,
                                         bool not_last_kernel,
                                         bool reduction_poly_check);

    template __global__ void InverseCore(Float32* output, Complex32* polynomial,
                                         Complex32* inverse_root_of_unity_table,
                                         int shared_index, int logm, int k,
                                         int outer_iteration_count, int N_power,
                                         Complex32 n_inverse, bool last_kernel,
                                         bool NTT_mult, int offset2,
                                         bool reduction_poly_check);

    template __global__ void InverseCore(Float64* output, Complex64* polynomial,
                                         Complex64* inverse_root_of_unity_table,
                                         int shared_index, int logm, int k,
                                         int outer_iteration_count, int N_power,
                                         Complex64 n_inverse, bool last_kernel,
                                         bool NTT_mult, int offset2,
                                         bool reduction_poly_check);

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

    template __global__ void
    Special_ForwardCore(Complex32* polynomial,
                        Complex32* inverse_root_of_unity_table,
                        int shared_index, int logm, int k,
                        int outer_iteration_count, int N_power);

    template __global__ void
    Special_ForwardCore(Complex64* polynomial,
                        Complex64* inverse_root_of_unity_table,
                        int shared_index, int logm, int k,
                        int outer_iteration_count, int N_power);

    template __global__ void
    Special_InverseCore(Complex32* polynomial, Complex32* root_of_unity_table,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, Complex32 n_inverse, bool not_last_kernel);

    template __global__ void
    Special_InverseCore(Complex64* polynomial, Complex64* root_of_unity_table,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, Complex64 n_inverse, bool not_last_kernel);

    template __host__ void GPU_Special_FFT(Complex32* device_inout,
                                           Complex32* root_of_unity_table,
                                           fft_configuration<Float32> cfg,
                                           int batch_size);

    template __host__ void GPU_Special_FFT(Complex64* device_inout,
                                           Complex64* root_of_unity_table,
                                           fft_configuration<Float64> cfg,
                                           int batch_size);

    template __global__ void
    ForwardFFNTCoreUnique(Float32* polynomial_in, Complex32* polynomial_out,
                          Complex32* twist_table,
                          Complex32* root_of_unity_table, int shared_index,
                          int logm, int outer_iteration_count, int N_power);

    template __global__ void
    ForwardFFNTCoreUnique(Float64* polynomial_in, Complex64* polynomial_out,
                          Complex64* twist_table,
                          Complex64* root_of_unity_table, int shared_index,
                          int logm, int outer_iteration_count, int N_power);

    template __global__ void ForwardFFNTCoreRegular(
        Complex32* polynomial, Complex32* root_of_unity_table, int shared_index,
        int logm, int outer_iteration_count, int N_power, bool not_last_kernel);

    template __global__ void ForwardFFNTCoreRegular(
        Complex64* polynomial, Complex64* root_of_unity_table, int shared_index,
        int logm, int outer_iteration_count, int N_power, bool not_last_kernel);

    template __global__ void InverseFFNTCoreRegular(
        Complex32* polynomial, Complex32* inverse_root_of_unity_table,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, bool NTT_mult, int offset2);

    template __global__ void InverseFFNTCoreRegular(
        Complex64* polynomial, Complex64* inverse_root_of_unity_table,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, bool NTT_mult, int offset2);

    template __global__ void InverseFFNTCoreUnique(
        Complex32* polynomial_in, Float32* polynomial_out,
        Complex32* untwist_table, Complex32* inverse_root_of_unity_table,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Complex32 n_inverse);

    template __global__ void InverseFFNTCoreUnique(
        Complex64* polynomial_in, Float64* polynomial_out,
        Complex64* untwist_table, Complex64* inverse_root_of_unity_table,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Complex64 n_inverse);

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