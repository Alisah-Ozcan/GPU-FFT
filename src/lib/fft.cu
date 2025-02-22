// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "fft.cuh"

namespace gpufft
{
    template <typename T>
    __device__ void CooleyTukeyUnit(COMPLEX<T>& U, COMPLEX<T>& V,
                                    COMPLEX<T>& root)
    {
        COMPLEX<T> u_ = U;
        COMPLEX<T> v_ = V * root;

        U = u_ + v_;
        V = u_ - v_;
    }

    template <typename T>
    __device__ void GentlemanSandeUnit(COMPLEX<T>& U, COMPLEX<T>& V,
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
                bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ COMPLEX<T> shared_memory[];
        extern __shared__ char shared_memory_typed[];
        COMPLEX<T>* shared_memory =
            reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load data from global & store to shared
        shared_memory[shared_addresss] = polynomial[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial[global_addresss + offset];

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
        if (not_last_kernel)
        {
#pragma unroll
            for (int lp = 0; lp < loops; lp++)
            {
                //__syncthreads();

                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
            //__syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++)
            {
                //__syncthreads();

                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
            //__syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }

        polynomial[global_addresss] = shared_memory[shared_addresss];
        polynomial[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }

    template <typename T>
    __global__ void
    InverseCore(COMPLEX<T>* polynomial, COMPLEX<T>* inverse_root_of_unity_table,
                int shared_index, int logm, int k, int outer_iteration_count,
                int N_power, COMPLEX<T> n_inverse, bool last_kernel,
                bool NTT_mult, int offset2, bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ COMPLEX<T> shared_memory[];
        extern __shared__ char shared_memory_typed[];
        COMPLEX<T>* shared_memory =
            reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);
        ;
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        if (NTT_mult)
        {
            COMPLEX<T> a0 = polynomial[global_addresss];
            COMPLEX<T> a1 = polynomial[global_addresss + offset];

            COMPLEX<T> b0 = polynomial[global_addresss + offset2];
            COMPLEX<T> b1 = polynomial[global_addresss + offset + offset2];

            shared_memory[shared_addresss] = a0 * b0;
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
                a1 * b1;
        }
        else
        {
            shared_memory[shared_addresss] = polynomial[global_addresss];
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
                polynomial[global_addresss + offset];
        }

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index]);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();

        if (last_kernel)
        {
            polynomial[global_addresss] =
                shared_memory[shared_addresss] * n_inverse;
            polynomial[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)] *
                n_inverse;
        }
        else
        {
            polynomial[global_addresss] = shared_memory[shared_addresss];
            polynomial[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __global__ void
    ForwardCore(T* input, COMPLEX<T>* polynomial,
                COMPLEX<T>* root_of_unity_table, int shared_index, int logm,
                int outer_iteration_count, int N_power, bool zero_padding,
                bool not_last_kernel, bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ COMPLEX<T> shared_memory[];
        extern __shared__ char shared_memory_typed[];
        COMPLEX<T>* shared_memory =
            reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load data from global & store to shared

        shared_memory[shared_addresss] = COMPLEX<T>(input[global_addresss]);
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            COMPLEX<T>(input[global_addresss + offset]);

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
        if (not_last_kernel)
        {
#pragma unroll
            for (int lp = 0; lp < loops; lp++)
            {
                //__syncthreads();

                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
            //__syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++)
            {
                //__syncthreads();

                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
            //__syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }

        polynomial[global_addresss] = shared_memory[shared_addresss];
        polynomial[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }

    template <typename T>
    __global__ void
    InverseCore(T* output, COMPLEX<T>* polynomial,
                COMPLEX<T>* inverse_root_of_unity_table, int shared_index,
                int logm, int k, int outer_iteration_count, int N_power,
                COMPLEX<T> n_inverse, bool last_kernel, bool NTT_mult,
                int offset2, bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ COMPLEX<T> shared_memory[];
        extern __shared__ char shared_memory_typed[];
        COMPLEX<T>* shared_memory =
            reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);
        ;
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        if (NTT_mult)
        {
            COMPLEX<T> a0 = polynomial[global_addresss];
            COMPLEX<T> a1 = polynomial[global_addresss + offset];

            COMPLEX<T> b0 = polynomial[global_addresss + offset2];
            COMPLEX<T> b1 = polynomial[global_addresss + offset + offset2];

            shared_memory[shared_addresss] = a0 * b0;
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
                a1 * b1;
        }
        else
        {
            shared_memory[shared_addresss] = polynomial[global_addresss];
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
                polynomial[global_addresss + offset];
        }

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index]);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();

        if (last_kernel)
        {
            output[global_addresss] =
                (shared_memory[shared_addresss] * n_inverse).real();
            output[global_addresss + offset] =
                (shared_memory[shared_addresss + (blockDim.x * blockDim.y)] *
                 n_inverse)
                    .real();
        }
        else
        {
            output[global_addresss] = shared_memory[shared_addresss].real();
            output[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)]
                    .real();
        }
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

        auto kernel_parameters = (cfg.fft_type == FORWARD)
                                     ? CreateForwardFFTKernel<T>()
                                     : CreateInverseFFTKernel<T>();

        switch (cfg.fft_type)
        {
            case FORWARD:
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
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
                        cfg.n_power, cfg.zero_padding,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }
                break;
            case INVERSE:
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
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

                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
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

        auto kernel_parameters = (cfg.fft_type == FORWARD)
                                     ? CreateForwardFFTKernel<T>()
                                     : CreateInverseFFTKernel<T>();

        switch (cfg.fft_type)
        {
            case FORWARD:
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
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
                    cfg.zero_padding, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUFFT_CUDA_CHECK(cudaGetLastError());

                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
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
                        cfg.n_power, cfg.zero_padding,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                }
            }
            break;
            case INVERSE:
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
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

                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    if (i == (kernel_parameters[cfg.n_power].size() - 1))
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
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
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
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
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ COMPLEX<T> shared_memory[];
        extern __shared__ char shared_memory_typed[];
        COMPLEX<T>* shared_memory =
            reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);

        location_t offset = 1 << (N_power - logm - 1);
        int t_ = 8;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << (N_power - logm - 1);

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load data from global & store to shared
        shared_memory[shared_addresss] = polynomial[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial[global_addresss + offset];

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
        if (not_last_kernel)
        {
#pragma unroll
            for (int lp = 0; lp < loops; lp++)
            {
                //__syncthreads();

                current_root_index = m + (omega_addresss & (m - 1));

                GentlemanSandeUnit(shared_memory[in_shared_address],
                                   shared_memory[in_shared_address + t],
                                   root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_ -= 1;
                m >>= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
            //__syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < 3; lp++)
            {
                //__syncthreads();

                current_root_index = m + (omega_addresss & (m - 1));

                GentlemanSandeUnit(shared_memory[in_shared_address],
                                   shared_memory[in_shared_address + t],
                                   root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_ -= 1;
                m >>= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                __syncthreads();
            }
            //__syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                current_root_index = m + (omega_addresss & (m - 1));

                GentlemanSandeUnit(shared_memory[in_shared_address],
                                   shared_memory[in_shared_address + t],
                                   root_of_unity_table[current_root_index]);

                t = t >> 1;
                t_ -= 1;
                m >>= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }

        if (not_last_kernel)
        {
            polynomial[global_addresss] = shared_memory[shared_addresss];
            polynomial[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
        else
        {
            polynomial[global_addresss] =
                shared_memory[shared_addresss] * n_inverse;
            polynomial[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)] *
                n_inverse;
        }
    }

    template <typename T>
    __global__ void Special_ForwardCore(COMPLEX<T>* polynomial,
                                        COMPLEX<T>* inverse_root_of_unity_table,
                                        int shared_index, int logm, int k,
                                        int outer_iteration_count, int N_power)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ COMPLEX<T> shared_memory[];
        extern __shared__ char shared_memory_typed[];
        COMPLEX<T>* shared_memory =
            reinterpret_cast<COMPLEX<T>*>(shared_memory_typed);

        location_t offset = 1 << (N_power - k - 1);
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << (N_power - logm - 1);

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        shared_memory[shared_addresss] = polynomial[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial[global_addresss + offset];

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;

        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();

            current_root_index = m + (omega_addresss & (m - 1));

            CooleyTukeyUnit(shared_memory[in_shared_address],
                            shared_memory[in_shared_address + t],
                            inverse_root_of_unity_table[current_root_index]);

            t = t << 1;
            t_ += 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();

        polynomial[global_addresss] = shared_memory[shared_addresss];
        polynomial[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
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

        auto kernel_parameters = (cfg.fft_type == FORWARD)
                                     ? CreateForwardSpecialFFTKernel<T>()
                                     : CreateInverseSpecialFFTKernel<T>();

        switch (cfg.fft_type)
        {
            case FORWARD:
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
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
                for (int i = 0; i < kernel_parameters[cfg.n_power].size() - 1;
                     i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
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

                auto& current_kernel_params =
                    kernel_parameters[cfg.n_power]
                                     [kernel_parameters[cfg.n_power].size() -
                                      1];
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

    template __device__ void CooleyTukeyUnit(Complex32& U, Complex32& V,
                                             Complex32& root);
    template __device__ void CooleyTukeyUnit(Complex64& U, Complex64& V,
                                             Complex64& root);

    template __device__ void GentlemanSandeUnit(Complex32& U, Complex32& V,
                                                Complex32& root);
    template __device__ void GentlemanSandeUnit(Complex64& U, Complex64& V,
                                                Complex64& root);

    template __global__ void
    ForwardCore(Complex32* polynomial, Complex32* root_of_unity_table,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool zero_padding, bool not_last_kernel,
                bool reduction_poly_check);

    template __global__ void
    ForwardCore(Complex64* polynomial, Complex64* root_of_unity_table,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool zero_padding, bool not_last_kernel,
                bool reduction_poly_check);

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

    template __global__ void
    ForwardCore(Float32* input, Complex32* polynomial,
                Complex32* root_of_unity_table, int shared_index, int logm,
                int outer_iteration_count, int N_power, bool zero_padding,
                bool not_last_kernel, bool reduction_poly_check);

    template __global__ void
    ForwardCore(Float64* input, Complex64* polynomial,
                Complex64* root_of_unity_table, int shared_index, int logm,
                int outer_iteration_count, int N_power, bool zero_padding,
                bool not_last_kernel, bool reduction_poly_check);

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

} // namespace gpufft