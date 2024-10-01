// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "fft.cuh"

namespace fft
{
    __device__ void CooleyTukeyUnit(COMPLEX& U, COMPLEX& V, COMPLEX& root)
    {
        COMPLEX u_ = U;
        COMPLEX v_ = V * root;

        U = u_ + v_;
        V = u_ - v_;
    }

    __device__ void GentlemanSandeUnit(COMPLEX& U, COMPLEX& V, COMPLEX& root)
    {
        COMPLEX u_ = U;
        COMPLEX v_ = V;

        U = (u_ + v_);
        V = (u_ - v_) * root;
    }

    __global__ void ForwardCore(COMPLEX* polynomial,
                                COMPLEX* root_of_unity_table, int logm,
                                int outer_iteration_count, int N_power,
                                bool zero_padding, bool not_last_kernel,
                                bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        extern __shared__ COMPLEX shared_memory[];

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = 8;
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
            for (int lp = 0; lp < 3; lp++)
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

    __global__ void InverseCore(COMPLEX* polynomial,
                                COMPLEX* inverse_root_of_unity_table, int logm,
                                int k, int outer_iteration_count, int N_power,
                                COMPLEX n_inverse, bool last_kernel,
                                bool NTT_mult, int offset2,
                                bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        extern __shared__ COMPLEX shared_memory[];

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        int t_ = 9 - outer_iteration_count;
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
            COMPLEX a0 = polynomial[global_addresss];
            COMPLEX a1 = polynomial[global_addresss + offset];

            COMPLEX b0 = polynomial[global_addresss + offset2];
            COMPLEX b1 = polynomial[global_addresss + offset + offset2];

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

    __host__ void GPU_FFT(COMPLEX* device_inout, COMPLEX* root_of_unity_table,
                          fft_configuration cfg, int batch_size,
                          bool multiplication)
    {
        switch (cfg.ntt_type)
        {
            case FORWARD:
                switch (cfg.n_power)
                {
                    case 12:
                        ForwardCore<<<dim3(8, 1, batch_size), dim3(64, 4),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 3,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 8, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 3, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 13:
                        ForwardCore<<<dim3(16, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 4,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 16, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 14:
                        ForwardCore<<<dim3(32, 1, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 5,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 32, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 5, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 15:
                        ForwardCore<<<dim3(64, 1, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 6,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 64, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 6, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 16:
                        ForwardCore<<<dim3(128, 1, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 7,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 128, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 7, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 17:
                        ForwardCore<<<dim3(256, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 4,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(16, 16, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 4,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 256, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 8, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 18:
                        ForwardCore<<<dim3(512, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 4,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(32, 16, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 5,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 512, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 9, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 19:
                        ForwardCore<<<dim3(1024, 1, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 5,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(32, 32, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 5, 5,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 1024, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 10, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 20:
                        ForwardCore<<<dim3(2048, 1, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 5,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(64, 32, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 5, 6,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 2048, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 11, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 21:
                        ForwardCore<<<dim3(4096, 1, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 6,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(64, 64, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 6, 6,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 4096, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 12, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 22: //
                        ForwardCore<<<dim3(8192, 1, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 6,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(128, 64, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 6, 7,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 8192, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 13, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());

                        break;
                    case 23:
                        ForwardCore<<<dim3(16384, 1, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 7,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(128, 128, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 7, 7,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 16384, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 14, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 24:
                        ForwardCore<<<dim3(32768, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 0, 4,
                            cfg.n_power, cfg.zero_padding, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(2048, 16, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 4,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(128, 256, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 8, 7,
                            cfg.n_power, false, true,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        ForwardCore<<<dim3(1, 32768, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 15, 9,
                            cfg.n_power, false, false,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;

                    default:
                        break;
                }
                break;
            case INVERSE:
                switch (cfg.n_power)
                {
                    case 12:
                        InverseCore<<<dim3(1, 8, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 11, 3, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(8, 1, batch_size), dim3(64, 4),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 2, 0, 3,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 13:
                        InverseCore<<<dim3(1, 16, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 12, 4, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(16, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 3, 0, 4,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 14:
                        InverseCore<<<dim3(1, 32, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 13, 5, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(32, 1, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 0, 5,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 15:
                        InverseCore<<<dim3(1, 64, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 14, 6, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(64, 1, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 5, 0, 6,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 16:
                        InverseCore<<<dim3(1, 128, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 15, 7, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(128, 1, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 6, 0, 7,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 17:
                        InverseCore<<<dim3(1, 256, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 16, 8, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(16, 16, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 7, 4, 4,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(256, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 3, 0, 4,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 18:
                        InverseCore<<<dim3(1, 512, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 17, 9, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(32, 16, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 8, 4, 5,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(512, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 3, 0, 4,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 19:
                        InverseCore<<<dim3(1, 1024, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 18, 10, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(32, 32, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 9, 5, 5,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(1024, 1, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 0, 5,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 20:
                        InverseCore<<<dim3(1, 2048, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 19, 11, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(64, 32, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 10, 5, 6,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(2048, 1, batch_size), dim3(16, 16),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 4, 0, 5,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 21: //
                        InverseCore<<<dim3(1, 4096, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 20, 12, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(64, 64, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 11, 6, 6,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(4096, 1, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 5, 0, 6,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 22:
                        InverseCore<<<dim3(1, 8192, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 21, 13, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(128, 64, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 12, 6, 7,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(8192, 1, batch_size), dim3(8, 32),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 5, 0, 6,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 23:
                        InverseCore<<<dim3(1, 16384, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 22, 14, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(128, 128, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 13, 7, 7,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(16384, 1, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 6, 0, 7,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        break;
                    case 24:
                        InverseCore<<<dim3(1, 32768, batch_size), dim3(256, 1),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 23, 15, 9,
                            cfg.n_power, cfg.mod_inverse, false, multiplication,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(128, 256, batch_size), dim3(4, 64),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 14, 8, 7,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(2048, 16, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 7, 4, 4,
                            cfg.n_power, cfg.mod_inverse, false, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        FFT_CUDA_CHECK(cudaGetLastError());
                        InverseCore<<<dim3(32768, 1, batch_size), dim3(32, 8),
                                      512 * sizeof(COMPLEX), cfg.stream>>>(
                            device_inout, root_of_unity_table, 3, 0, 4,
                            cfg.n_power, cfg.mod_inverse, true, false,
                            (batch_size << cfg.n_power),
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        break;

                    default:
                        break;
                }
                break;

            default:
                break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __global__ void GPU_ACTIVITY(unsigned long long* output,
                                 unsigned long long fix_num)
    {
        int idx = blockIdx.x + blockDim.x + threadIdx.x;

        output[idx] = fix_num;
    }

    __host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                    unsigned long long fix_num)
    {
        GPU_ACTIVITY<<<64, 512>>>(output, fix_num);
    }

} // namespace fft