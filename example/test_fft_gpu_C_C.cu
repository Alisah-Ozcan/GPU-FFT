// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib> // For atoi or atof functions
#include <iomanip>
#include <iostream>
#include <random>

#include "fft.cuh"
#include "fft_cpu.cuh"

using namespace std;
using namespace gpufft;

int q;
int logn;
int batch;
int n;

// typedef Float32 TestDataType; // Use for 32-bit benchmark
typedef Float64 TestDataType; // Use for 64-bit benchmark

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        q = 7;
        logn = 12;
        batch = 1;
        n = 1 << logn;
    }
    else
    {
        q = 7;
        logn = atoi(argv[1]);
        batch = atoi(argv[2]);
        n = 1 << logn;

        if ((logn < 12) || (24 < logn))
        {
            throw std::runtime_error("LOGN should be in range 12 to 24.");
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    int minNumber = 0;
    int maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<vector<int>> A_poly(batch, vector<int>(n * 2));
    vector<vector<int>> B_poly(batch, vector<int>(n * 2));

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i] = dis(gen);
            B_poly[j][i] = dis(gen);
        }
    }

    // Zero Pad
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i + n] = 0;
            B_poly[j][i + n] = 0;
        }
    }

    std::vector<std::vector<COMPLEX<TestDataType>>> A_vec(
        batch, std::vector<COMPLEX<TestDataType>>(n * 2));
    std::vector<std::vector<COMPLEX<TestDataType>>> B_vec(
        batch, std::vector<COMPLEX<TestDataType>>(n * 2));

    std::vector<std::vector<COMPLEX<TestDataType>>> vec_GPU(
        2 * batch,
        std::vector<COMPLEX<TestDataType>>(n * 2)); // A and B together

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX<TestDataType> A_element = A_poly[j][i];
            A_vec[j][i] = A_element;

            COMPLEX<TestDataType> B_element = B_poly[j][i];
            B_vec[j][i] = B_element;
        }
    }

    for (int j = 0; j < batch; j++)
    { // LOAD A
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX<TestDataType> element = A_poly[j][i];
            vec_GPU[j][i] = element;
        }
    }

    for (int j = 0; j < batch; j++)
    { // LOAD B
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX<TestDataType> element = B_poly[j][i];
            vec_GPU[j + batch][i] = element;
        }
    }

    FFT<TestDataType> fft_generator(n);

    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Forward_InOut_Datas;

    GPUFFT_CUDA_CHECK(cudaMalloc(
        &Forward_InOut_Datas,
        2 * batch * n * 2 *
            sizeof(COMPLEX<TestDataType>))); // 2 --> A and B, batch -->
                                             // batch size, 2 --> zero pad

    for (int j = 0; j < 2 * batch; j++)
    {
        GPUFFT_CUDA_CHECK(cudaMemcpy(
            Forward_InOut_Datas + (n * 2 * j), vec_GPU[j].data(),
            n * 2 * sizeof(COMPLEX<TestDataType>), cudaMemcpyHostToDevice));
    }
    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Root_Table_Device;

    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Root_Table_Device, n * sizeof(COMPLEX<TestDataType>)));

    vector<COMPLEX<TestDataType>> reverse_table =
        fft_generator.ReverseRootTable();
    GPUFFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                 n * sizeof(COMPLEX<TestDataType>),
                                 cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Inverse_Root_Table_Device;

    GPUFFT_CUDA_CHECK(cudaMalloc(&Inverse_Root_Table_Device,
                                 n * sizeof(COMPLEX<TestDataType>)));

    vector<COMPLEX<TestDataType>> inverse_reverse_table =
        fft_generator.InverseReverseRootTable();
    GPUFFT_CUDA_CHECK(
        cudaMemcpy(Inverse_Root_Table_Device, inverse_reverse_table.data(),
                   n * sizeof(COMPLEX<TestDataType>), cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    fft_configuration<TestDataType> cfg_fft{};
    cfg_fft.n_power = (logn + 1);
    cfg_fft.fft_type = FORWARD;
    cfg_fft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_fft.zero_padding = false;
    cfg_fft.stream = 0;

    GPU_FFT(Forward_InOut_Datas, Root_Table_Device, cfg_fft, batch * 2, false);

    fft_configuration<TestDataType> cfg_ifft{};
    cfg_ifft.n_power = (logn + 1);
    cfg_ifft.fft_type = INVERSE;
    cfg_ifft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_ifft.zero_padding = false;
    cfg_ifft.mod_inverse = COMPLEX<TestDataType>(fft_generator.n_inverse, 0.0);
    cfg_ifft.stream = 0;

    GPU_FFT(Forward_InOut_Datas, Inverse_Root_Table_Device, cfg_ifft, batch,
            true);

    COMPLEX<TestDataType> test[batch * 2 * n];
    GPUFFT_CUDA_CHECK(cudaMemcpy(test, Forward_InOut_Datas,
                                 batch * n * 2 * sizeof(COMPLEX<TestDataType>),
                                 cudaMemcpyDeviceToHost));

    for (int j = 0; j < batch; j++)
    {
        vector<int> test_school =
            schoolbook_poly_multiplication_without_reduction(A_poly[j],
                                                             B_poly[j], q, n);
        for (int i = 0; i < n * 2; i++)
        {
            signed gpu_result = std::round(test[(j * (n * 2)) + i].real());
            if (test_school[i] != (gpu_result % q))
            {
                throw runtime_error("ERROR");
            }

            if (i < 10)
            {
                cout << test_school[i] << " - " << gpu_result % q << endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
