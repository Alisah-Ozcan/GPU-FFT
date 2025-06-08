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

unsigned long long q;
int logn;
int batch;
int n;

// typedef Float32 TestDataType; // Use for 32-bit benchmark
typedef Float64 TestDataType; // Use for 64-bit benchmark

int main(int argc, char* argv[])
{
    if (argc < 2)
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
        batch = 1;
        n = 1 << logn;

        if ((logn < 12) || (24 < logn))
        {
            throw std::runtime_error("LOGN should be in range 12 to 24.");
        }
    }

    std::cout << "logn: " << logn << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<unsigned long long> A_poly(n);
    vector<unsigned long long> B_poly(n);

    vector<TestDataType> A_poly_f(n);
    vector<TestDataType> B_poly_f(n);
    for (int i = 0; i < n; i++)
    {
        A_poly[i] = dis(gen);
        A_poly_f[i] = A_poly[i];

        B_poly[i] = dis(gen);
        B_poly_f[i] = B_poly[i];
    }

    FFNT<TestDataType> fft_generator(n);

    COMPLEX<TestDataType>* Temp_Datas;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Temp_Datas, 2 * (n >> 1) * sizeof(COMPLEX<TestDataType>)));

    TestDataType* Forward_InOut_Datas;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Forward_InOut_Datas, 2 * n * sizeof(TestDataType)));

    cudaMemcpy(Forward_InOut_Datas, A_poly_f.data(), n * sizeof(TestDataType),
               cudaMemcpyHostToDevice);

    cudaMemcpy(Forward_InOut_Datas + n, B_poly_f.data(),
               n * sizeof(TestDataType), cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Root_Table_Device;

    GPUFFT_CUDA_CHECK(cudaMalloc(&Root_Table_Device,
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>)));

    vector<COMPLEX<TestDataType>> reverse_table =
        fft_generator.ReverseRootTable_ffnt();
    GPUFFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>),
                                 cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Inverse_Root_Table_Device;

    GPUFFT_CUDA_CHECK(cudaMalloc(&Inverse_Root_Table_Device,
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>)));

    vector<COMPLEX<TestDataType>> inverse_reverse_table =
        fft_generator.InverseReverseRootTable_ffnt();
    GPUFFT_CUDA_CHECK(cudaMemcpy(
        Inverse_Root_Table_Device, inverse_reverse_table.data(),
        (n >> 1) * sizeof(COMPLEX<TestDataType>), cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Twist_Table_Device;

    GPUFFT_CUDA_CHECK(cudaMalloc(&Twist_Table_Device,
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>)));

    vector<COMPLEX<TestDataType>> twist_table =
        fft_generator.twist_table_ffnt();
    GPUFFT_CUDA_CHECK(cudaMemcpy(Twist_Table_Device, twist_table.data(),
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>),
                                 cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    COMPLEX<TestDataType>* Untwist_Table_Device;

    GPUFFT_CUDA_CHECK(cudaMalloc(&Untwist_Table_Device,
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>)));

    vector<COMPLEX<TestDataType>> untwist_table =
        fft_generator.untwist_table_ffnt();
    GPUFFT_CUDA_CHECK(cudaMemcpy(Untwist_Table_Device, untwist_table.data(),
                                 (n >> 1) * sizeof(COMPLEX<TestDataType>),
                                 cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    fft_configuration<TestDataType> cfg_fft{};
    cfg_fft.n_power = logn;
    cfg_fft.fft_type = FORWARD;
    cfg_fft.zero_padding = false;
    cfg_fft.stream = 0;

    GPU_FFNT(Forward_InOut_Datas, Temp_Datas, Twist_Table_Device,
             Root_Table_Device, cfg_fft, batch * 2, false);

    TestDataType n_inverse_new = 1.0 / (n >> 1);
    fft_configuration<TestDataType> cfg_ifft{};
    cfg_ifft.n_power = logn;
    cfg_ifft.fft_type = INVERSE;
    cfg_ifft.zero_padding = false;
    cfg_ifft.mod_inverse = COMPLEX<TestDataType>(n_inverse_new, 0.0);
    cfg_ifft.stream = 0;

    GPU_FFNT(Forward_InOut_Datas, Temp_Datas, Untwist_Table_Device,
             Inverse_Root_Table_Device, cfg_ifft, batch, true);

    std::vector<TestDataType> test(n);
    GPUFFT_CUDA_CHECK(cudaMemcpy(test.data(), Forward_InOut_Datas,
                                 n * sizeof(TestDataType),
                                 cudaMemcpyDeviceToHost));

    vector<unsigned long long> test_school =
        schoolbook_poly_multiplication(A_poly, B_poly, q, n);

    for (int i = 0; i < n; i++)
    {
        // signed gpu_result = std::round(test[i]) ;
        unsigned long long gpu_result = std::round(test[i]);
        gpu_result = gpu_result % q;
        if (test_school[i] != (gpu_result))
        {
            throw runtime_error("ERROR");
        }

        if (i < 10)
        {
            cout << i << ": " << test_school[i] << " - " << gpu_result << endl;
        }
    }

    return EXIT_SUCCESS;
}
