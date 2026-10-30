// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib> // For atoi or atof functions
#include <iomanip>
#include <iostream>
#include <random>

#include "gpufft/fft.cuh"
#include "gpufft/fft_cpu.cuh"

using namespace std;
using namespace gpufft;

int q;
int logn;
int batch;
int n;

// typedef Float32 TestDataType; // Use for 32-bit benchmark
typedef Float64 TestDataType; // Use for 64-bit benchmark

// a * b mod X^n + 1, without reduction of the coefficients
vector<long long> negacyclic_multiplication(const vector<int>& a,
                                            const vector<int>& b, int size)
{
    vector<long long> result(size, 0);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            long long mult = static_cast<long long>(a[i]) * b[j];
            if (i + j < size)
            {
                result[i + j] += mult;
            }
            else
            {
                result[i + j - size] -= mult;
            }
        }
    }
    return result;
}

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

    std::cout << "logn: " << logn << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    int minNumber = 0;
    int maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<vector<int>> A_poly(batch, vector<int>(n));
    vector<vector<int>> B_poly(batch, vector<int>(n));

    vector<TestDataType> poly_f(2 * batch * n); // A and B together
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i] = dis(gen);
            poly_f[(j * n) + i] = A_poly[j][i];

            B_poly[j][i] = dis(gen);
            poly_f[((j + batch) * n) + i] = B_poly[j][i];
        }
    }

    FFNT<TestDataType> fft_generator(n);

    COMPLEX<TestDataType>* Temp_Datas;
    GPUFFT_CUDA_CHECK(cudaMalloc(&Temp_Datas, 2 * batch * (n >> 1) *
                                                  sizeof(COMPLEX<TestDataType>)));

    TestDataType* Forward_InOut_Datas;
    GPUFFT_CUDA_CHECK(cudaMalloc(&Forward_InOut_Datas,
                                 2 * batch * n * sizeof(TestDataType)));

    GPUFFT_CUDA_CHECK(cudaMemcpy(Forward_InOut_Datas, poly_f.data(),
                                 2 * batch * n * sizeof(TestDataType),
                                 cudaMemcpyHostToDevice));

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

    std::vector<TestDataType> test(batch * n);
    GPUFFT_CUDA_CHECK(cudaMemcpy(test.data(), Forward_InOut_Datas,
                                 batch * n * sizeof(TestDataType),
                                 cudaMemcpyDeviceToHost));

    for (int j = 0; j < batch; j++)
    {
        vector<long long> test_school =
            negacyclic_multiplication(A_poly[j], B_poly[j], n);

        for (int i = 0; i < n; i++)
        {
            long long gpu_result = std::llround(test[(j * n) + i]);
            if (test_school[i] != gpu_result)
            {
                throw runtime_error("ERROR");
            }

            if (i < 10)
            {
                cout << i << ": " << test_school[i] << " - " << gpu_result
                     << endl;
            }
        }
    }

    GPUFFT_CUDA_CHECK(cudaFree(Temp_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(Forward_InOut_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(Root_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Inverse_Root_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Twist_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Untwist_Table_Device));

    return EXIT_SUCCESS;
}
