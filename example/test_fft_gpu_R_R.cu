// Copyright 2024 Alişah Özcan
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
using namespace fft;

int q;
int logn;
int batch;
int n;

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
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<vector<unsigned long long>> A_poly(
        batch, vector<unsigned long long>(n * 2));
    vector<vector<unsigned long long>> B_poly(
        batch, vector<unsigned long long>(n * 2));

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

    std::vector<std::vector<FIXED_POINT>> vec_GPU(
        2 * batch, std::vector<FIXED_POINT>(n * 2)); // A and B together

    for (int j = 0; j < batch; j++)
    { // LOAD A
        for (int i = 0; i < n * 2; i++)
        {
            vec_GPU[j][i] = static_cast<FIXED_POINT>(A_poly[j][i]);
        }
    }

    for (int j = 0; j < batch; j++)
    { // LOAD B
        for (int i = 0; i < n * 2; i++)
        {
            vec_GPU[j + batch][i] = static_cast<FIXED_POINT>(B_poly[j][i]);
        }
    }

    FFT fft_generator(n);

    /////////////////////////////////////////////////////////////////////////

    COMPLEX* Temp_Datas;
    FFT_CUDA_CHECK(cudaMalloc(
        &Temp_Datas,
        2 * batch * n * 2 * sizeof(COMPLEX))); // 2 --> A and B, batch -->
                                               // batch size, 2 --> zero pad

    FIXED_POINT* InOut_Datas;
    FFT_CUDA_CHECK(cudaMalloc(
        &InOut_Datas,
        2 * batch * n * 2 * sizeof(FIXED_POINT))); // 2 --> A and B, batch -->
                                                   // batch size, 2 --> zero pad

    for (int j = 0; j < 2 * batch; j++)
    {
        FFT_CUDA_CHECK(cudaMemcpy(InOut_Datas + (n * 2 * j), vec_GPU[j].data(),
                                  n * 2 * sizeof(FIXED_POINT),
                                  cudaMemcpyHostToDevice));
    }
    /////////////////////////////////////////////////////////////////////////

    COMPLEX* Root_Table_Device;

    FFT_CUDA_CHECK(cudaMalloc(&Root_Table_Device, n * sizeof(COMPLEX)));

    vector<COMPLEX_C> reverse_table = fft_generator.ReverseRootTable();
    FFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                              n * sizeof(COMPLEX), cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    COMPLEX* Inverse_Root_Table_Device;

    FFT_CUDA_CHECK(cudaMalloc(&Inverse_Root_Table_Device, n * sizeof(COMPLEX)));

    vector<COMPLEX_C> inverse_reverse_table =
        fft_generator.InverseReverseRootTable();
    FFT_CUDA_CHECK(cudaMemcpy(Inverse_Root_Table_Device,
                              inverse_reverse_table.data(), n * sizeof(COMPLEX),
                              cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    fft_configuration cfg_fft = {.n_power = (logn + 1),
                                 .ntt_type = FORWARD,
                                 .reduction_poly =
                                     ReductionPolynomial::X_N_minus,
                                 .zero_padding = false,
                                 .stream = 0};
    GPU_FFT(InOut_Datas, Temp_Datas, Root_Table_Device, cfg_fft, batch * 2,
            false);

    fft_configuration cfg_ifft = {
        .n_power = (logn + 1),
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = COMPLEX(fft_generator.n_inverse, 0.0),
        .stream = 0};

    GPU_FFT(InOut_Datas, Temp_Datas, Inverse_Root_Table_Device, cfg_ifft, batch,
            true);

    FIXED_POINT test[batch * 2 * n];
    FFT_CUDA_CHECK(cudaMemcpy(test, InOut_Datas,
                              batch * n * 2 * sizeof(FIXED_POINT),
                              cudaMemcpyDeviceToHost));

    for (int j = 0; j < batch; j++)
    {
        vector<unsigned long long> test_school =
            schoolbook_poly_multiplication(A_poly[j], B_poly[j], q, n);
        for (int i = 0; i < n * 2; i++)
        {
            signed gpu_result = std::round(test[(j * (n * 2)) + i]);
            if (test_school[i] != (gpu_result % q))
            {
                throw("ERROR");
            }

            if (i < 10)
            {
                cout << test_school[i] << " - " << gpu_result % q << endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
