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

    const int test_count = 50;
    const int bestof = 10;
    float time_measurements[test_count];
    for (int loop = 0; loop < test_count; loop++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        // std::mt19937 gen(0);
        unsigned long long minNumber = 0;
        unsigned long long maxNumber = q - 1;
        std::uniform_int_distribution<int> dis(minNumber, maxNumber);

        std::vector<std::vector<COMPLEX_C>> vec_GPU(
            batch, std::vector<COMPLEX_C>(n * 2)); // A

        for (int j = 0; j < batch; j++)
        { // LOAD A
            for (int i = 0; i < n * 2; i++)
            {
                COMPLEX_C element = dis(gen);
                vec_GPU[j][i] = element;
            }
        }

        FFT fft_generator(n);

        /////////////////////////////////////////////////////////////////////////

        COMPLEX* Forward_InOut_Datas;

        FFT_CUDA_CHECK(
            cudaMalloc(&Forward_InOut_Datas, batch * n * 2 * sizeof(COMPLEX)));

        for (int j = 0; j < batch; j++)
        {
            FFT_CUDA_CHECK(
                cudaMemcpy(Forward_InOut_Datas + (n * 2 * j), vec_GPU[j].data(),
                           n * 2 * sizeof(COMPLEX), cudaMemcpyHostToDevice));
        }
        /////////////////////////////////////////////////////////////////////////

        COMPLEX* Root_Table_Device;

        FFT_CUDA_CHECK(cudaMalloc(&Root_Table_Device, n * sizeof(COMPLEX)));

        vector<COMPLEX_C> reverse_table = fft_generator.ReverseRootTable();
        FFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                  n * sizeof(COMPLEX), cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////////////////////

        COMPLEX* Inverse_Root_Table_Device;

        FFT_CUDA_CHECK(
            cudaMalloc(&Inverse_Root_Table_Device, n * sizeof(COMPLEX)));

        vector<COMPLEX_C> inverse_reverse_table =
            fft_generator.InverseReverseRootTable();
        FFT_CUDA_CHECK(cudaMemcpy(Inverse_Root_Table_Device,
                                  inverse_reverse_table.data(),
                                  n * sizeof(COMPLEX), cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////////////////////

        unsigned long long* activity_output;
        FFT_CUDA_CHECK(cudaMalloc(&activity_output,
                                  64 * 512 * sizeof(unsigned long long)));
        GPU_ACTIVITY_HOST(activity_output, 111111);
        FFT_CUDA_CHECK(cudaFree(activity_output));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cudaDeviceSynchronize();

        fft_configuration cfg_fft = {.n_power = (logn + 1),
                                     .ntt_type = FORWARD,
                                     .reduction_poly = X_N_minus,
                                     .zero_padding = false,
                                     .stream = 0};

        float time = 0;
        cudaEvent_t startx, stopx;
        cudaEventCreate(&startx);
        cudaEventCreate(&stopx);

        cudaEventRecord(startx);

        GPU_FFT(Forward_InOut_Datas, Root_Table_Device, cfg_fft, batch, false);

        cudaEventRecord(stopx);
        cudaEventSynchronize(stopx);
        cudaEventElapsedTime(&time, startx, stopx);
        // cout << loop << ": " << time << " milliseconds" << endl;
        // cout << time << ", " ;

        time_measurements[loop] = time;
        FFT_CUDA_CHECK(cudaFree(Forward_InOut_Datas));
        FFT_CUDA_CHECK(cudaFree(Root_Table_Device));
        FFT_CUDA_CHECK(cudaFree(Inverse_Root_Table_Device));
    }

    cout << endl
         << endl
         << "Average: " << calculate_mean(time_measurements, test_count)
         << endl;
    cout << "Best Average: "
         << find_min_average(time_measurements, test_count, bestof) << endl;

    cout << "Standart Deviation: "
         << calculate_standard_deviation(time_measurements, test_count) << endl;

    return EXIT_SUCCESS;
}
