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
using namespace gpufft;

int q;
int logn;
int batch;
int n;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        q = 7;
        logn = 11;
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

        std::vector<std::vector<Complex64>> vec_GPU(
            2 * batch, std::vector<Complex64>(n * 2)); // A and B together

        for (int j = 0; j < 2 * batch; j++)
        { // LOAD A
            for (int i = 0; i < n * 2; i++)
            {
                Complex64 element = dis(gen);
                vec_GPU[j][i] = element;
            }
        }

        FFT<Float64> fft_generator(n);

        /////////////////////////////////////////////////////////////////////////

        Complex64* Forward_InOut_Datas;

        GPUFFT_CUDA_CHECK(
            cudaMalloc(&Forward_InOut_Datas,
                       2 * batch * n * 2 *
                           sizeof(Complex64))); // 2 --> A and B, batch -->
                                                // batch size, 2 --> zero pad

        for (int j = 0; j < 2 * batch; j++)
        {
            GPUFFT_CUDA_CHECK(
                cudaMemcpy(Forward_InOut_Datas + (n * 2 * j), vec_GPU[j].data(),
                           n * 2 * sizeof(Complex64), cudaMemcpyHostToDevice));
        }
        /////////////////////////////////////////////////////////////////////////

        Complex64* Root_Table_Device;

        GPUFFT_CUDA_CHECK(
            cudaMalloc(&Root_Table_Device, n * sizeof(Complex64)));

        vector<Complex64> reverse_table = fft_generator.ReverseRootTable();
        GPUFFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                     n * sizeof(Complex64),
                                     cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////////////////////

        Complex64* Inverse_Root_Table_Device;

        GPUFFT_CUDA_CHECK(
            cudaMalloc(&Inverse_Root_Table_Device, n * sizeof(Complex64)));

        vector<Complex64> inverse_reverse_table =
            fft_generator.InverseReverseRootTable();
        GPUFFT_CUDA_CHECK(
            cudaMemcpy(Inverse_Root_Table_Device, inverse_reverse_table.data(),
                       n * sizeof(Complex64), cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////////////////////

        unsigned long long* activity_output;
        GPUFFT_CUDA_CHECK(cudaMalloc(&activity_output,
                                     64 * 512 * sizeof(unsigned long long)));
        GPU_ACTIVITY_HOST(activity_output, 111111);
        GPUFFT_CUDA_CHECK(cudaFree(activity_output));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cudaDeviceSynchronize();

        fft_configuration<Float64> cfg_fft = {
            .n_power = (logn + 1),
            .fft_type = FORWARD,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};
        fft_configuration<Float64> cfg_ifft = {
            .n_power = (logn + 1),
            .fft_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = Complex64(fft_generator.n_inverse, 0.0),
            .stream = 0};

        float time = 0;
        cudaEvent_t startx, stopx;
        cudaEventCreate(&startx);
        cudaEventCreate(&stopx);

        cudaEventRecord(startx);

        GPU_FFT(Forward_InOut_Datas, Root_Table_Device, cfg_fft, batch * 2,
                false);

        GPU_FFT(Forward_InOut_Datas, Inverse_Root_Table_Device, cfg_ifft, batch,
                true);

        cudaEventRecord(stopx);
        cudaEventSynchronize(stopx);
        cudaEventElapsedTime(&time, startx, stopx);
        // cout << loop << ": " << time << " milliseconds" << endl;
        // cout << time << ", " ;

        time_measurements[loop] = time;
        GPUFFT_CUDA_CHECK(cudaFree(Forward_InOut_Datas));
        GPUFFT_CUDA_CHECK(cudaFree(Root_Table_Device));
        GPUFFT_CUDA_CHECK(cudaFree(Inverse_Root_Table_Device));
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
