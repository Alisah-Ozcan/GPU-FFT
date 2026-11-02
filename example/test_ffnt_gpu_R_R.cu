// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "gpufft/fft.cuh"
#include "gpufft/fft_cpu.cuh"
#include "gpufft/twiddles.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpufft;
using namespace gpufft_example;

// Multiplies batch pairs of real polynomials with n = 2^logn coefficients
// modulo X^n + 1 with the FFNT (an FFT of n / 2 complex values), checked
// against the schoolbook product.
template <typename T> bool run(int logn, int batch)
{
    check_logn(logn, 12, 28);
    const int n = 1 << logn;

    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(0, coefficient_bound<T>());

    vector<vector<int>> A_poly(batch, vector<int>(n));
    vector<vector<int>> B_poly(batch, vector<int>(n));
    vector<T> poly_f(2 * batch * n); // A (batch polynomials) followed by B
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

    COMPLEX<T>* Temp_Datas;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Temp_Datas, 2 * batch * (n >> 1) * sizeof(COMPLEX<T>)));

    T* InOut_Datas;
    GPUFFT_CUDA_CHECK(cudaMalloc(&InOut_Datas, poly_f.size() * sizeof(T)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(InOut_Datas, poly_f.data(),
                                 poly_f.size() * sizeof(T),
                                 cudaMemcpyHostToDevice));

    // Copies a table to the GPU.
    auto upload_table = [&](const vector<COMPLEX<T>>& table)
    {
        COMPLEX<T>* device;
        GPUFFT_CUDA_CHECK(
            cudaMalloc(&device, table.size() * sizeof(COMPLEX<T>)));
        GPUFFT_CUDA_CHECK(cudaMemcpy(device, table.data(),
                                     table.size() * sizeof(COMPLEX<T>),
                                     cudaMemcpyHostToDevice));
        return device;
    };
    // Root tables of n / 4 entries, twist tables of n / 2 entries.
    COMPLEX<T>* Root_Table_Device =
        upload_table(make_ffnt_twiddles_host<T>(logn, FORWARD));
    COMPLEX<T>* Inverse_Root_Table_Device =
        upload_table(make_ffnt_twiddles_host<T>(logn, INVERSE));
    COMPLEX<T>* Twist_Table_Device =
        upload_table(make_ffnt_twist_host<T>(logn, FORWARD));
    COMPLEX<T>* Untwist_Table_Device =
        upload_table(make_ffnt_twist_host<T>(logn, INVERSE));

    fft_configuration<T> cfg_fft{};
    cfg_fft.n_power = logn;
    cfg_fft.fft_type = FORWARD;
    cfg_fft.stream = 0;

    GPU_FFNT(InOut_Datas, Temp_Datas, Twist_Table_Device, Root_Table_Device,
             cfg_fft, batch * 2, false);

    fft_configuration<T> cfg_ifft{};
    cfg_ifft.n_power = logn;
    cfg_ifft.fft_type = INVERSE;
    cfg_ifft.mod_inverse = COMPLEX<T>(1.0 / (n >> 1), 0.0);
    cfg_ifft.stream = 0;

    GPU_FFNT(InOut_Datas, Temp_Datas, Untwist_Table_Device,
             Inverse_Root_Table_Device, cfg_ifft, batch, true);

    vector<T> result(batch * n);
    GPUFFT_CUDA_CHECK(cudaMemcpy(result.data(), InOut_Datas,
                                 result.size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int j = 0; j < batch && correct; j++)
    {
        vector<T> poly(result.begin() + (j * n),
                       result.begin() + ((j + 1) * n));
        correct = check_rounded(
            poly, negacyclic_multiplication(A_poly[j], B_poly[j]), j);
    }
    cout << (correct ? "All Correct for GPU FFNT multiplication."
                     : "GPU FFNT multiplication failed.")
         << endl;

    GPUFFT_CUDA_CHECK(cudaFree(Temp_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(Root_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Inverse_Root_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Twist_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Untwist_Table_Device));
    return correct;
}

int main(int argc, char* argv[])
{
    const ExampleArgs args = parse_example_args(argc, argv);
    const bool correct = (args.bits == 32)
                             ? run<Float32>(args.logn, args.batch)
                             : run<Float64>(args.logn, args.batch);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
