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

// Same product as gpu_fft_C_C_example, with real input and real output: the
// forward FFT reads the real coefficients, the inverse FFT writes the real
// part of the result.
template <typename T> bool run(int logn, int batch)
{
    check_logn(logn, 1, 23); // FFT size 2n: n_power 2 to 24
    const int n = 1 << logn;

    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(0, coefficient_bound<T>());

    vector<vector<int>> A_poly(batch, vector<int>(n));
    vector<vector<int>> B_poly(batch, vector<int>(n));
    // A (batch polynomials) followed by B, zero padded
    vector<T> vec_GPU(2 * batch * n * 2, T(0));
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i] = dis(gen);
            B_poly[j][i] = dis(gen);
            vec_GPU[(j * n * 2) + i] = static_cast<T>(A_poly[j][i]);
            vec_GPU[((j + batch) * n * 2) + i] = static_cast<T>(B_poly[j][i]);
        }
    }

    COMPLEX<T>* Temp_Datas;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Temp_Datas, vec_GPU.size() * sizeof(COMPLEX<T>)));

    T* InOut_Datas;
    GPUFFT_CUDA_CHECK(cudaMalloc(&InOut_Datas, vec_GPU.size() * sizeof(T)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(InOut_Datas, vec_GPU.data(),
                                 vec_GPU.size() * sizeof(T),
                                 cudaMemcpyHostToDevice));

    // Root tables of an FFT of 2n values (X^N - 1): n entries each.
    vector<COMPLEX<T>> reverse_table =
        make_fft_twiddles_host<T>(logn + 1, X_N_minus, FORWARD);
    COMPLEX<T>* Root_Table_Device;
    GPUFFT_CUDA_CHECK(cudaMalloc(&Root_Table_Device, n * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                 n * sizeof(COMPLEX<T>),
                                 cudaMemcpyHostToDevice));

    vector<COMPLEX<T>> inverse_reverse_table =
        make_fft_twiddles_host<T>(logn + 1, X_N_minus, INVERSE);
    COMPLEX<T>* Inverse_Root_Table_Device;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Inverse_Root_Table_Device, n * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(
        cudaMemcpy(Inverse_Root_Table_Device, inverse_reverse_table.data(),
                   n * sizeof(COMPLEX<T>), cudaMemcpyHostToDevice));

    fft_configuration<T> cfg_fft{};
    cfg_fft.n_power = logn + 1;
    cfg_fft.fft_type = FORWARD;
    cfg_fft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_fft.stream = 0;

    GPU_FFT(InOut_Datas, Temp_Datas, Root_Table_Device, cfg_fft, batch * 2,
            false);

    fft_configuration<T> cfg_ifft{};
    cfg_ifft.n_power = logn + 1;
    cfg_ifft.fft_type = INVERSE;
    cfg_ifft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_ifft.mod_inverse = COMPLEX<T>(1.0 / (2 * n), 0.0);
    cfg_ifft.stream = 0;

    GPU_FFT(InOut_Datas, Temp_Datas, Inverse_Root_Table_Device, cfg_ifft, batch,
            true);

    vector<T> result(batch * n * 2);
    GPUFFT_CUDA_CHECK(cudaMemcpy(result.data(), InOut_Datas,
                                 result.size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int j = 0; j < batch && correct; j++)
    {
        vector<T> poly(result.begin() + (j * n * 2),
                       result.begin() + ((j + 1) * n * 2));
        correct =
            check_rounded(poly, linear_multiplication(A_poly[j], B_poly[j]), j);
    }
    cout << (correct ? "All Correct for GPU FFT (real) multiplication."
                     : "GPU FFT (real) multiplication failed.")
         << endl;

    GPUFFT_CUDA_CHECK(cudaFree(Temp_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(Root_Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Inverse_Root_Table_Device));
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
