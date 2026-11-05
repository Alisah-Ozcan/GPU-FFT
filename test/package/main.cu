// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Runs a forward and an inverse FFT on the GPU with an installed GPU-FFT and
// checks that the input comes back. Needs a GPU to run; building and linking
// it already checks the installed package.

#include <gpufft/fft.cuh>
#include <gpufft/twiddles.cuh>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace gpufft;

int main()
{
    constexpr int n_power = 10;
    constexpr int n = 1 << n_power;

    std::vector<COMPLEX<Float64>> input(n);
    for (int i = 0; i < n; i++)
        input[i] = COMPLEX<Float64>(i % 17, 0.0);

    COMPLEX<Float64>* data;
    GPUFFT_CUDA_CHECK(cudaMalloc(&data, n * sizeof(COMPLEX<Float64>)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(data, input.data(),
                                 n * sizeof(COMPLEX<Float64>),
                                 cudaMemcpyHostToDevice));

    // Root tables of an FFT of n values (X^N - 1): n / 2 entries each.
    COMPLEX<Float64>* tables[2];
    const type directions[2] = {FORWARD, INVERSE};
    for (int i = 0; i < 2; i++)
    {
        std::vector<COMPLEX<Float64>> table =
            make_fft_twiddles_host<Float64>(n_power, X_N_minus, directions[i]);
        GPUFFT_CUDA_CHECK(
            cudaMalloc(&tables[i], (n / 2) * sizeof(COMPLEX<Float64>)));
        GPUFFT_CUDA_CHECK(cudaMemcpy(tables[i], table.data(),
                                     (n / 2) * sizeof(COMPLEX<Float64>),
                                     cudaMemcpyHostToDevice));
    }

    fft_configuration<Float64> cfg{};
    cfg.n_power = n_power;
    cfg.fft_type = FORWARD;
    cfg.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg.stream = 0;
    GPU_FFT(data, tables[0], cfg, 1, false);

    cfg.fft_type = INVERSE;
    cfg.mod_inverse = COMPLEX<Float64>(1.0 / n, 0.0);
    GPU_FFT(data, tables[1], cfg, 1, false);

    std::vector<COMPLEX<Float64>> result(n);
    GPUFFT_CUDA_CHECK(cudaMemcpy(result.data(), data,
                                 n * sizeof(COMPLEX<Float64>),
                                 cudaMemcpyDeviceToHost));
    GPUFFT_CUDA_CHECK(cudaFree(data));
    GPUFFT_CUDA_CHECK(cudaFree(tables[0]));
    GPUFFT_CUDA_CHECK(cudaFree(tables[1]));

    bool correct = true;
    for (int i = 0; i < n; i++)
        correct = correct && std::fabs(result[i].real() - input[i].real()) < 1e-6 &&
                  std::fabs(result[i].imag()) < 1e-6;
    std::cout << (correct ? "GPU FFT and inverse FFT return the input."
                          : "GPU FFT and inverse FFT do not return the input.")
              << std::endl;
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
