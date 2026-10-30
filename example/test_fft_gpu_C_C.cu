// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "gpufft/fft.cuh"
#include "gpufft/fft_cpu.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpufft;
using namespace gpufft_example;

// Multiplies batch pairs of polynomials with n = 2^logn coefficients: zero
// padded to 2n complex values, forward FFT of A and B, inverse FFT with the
// fused pointwise product, checked against the schoolbook product.
template <typename T> bool run(int logn, int batch)
{
    check_logn(logn, 11, 23); // FFT size 2n: n_power 12 to 24
    const int n = 1 << logn;

    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(0, coefficient_bound<T>());

    vector<vector<int>> A_poly(batch, vector<int>(n));
    vector<vector<int>> B_poly(batch, vector<int>(n));
    // A (batch polynomials) followed by B, zero padded
    vector<COMPLEX<T>> vec_GPU(2 * batch * n * 2, COMPLEX<T>(0.0, 0.0));
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i] = dis(gen);
            B_poly[j][i] = dis(gen);
            vec_GPU[(j * n * 2) + i] = COMPLEX<T>(A_poly[j][i], 0.0);
            vec_GPU[((j + batch) * n * 2) + i] = COMPLEX<T>(B_poly[j][i], 0.0);
        }
    }

    FFT<T> fft_generator(n);

    COMPLEX<T>* InOut_Datas;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&InOut_Datas, vec_GPU.size() * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(InOut_Datas, vec_GPU.data(),
                                 vec_GPU.size() * sizeof(COMPLEX<T>),
                                 cudaMemcpyHostToDevice));

    vector<COMPLEX<T>> reverse_table = fft_generator.ReverseRootTable();
    COMPLEX<T>* Root_Table_Device;
    GPUFFT_CUDA_CHECK(cudaMalloc(&Root_Table_Device, n * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                 n * sizeof(COMPLEX<T>),
                                 cudaMemcpyHostToDevice));

    vector<COMPLEX<T>> inverse_reverse_table =
        fft_generator.InverseReverseRootTable();
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

    GPU_FFT(InOut_Datas, Root_Table_Device, cfg_fft, batch * 2, false);

    fft_configuration<T> cfg_ifft{};
    cfg_ifft.n_power = logn + 1;
    cfg_ifft.fft_type = INVERSE;
    cfg_ifft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_ifft.mod_inverse = COMPLEX<T>(fft_generator.n_inverse, 0.0);
    cfg_ifft.stream = 0;

    GPU_FFT(InOut_Datas, Inverse_Root_Table_Device, cfg_ifft, batch, true);

    vector<COMPLEX<T>> result(batch * n * 2);
    GPUFFT_CUDA_CHECK(cudaMemcpy(result.data(), InOut_Datas,
                                 result.size() * sizeof(COMPLEX<T>),
                                 cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int j = 0; j < batch && correct; j++)
    {
        vector<T> real(n * 2);
        for (int i = 0; i < n * 2; i++)
            real[i] = result[(j * n * 2) + i].real();
        correct =
            check_rounded(real, linear_multiplication(A_poly[j], B_poly[j]), j);
    }
    cout << (correct ? "All Correct for GPU FFT (complex) multiplication."
                     : "GPU FFT (complex) multiplication failed.")
         << endl;

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
