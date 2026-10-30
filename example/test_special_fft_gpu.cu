// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cmath>
#include <cstdlib>
#include <random>

#include "gpufft/fft.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpufft;
using namespace gpufft_example;

// Root table of GPU_Special_FFT for l = 2^logl slots (the CKKS canonical
// embedding): entry m / 2 + j of the stage with sub-arrays of m elements is
// Psi[(5^j mod 4m) * (l / m)] with Psi[k] = exp(pi i k / (2l)). The inverse
// table holds the conjugates.
template <typename T>
vector<COMPLEX<T>> special_fft_table(int logl, bool inverse)
{
    const int l = 1 << logl;
    vector<COMPLEX<T>> table(l, COMPLEX<T>(0.0, 0.0));
    for (int logm = 1; logm <= logl; logm++)
    {
        const int m = 1 << logm;
        long long rotation = 1; // 5^j mod 4l
        for (int j = 0; j < m / 2; j++)
        {
            const long long k = (rotation % (4 * m)) * (l / m);
            const double angle = M_PI * k / (2.0 * l);
            table[m / 2 + j] = COMPLEX<T>(
                std::cos(angle), inverse ? -std::sin(angle) : std::sin(angle));
            rotation = (5 * rotation) % (4 * l);
        }
    }
    return table;
}

// The FFT-like algorithm of the GPU_Special_FFT comment (without the bit
// reversal of the input), in double precision.
vector<COMPLEX<double>> special_fft_cpu(vector<COMPLEX<double>> w, int logl)
{
    const int l = 1 << logl;
    vector<COMPLEX<double>> table = special_fft_table<double>(logl, false);
    for (int m = 2; m <= l; m <<= 1)
    {
        for (int i = 0; i < l; i += m)
        {
            for (int j = 0; j < m / 2; j++)
            {
                COMPLEX<double> u = w[i + j];
                COMPLEX<double> v = w[i + j + m / 2] * table[m / 2 + j];
                w[i + j] = u + v;
                w[i + j + m / 2] = u - v;
            }
        }
    }
    return w;
}

// Largest error relative to the largest magnitude of the expected values.
template <typename T>
double relative_error(const vector<COMPLEX<T>>& got,
                      const vector<COMPLEX<double>>& expected)
{
    double error = 0.0, scale = 0.0;
    for (size_t i = 0; i < expected.size(); i++)
    {
        error = std::max(error, std::hypot(got[i].real() - expected[i].real(),
                                           got[i].imag() - expected[i].imag()));
        scale =
            std::max(scale, std::hypot(expected[i].real(), expected[i].imag()));
    }
    return error / scale;
}

// Forward special FFT of batch vectors of l = 2^logl slots, checked against
// the CPU algorithm, followed by the inverse, checked against the input.
template <typename T> bool run(int logl, int batch)
{
    check_logn(logl, 11, 24);
    const int l = 1 << logl;
    const double tolerance = sizeof(T) == 4 ? 1e-5 : 1e-12;

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    vector<COMPLEX<T>> slots(batch * l);
    for (auto& s : slots)
    {
        T re = static_cast<T>(dis(gen));
        s = COMPLEX<T>(re, static_cast<T>(dis(gen)));
    }

    COMPLEX<T>* InOut_Datas;
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&InOut_Datas, slots.size() * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(InOut_Datas, slots.data(),
                                 slots.size() * sizeof(COMPLEX<T>),
                                 cudaMemcpyHostToDevice));

    vector<COMPLEX<T>> table = special_fft_table<T>(logl, false);
    vector<COMPLEX<T>> inverse_table = special_fft_table<T>(logl, true);
    COMPLEX<T>* Table_Device;
    COMPLEX<T>* Inverse_Table_Device;
    GPUFFT_CUDA_CHECK(cudaMalloc(&Table_Device, l * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(
        cudaMalloc(&Inverse_Table_Device, l * sizeof(COMPLEX<T>)));
    GPUFFT_CUDA_CHECK(cudaMemcpy(Table_Device, table.data(),
                                 l * sizeof(COMPLEX<T>),
                                 cudaMemcpyHostToDevice));
    GPUFFT_CUDA_CHECK(cudaMemcpy(Inverse_Table_Device, inverse_table.data(),
                                 l * sizeof(COMPLEX<T>),
                                 cudaMemcpyHostToDevice));

    fft_configuration<T> cfg_fft{};
    cfg_fft.n_power = logl;
    cfg_fft.fft_type = FORWARD;
    cfg_fft.stream = 0;
    GPU_Special_FFT(InOut_Datas, Table_Device, cfg_fft, batch);

    vector<COMPLEX<T>> forward(slots.size());
    GPUFFT_CUDA_CHECK(cudaMemcpy(forward.data(), InOut_Datas,
                                 forward.size() * sizeof(COMPLEX<T>),
                                 cudaMemcpyDeviceToHost));

    fft_configuration<T> cfg_ifft{};
    cfg_ifft.n_power = logl;
    cfg_ifft.fft_type = INVERSE;
    cfg_ifft.mod_inverse = COMPLEX<T>(1.0 / l, 0.0);
    cfg_ifft.stream = 0;
    GPU_Special_FFT(InOut_Datas, Inverse_Table_Device, cfg_ifft, batch);

    vector<COMPLEX<T>> back(slots.size());
    GPUFFT_CUDA_CHECK(cudaMemcpy(back.data(), InOut_Datas,
                                 back.size() * sizeof(COMPLEX<T>),
                                 cudaMemcpyDeviceToHost));

    double forward_error = 0.0, round_trip_error = 0.0;
    for (int j = 0; j < batch; j++)
    {
        vector<COMPLEX<double>> input(l);
        for (int i = 0; i < l; i++)
            input[i] = COMPLEX<double>(slots[j * l + i].real(),
                                       slots[j * l + i].imag());
        vector<COMPLEX<T>> got_forward(forward.begin() + j * l,
                                       forward.begin() + (j + 1) * l);
        vector<COMPLEX<T>> got_back(back.begin() + j * l,
                                    back.begin() + (j + 1) * l);
        forward_error =
            std::max(forward_error,
                     relative_error(got_forward, special_fft_cpu(input, logl)));
        round_trip_error =
            std::max(round_trip_error, relative_error(got_back, input));
    }

    const bool correct =
        forward_error < tolerance && round_trip_error < tolerance;
    cout << "Forward special FFT error vs CPU: " << forward_error << endl;
    cout << "Inverse after forward error: " << round_trip_error << endl;
    cout << (correct ? "All Correct for GPU special FFT."
                     : "GPU special FFT failed.")
         << endl;

    GPUFFT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUFFT_CUDA_CHECK(cudaFree(Table_Device));
    GPUFFT_CUDA_CHECK(cudaFree(Inverse_Table_Device));
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
