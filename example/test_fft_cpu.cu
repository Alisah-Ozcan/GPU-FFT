// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "gpufft/fft_cpu.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpufft;
using namespace gpufft_example;

// Multiplies batch pairs of polynomials with n = 2^logn coefficients with the
// CPU FFT (2n points, zero padded), checked against the schoolbook product.
template <typename T> bool run(int logn, int batch)
{
    check_logn(logn, 1, 23);
    const int n = 1 << logn;

    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(0, coefficient_bound<T>());

    FFT<T> fft_generator(n);

    bool correct = true;
    for (int j = 0; j < batch && correct; j++)
    {
        vector<int> A_poly(n), B_poly(n);
        vector<COMPLEX<T>> A_vec(n * 2, COMPLEX<T>(0.0, 0.0));
        vector<COMPLEX<T>> B_vec(n * 2, COMPLEX<T>(0.0, 0.0));
        for (int i = 0; i < n; i++)
        {
            A_poly[i] = dis(gen);
            B_poly[i] = dis(gen);
            A_vec[i] = COMPLEX<T>(A_poly[i], 0.0);
            B_vec[i] = COMPLEX<T>(B_poly[i], 0.0);
        }

        fft_generator.fft(A_vec);
        fft_generator.fft(B_vec);
        for (int i = 0; i < n * 2; i++)
        {
            A_vec[i] = A_vec[i] * B_vec[i];
        }
        fft_generator.ifft(A_vec);

        vector<T> real(n * 2);
        for (int i = 0; i < n * 2; i++)
            real[i] = A_vec[i].real();
        correct = check_rounded(real, linear_multiplication(A_poly, B_poly), j);
    }
    cout << (correct ? "All Correct for CPU FFT multiplication."
                     : "CPU FFT multiplication failed.")
         << endl;
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
