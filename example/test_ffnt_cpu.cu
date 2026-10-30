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

// Multiplies batch pairs of polynomials with n = 2^logn coefficients modulo
// X^n + 1 with the CPU FFNT, checked against the schoolbook product.
template <typename T> bool run(int logn, int batch)
{
    check_logn(logn, 2, 24);
    const int n = 1 << logn;

    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(0, coefficient_bound<T>());

    FFNT<T> fft_generator(n);

    bool correct = true;
    for (int j = 0; j < batch && correct; j++)
    {
        vector<int> A_poly(n), B_poly(n);
        vector<unsigned long long> A_input(n), B_input(n);
        for (int i = 0; i < n; i++)
        {
            A_poly[i] = dis(gen);
            B_poly[i] = dis(gen);
            A_input[i] = A_poly[i];
            B_input[i] = B_poly[i];
        }

        // MULT returns negative coefficients modulo 2^64.
        vector<unsigned long long> result =
            fft_generator.MULT(A_input, B_input);
        vector<long long> signed_result(result.begin(), result.end());
        correct = check_rounded(signed_result,
                                negacyclic_multiplication(A_poly, B_poly), j);
    }
    cout << (correct ? "All Correct for CPU FFNT multiplication."
                     : "CPU FFNT multiplication failed.")
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
