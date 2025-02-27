// Copyright 2023-2025 Alişah Özcan
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

unsigned long long q;
int logn;
int batch;
int n;

// typedef Float32 TestDataType; // Use for 32-bit benchmark
typedef Float64 TestDataType; // Use for 64-bit benchmark

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
    // std::mt19937 gen(0);
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<unsigned long long> A_poly(n);
    vector<unsigned long long> B_poly(n);
    for (int i = 0; i < n; i++)
    {
        A_poly[i] = dis(gen);
        B_poly[i] = dis(gen);
    }

    FFNT<TestDataType> fft_generator(n);

    vector<unsigned long long> result = fft_generator.MULT(A_poly, B_poly);

    vector<unsigned long long> test_school =
        schoolbook_poly_multiplication(A_poly, B_poly, q, n);
    for (int i = 0; i < n; i++)
    {
        if (test_school[i] != (result[i] % q))
        {
            std::cout << i << std::endl;
            throw runtime_error("ERROR");
        }

        if (i < 10)
        {
            cout << i << ": " << test_school[i] << " - " << result[i] % q
                 << endl;
        }
    }

    return EXIT_SUCCESS;
}
