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

int q;
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
    int minNumber = 0;
    int maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<vector<int>> A_poly(batch, vector<int>(n * 2));
    vector<vector<int>> B_poly(batch, vector<int>(n * 2));

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i] = dis(gen);
            B_poly[j][i] = dis(gen);
        }
    }

    // Zero Pad
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i + n] = 0;
            B_poly[j][i + n] = 0;
        }
    }

    std::vector<std::vector<COMPLEX<TestDataType>>> A_vec(
        batch, std::vector<COMPLEX<TestDataType>>(n * 2));
    std::vector<std::vector<COMPLEX<TestDataType>>> B_vec(
        batch, std::vector<COMPLEX<TestDataType>>(n * 2));

    std::vector<std::vector<COMPLEX<TestDataType>>> vec_GPU(
        2 * batch,
        std::vector<COMPLEX<TestDataType>>(n * 2)); // A and B together

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX<TestDataType> A_element = A_poly[j][i];
            A_vec[j][i] = A_element;

            COMPLEX<TestDataType> B_element = B_poly[j][i];
            B_vec[j][i] = B_element;
        }
    }

    for (int j = 0; j < batch; j++)
    { // LOAD A
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX<TestDataType> element = A_poly[j][i];
            vec_GPU[j][i] = element;
        }
    }

    for (int j = 0; j < batch; j++)
    { // LOAD B
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX<TestDataType> element = B_poly[j][i];
            vec_GPU[j + batch][i] = element;
        }
    }

    FFT<TestDataType> fft_generator(n);

    for (int j = 0; j < batch; j++)
    {
        fft_generator.fft(A_vec[j]);
        fft_generator.fft(B_vec[j]);
    }

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n * 2; i++)
        {
            A_vec[j][i] = A_vec[j][i] * B_vec[j][i];
        }
    }

    for (int j = 0; j < batch; j++)
    {
        fft_generator.ifft(A_vec[j]);
    }

    for (int j = 0; j < batch; j++)
    {
        vector<int> test_school =
            schoolbook_poly_multiplication_without_reduction(A_poly[j],
                                                             B_poly[j], q, n);
        for (int i = 0; i < n * 2; i++)
        {
            signed cpu_result = std::round(A_vec[j][i].real());

            if (test_school[i] != (cpu_result % q))
            {
                throw("ERROR");
            }

            if (i < 10)
            {
                cout << test_school[i] << " - " << cpu_result % q << endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
