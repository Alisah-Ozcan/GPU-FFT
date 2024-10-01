// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "fft_cpu.cuh"

namespace fft
{

    int bitreverse(int index, int n_power)
    {
        int res_1 = 0;
        for (int i = 0; i < n_power; i++)
        {
            res_1 <<= 1;
            res_1 = (index & 1) | res_1;
            index >>= 1;
        }
        return res_1;
    }

    std::vector<unsigned long long>
    schoolbook_poly_multiplication(std::vector<unsigned long long> a,
                                   std::vector<unsigned long long> b,
                                   unsigned long long modulus, int size)
    {
        std::vector<unsigned long long> mult_vector(size * 2, 0);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                unsigned long long mult = a[i] * b[j];
                mult_vector[i + j] += mult;
            }
        }

        for (int i = 0; i < 2 * size; i++)
        {
            mult_vector[i] = mult_vector[i] % modulus;
        }

        return mult_vector;
    }

    std::vector<unsigned long long>
    schoolbook_poly_multiplication_without_reduction(
        std::vector<unsigned long long> a, std::vector<unsigned long long> b,
        unsigned long long modulus, int size)
    {
        std::vector<unsigned long long> mult_vector(size * 2, 0);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                unsigned long long mult = a[i] * b[j];
                mult_vector[i + j] += mult;
            }
        }

        for (int i = 0; i < 2 * size; i++)
        {
            mult_vector[i] = mult_vector[i] % modulus;
        }

        return mult_vector;
    }

    int FFT::n;
    int FFT::logn;
    COMPLEX_C FFT::x;
    int FFT::max_size;
    COMPLEX_C FFT::root; // it was float
    std::vector<COMPLEX_C> FFT::root_tables;
    std::vector<COMPLEX_C> FFT::inverse_root_tables;

    float FFT::n_inverse;

    FFT::FFT(int size)
    {
        n = size;
        logn = int(log2(n));
        COMPLEX_C x_(1.0, 0.0);
        x = x_;
        max_size = n * 2;
        root = static_cast<COMPLEX_C>(2.0) * static_cast<COMPLEX_C>(M_PI) /
               static_cast<COMPLEX_C>(max_size); // it was float
        n_inverse = 1.0 / max_size;

        GenerateRootTable();
        GenerateInverseRootTable();
    }

    void FFT::GenerateRootTable()
    {
        COMPLEX_C j(0.0, 1.0); // Define the complex unit (imaginary part)

        for (int i = 0; i < n; i++)
        {
            COMPLEX_C element =
                std::exp(j * static_cast<COMPLEX_C>(i) * root); // it was float
            root_tables.push_back(element);
        }
    }

    void FFT::GenerateInverseRootTable()
    {
        COMPLEX_C one(1.0); // Define the complex unit (imaginary part)

        for (int i = 0; i < n; i++)
        {
            COMPLEX_C element = one / root_tables[i];
            inverse_root_tables.push_back(element);
        }
    }

    void FFT::fft(std::vector<COMPLEX_C>& input)
    {
        int t = max_size;
        int m = 1;

        while (m < max_size)
        {
            t = t >> 1;

            for (int i = 0; i < m; i++)
            {
                int j1 = 2 * i * t;
                int j2 = j1 + t - 1;

                int index = bitreverse(i, logn);

                COMPLEX_C S = root_tables[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX_C U = input[j];
                    COMPLEX_C V = input[j + t];

                    input[j] = U + (V * S);
                    input[j + t] = U - (V * S);
                }
            }

            m = m << 1;
        }
    }

    void FFT::ifft(std::vector<COMPLEX_C>& input)
    {
        int t = 1;
        int m = max_size;
        while (m > 1)
        {
            int j1 = 0;
            int h = m >> 1;
            for (int i = 0; i < h; i++)
            {
                int j2 = j1 + t - 1;
                int index = bitreverse(i, logn);

                COMPLEX_C S = inverse_root_tables[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX_C U = input[j];
                    COMPLEX_C V = input[j + t];

                    input[j] = (U + V);
                    input[j + t] = (U - V) * S;
                }

                j1 = j1 + (t << 1);
            }

            t = t << 1;
            m = m >> 1;
        }

        for (int i = 0; i < max_size; i++)
        {
            input[i] = input[i] * COMPLEX_C(n_inverse, 0.0);
        }
    }

    std::vector<COMPLEX_C> FFT::ReverseRootTable()
    {
        std::vector<COMPLEX_C> reverse_root_table;

        int lg = log2(n);
        for (int i = 0; i < n; i++)
        {
            reverse_root_table.push_back(root_tables[bitreverse(i, lg)]);
        }

        return reverse_root_table;
    }

    std::vector<COMPLEX_C> FFT::InverseReverseRootTable()
    {
        std::vector<COMPLEX_C> inverse_reverse_root_table;

        int lg = log2(n);
        for (int i = 0; i < n; i++)
        {
            inverse_reverse_root_table.push_back(
                inverse_root_tables[bitreverse(i, lg)]);
        }

        return inverse_reverse_root_table;
    }

} // namespace fft
