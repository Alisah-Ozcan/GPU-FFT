// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef GPUFFT_EXAMPLE_UTIL_H
#define GPUFFT_EXAMPLE_UTIL_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace gpufft_example
{
    // Command line of the examples:
    //   <program> [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
    // with defaults 12, 1 and 64 (the data type Float32 or Float64).
    struct ExampleArgs
    {
        int logn = 12;
        int batch = 1;
        int bits = 64;
    };

    inline ExampleArgs parse_example_args(int argc, char* argv[])
    {
        ExampleArgs args;
        if (argc > 1)
            args.logn = atoi(argv[1]);
        if (argc > 2)
            args.batch = atoi(argv[2]);
        if (argc > 3)
            args.bits = atoi(argv[3]);

        if (argc > 4 || args.logn < 1 || args.batch < 1 ||
            (args.bits != 32 && args.bits != 64))
        {
            std::cerr << "Usage: " << argv[0]
                      << " [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        return args;
    }

    // Exits if logn is outside [min_logn, max_logn].
    inline void check_logn(int logn, int min_logn, int max_logn)
    {
        if (logn < min_logn || logn > max_logn)
        {
            std::cerr << "RING_SIZE_IN_LOG2 should be in range " << min_logn
                      << " to " << max_logn << "." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Float32 holds the exact products of smaller coefficients.
    template <typename T> int coefficient_bound()
    {
        return sizeof(T) == 4 ? 2 : 7;
    }

    // a * b mod X^n + 1, without reduction of the coefficients.
    inline std::vector<long long>
    negacyclic_multiplication(const std::vector<int>& a,
                              const std::vector<int>& b)
    {
        const int size = static_cast<int>(a.size());
        std::vector<long long> result(size, 0);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                long long mult = static_cast<long long>(a[i]) * b[j];
                if (i + j < size)
                {
                    result[i + j] += mult;
                }
                else
                {
                    result[i + j - size] -= mult;
                }
            }
        }
        return result;
    }

    // a * b without reduction (2n coefficients).
    inline std::vector<long long>
    linear_multiplication(const std::vector<int>& a, const std::vector<int>& b)
    {
        std::vector<long long> result(a.size() + b.size(), 0);
        for (size_t i = 0; i < a.size(); i++)
        {
            for (size_t j = 0; j < b.size(); j++)
            {
                result[i + j] += static_cast<long long>(a[i]) * b[j];
            }
        }
        return result;
    }

    // Prints the first values and the first mismatch; returns true if all
    // rounded results are equal to the expected ones.
    template <typename V>
    bool check_rounded(const std::vector<V>& got,
                       const std::vector<long long>& expected, int poly)
    {
        for (size_t i = 0; i < expected.size(); i++)
        {
            const long long value = std::llround(static_cast<double>(got[i]));
            if (poly == 0 && i < 10)
            {
                std::cout << i << ": " << expected[i] << " - " << value
                          << std::endl;
            }
            if (value != expected[i])
            {
                std::cout << "Error in poly " << poly << ", index " << i
                          << ": expected " << expected[i] << ", got " << got[i]
                          << std::endl;
                return false;
            }
        }
        return true;
    }

} // namespace gpufft_example
#endif // GPUFFT_EXAMPLE_UTIL_H
