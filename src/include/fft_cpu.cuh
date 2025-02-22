// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef CPU_FFT_H
#define CPU_FFT_H

#include "complex.cuh"

namespace gpufft
{

    int bitreverse(int index, int n_power);

    enum ReductionPolynomial
    {
        X_N_plus,
        X_N_minus
    }; // X_N_minus: X^n - 1, X_N_plus: X^n + 1

    std::vector<unsigned long long>
    schoolbook_poly_multiplication(std::vector<unsigned long long> a,
                                   std::vector<unsigned long long> b,
                                   unsigned long long modulus, int size);

    std::vector<unsigned long long>
    schoolbook_poly_multiplication_without_reduction(
        std::vector<unsigned long long> a, std::vector<unsigned long long> b,
        unsigned long long modulus, int size);

    template <typename T> class FFT
    {
      public:
        int n;
        int logn;
        COMPLEX<T> x;
        int max_size;

        COMPLEX<T> root; // it was float
        std::vector<COMPLEX<T>> root_tables;
        std::vector<COMPLEX<T>> inverse_root_tables;

        Float<T> n_inverse;

        FFT(int size);

      private:
        void GenerateRootTable();

        void GenerateInverseRootTable();

      public:
        void fft(std::vector<COMPLEX<T>>& input);

        void ifft(std::vector<COMPLEX<T>>& input);

        std::vector<COMPLEX<T>> ReverseRootTable();

        std::vector<COMPLEX<T>> InverseReverseRootTable();
    };

} // namespace gpufft
#endif // CPU_FFT_H