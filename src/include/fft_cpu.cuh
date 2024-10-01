// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef CPU_FFT_H
#define CPU_FFT_H

#include "complex.cuh"

namespace fft
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

    class FFT
    {
      public:
        static int n;
        static int logn;
        static COMPLEX_C x;
        static int max_size;

        static COMPLEX_C root; // it was float
        static std::vector<COMPLEX_C> root_tables;
        static std::vector<COMPLEX_C> inverse_root_tables;

        static float n_inverse;

        FFT(int size);

      private:
        static void GenerateRootTable();

        static void GenerateInverseRootTable();

      public:
        void fft(std::vector<COMPLEX_C>& input);

        void ifft(std::vector<COMPLEX_C>& input);

        std::vector<COMPLEX_C> ReverseRootTable();

        std::vector<COMPLEX_C> InverseReverseRootTable();
    };

} // namespace fft
#endif // CPU_FFT_H