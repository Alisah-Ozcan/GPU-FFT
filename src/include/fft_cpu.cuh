// Copyright 2023-2025 Alişah Özcan
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

    template <typename T>
    std::vector<T> schoolbook_poly_multiplication(std::vector<T> a,
                                                  std::vector<T> b, T modulus,
                                                  int size);

    template <typename T>
    std::vector<T> schoolbook_poly_multiplication_without_reduction(
        std::vector<T> a, std::vector<T> b, T modulus, int size);

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

    ////////////////////////////////////////////////
    ////////////////////////////////////////////////

    template <typename T> class FFNT
    {
      public:
        int n;
        int logn;
        COMPLEX<T> x;
        int max_size;

        COMPLEX<T> root; // it was float
        std::vector<COMPLEX<T>> root_tables_new;
        std::vector<COMPLEX<T>> inverse_root_tables_new;

        std::vector<COMPLEX<T>> root_tables_twist;
        std::vector<COMPLEX<T>> inverse_root_tables_twist;

        T n_inverse;

        FFNT(int size);

      private:
        void GenerateRootTable();

        void GenerateInverseRootTable();

        void GenerateRootTableTwist();

        void GenerateInverseRootTableTwist();

      public:
        std::vector<unsigned long long>
        MULT(std::vector<unsigned long long>& input1,
             std::vector<unsigned long long>& input2);

        std::vector<COMPLEX<T>> ReverseRootTable_ffnt();

        std::vector<COMPLEX<T>> InverseReverseRootTable_ffnt();

        std::vector<COMPLEX<T>> twist_table_ffnt();

        std::vector<COMPLEX<T>> untwist_table_ffnt();
    };

} // namespace gpufft
#endif // CPU_FFT_H