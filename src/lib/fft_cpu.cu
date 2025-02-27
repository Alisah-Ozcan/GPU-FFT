// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "fft_cpu.cuh"

namespace gpufft
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

    template <typename T>
    std::vector<T> schoolbook_poly_multiplication(std::vector<T> a,
                                                  std::vector<T> b, T modulus,
                                                  int size)
    {
        std::vector<T> mult_vector(size * 2, 0);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                T mult = a[i] * b[j];
                mult_vector[i + j] += mult;
            }
        }

        // for(int i = 0; i < 2 * size; i++)
        //{
        //     mult_vector[i] = mult_vector[i] % modulus;
        // }

        for (int i = 0; i < size; i++)
        {
            mult_vector[i] = (mult_vector[i] - mult_vector[i + size]) % modulus;
        }

        return mult_vector;
    }

    template <typename T>
    std::vector<T> schoolbook_poly_multiplication_without_reduction(
        std::vector<T> a, std::vector<T> b, T modulus, int size)
    {
        std::vector<T> mult_vector(size * 2, 0);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                T mult = a[i] * b[j];
                mult_vector[i + j] += mult;
            }
        }

        for (int i = 0; i < 2 * size; i++)
        {
            mult_vector[i] = mult_vector[i] % modulus;
        }

        return mult_vector;
    }

    template std::vector<int> schoolbook_poly_multiplication(std::vector<int> a,
                                                             std::vector<int> b,
                                                             int modulus,
                                                             int size);

    template std::vector<unsigned long long>
    schoolbook_poly_multiplication(std::vector<unsigned long long> a,
                                   std::vector<unsigned long long> b,
                                   unsigned long long modulus, int size);

    template std::vector<int> schoolbook_poly_multiplication_without_reduction(
        std::vector<int> a, std::vector<int> b, int modulus, int size);

    template std::vector<unsigned long long>
    schoolbook_poly_multiplication_without_reduction(
        std::vector<unsigned long long> a, std::vector<unsigned long long> b,
        unsigned long long modulus, int size);

    template <typename T> FFT<T>::FFT(int size)
    {
        n = size;
        logn = int(log2(n));
        COMPLEX<T> x_(1.0, 0.0);
        x = x_;
        max_size = n * 2;
        root = COMPLEX<T>(2.0) * COMPLEX<T>(M_PI) /
               COMPLEX<T>(max_size); // it was float
        n_inverse = 1.0 / max_size;

        GenerateRootTable();
        GenerateInverseRootTable();
    }

    template <typename T> void FFT<T>::GenerateRootTable()
    {
        COMPLEX<T> j(0.0, 1.0); // Define the complex unit (imaginary part)

        for (int i = 0; i < n; i++)
        {
            COMPLEX<T> element = complex_arithmetic::exp(j * COMPLEX<T>(i) *
                                                         root); // it was float
            root_tables.push_back(element);
        }
    }

    template <typename T> void FFT<T>::GenerateInverseRootTable()
    {
        COMPLEX<T> one(1.0); // Define the complex unit (imaginary part)

        for (int i = 0; i < n; i++)
        {
            COMPLEX<T> element = one / root_tables[i];
            inverse_root_tables.push_back(element);
        }
    }

    template <typename T> void FFT<T>::fft(std::vector<COMPLEX<T>>& input)
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

                COMPLEX<T> S = root_tables[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX<T> U = input[j];
                    COMPLEX<T> V = input[j + t];

                    input[j] = U + (V * S);
                    input[j + t] = U - (V * S);
                }
            }

            m = m << 1;
        }
    }

    template <typename T> void FFT<T>::ifft(std::vector<COMPLEX<T>>& input)
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

                COMPLEX<T> S = inverse_root_tables[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX<T> U = input[j];
                    COMPLEX<T> V = input[j + t];

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
            input[i] = input[i] * n_inverse;
        }
    }

    template <typename T> std::vector<COMPLEX<T>> FFT<T>::ReverseRootTable()
    {
        std::vector<COMPLEX<T>> reverse_root_table;

        int lg = log2(n);
        for (int i = 0; i < n; i++)
        {
            reverse_root_table.push_back(root_tables[bitreverse(i, lg)]);
        }

        return reverse_root_table;
    }

    template <typename T>
    std::vector<COMPLEX<T>> FFT<T>::InverseReverseRootTable()
    {
        std::vector<COMPLEX<T>> inverse_reverse_root_table;

        int lg = log2(n);
        for (int i = 0; i < n; i++)
        {
            inverse_reverse_root_table.push_back(
                inverse_root_tables[bitreverse(i, lg)]);
        }

        return inverse_reverse_root_table;
    }

    template class FFT<Float32>;
    template class FFT<Float64>;

    ////////////////////////////////////////////////
    ////////////////////////////////////////////////

    template <typename T> FFNT<T>::FFNT(int size)
    {
        n = size;
        logn = int(log2(n));
        COMPLEX<T> x_(1.0, 0.0);
        x = x_;
        max_size = n * 2;
        root = 2.0 * M_PI / static_cast<T>(max_size);
        n_inverse = 1.0 / max_size;

        GenerateRootTable();
        GenerateInverseRootTable();

        GenerateRootTableTwist();
        GenerateInverseRootTableTwist();
    }

    template <typename T> void FFNT<T>::GenerateRootTable()
    {
        COMPLEX<T> j(0.0, 1.0); // Define the complex unit (imaginary part)

        T root_new = 2.0 * M_PI / static_cast<T>(n >> 1);

        for (int i = 0; i < (n >> 1); i++)
        {
            COMPLEX<T> element = complex_arithmetic::exp(j * COMPLEX<T>(i) *
                                                         COMPLEX<T>(root_new));
            root_tables_new.push_back(element);
        }
    }

    template <typename T> void FFNT<T>::GenerateInverseRootTable()
    {
        COMPLEX<T> one(1.0); // Define the complex unit (imaginary part)

        for (int i = 0; i < (n >> 1); i++)
        {
            COMPLEX<T> element = one / root_tables_new[i];
            inverse_root_tables_new.push_back(element);
        }
    }

    template <typename T> void FFNT<T>::GenerateRootTableTwist()
    {
        COMPLEX<T> j(0.0, 1.0); // Define the complex unit (imaginary part)

        T root_new = 2.0 * M_PI / static_cast<T>(2 * n);

        for (int i = 0; i < (n >> 1); i++)
        {
            COMPLEX<T> element = complex_arithmetic::exp(j * COMPLEX<T>(i) *
                                                         COMPLEX<T>(root_new));
            root_tables_twist.push_back(element);
        }
    }

    template <typename T> void FFNT<T>::GenerateInverseRootTableTwist()
    {
        COMPLEX<T> one(1.0); // Define the complex unit (imaginary part)

        for (int i = 0; i < (n >> 1); i++)
        {
            COMPLEX<T> element = one / root_tables_twist[i];
            inverse_root_tables_twist.push_back(element);
        }
    }

    template <typename T>
    std::vector<unsigned long long>
    FFNT<T>::MULT(std::vector<unsigned long long>& input1,
                  std::vector<unsigned long long>& input2)
    {
        std::vector<COMPLEX<T>> input1_complex;
        std::vector<COMPLEX<T>> input2_complex;
        for (int i = 0; i < (n >> 1); i++)
        {
            COMPLEX<T> temp1(input1[i], input1[i + (n >> 1)]); // fold
            temp1 = temp1 * root_tables_twist[i];
            input1_complex.push_back(temp1);

            COMPLEX<T> temp2(input2[i], input2[i + (n >> 1)]); // fold
            temp2 = temp2 * root_tables_twist[i];
            input2_complex.push_back(temp2);
        }

        // FFT1
        int t = (n >> 1);
        int logn_new = int(log2((n >> 1)));
        int m = 1;

        while (m < (n >> 1))
        {
            t = t >> 1;

            for (int i = 0; i < m; i++)
            {
                int j1 = 2 * i * t;
                int j2 = j1 + t - 1;

                int index = bitreverse(i, logn_new - 1);

                COMPLEX<T> S = root_tables_new[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX<T> U = input1_complex[j];
                    COMPLEX<T> V = input1_complex[j + t];

                    input1_complex[j] = U + (V * S);
                    input1_complex[j + t] = U - (V * S);
                }
            }

            m = m << 1;
        }

        // FFT2
        t = (n >> 1);
        logn_new = int(log2((n >> 1)));
        m = 1;

        while (m < (n >> 1))
        {
            t = t >> 1;

            for (int i = 0; i < m; i++)
            {
                int j1 = 2 * i * t;
                int j2 = j1 + t - 1;

                int index = bitreverse(i, logn_new - 1);

                COMPLEX<T> S = root_tables_new[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX<T> U = input2_complex[j];
                    COMPLEX<T> V = input2_complex[j + t];

                    input2_complex[j] = U + (V * S);
                    input2_complex[j + t] = U - (V * S);
                }
            }

            m = m << 1;
        }

        // Multiplication
        for (int i = 0; i < (n >> 1); i++)
        {
            input1_complex[i] = input1_complex[i] * input2_complex[i];
        }

        // IFFT
        t = 1;
        m = (n >> 1);
        while (m > 1)
        {
            int j1 = 0;
            int h = m >> 1;
            for (int i = 0; i < h; i++)
            {
                int j2 = j1 + t - 1;
                int index = bitreverse(i, logn_new - 1);

                COMPLEX<T> S = inverse_root_tables_new[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    COMPLEX<T> U = input1_complex[j];
                    COMPLEX<T> V = input1_complex[j + t];

                    input1_complex[j] = (U + V);
                    input1_complex[j + t] = (U - V) * S;
                }

                j1 = j1 + (t << 1);
            }

            t = t << 1;
            m = m >> 1;
        }

        T n_inverse_new = 1.0 / (n >> 1);
        std::vector<unsigned long long> result(n);
        for (int i = 0; i < (n >> 1); i++)
        {
            input1_complex[i] =
                input1_complex[i] * inverse_root_tables_twist[i]; // untwist
            input1_complex[i] =
                input1_complex[i] * COMPLEX<T>(n_inverse_new); // div

            result[i] = std::round(input1_complex[i].real()); // unfold
            result[i + (n >> 1)] =
                std::round(input1_complex[i].imag()); // unfold
        }

        return result;
    }

    template <typename T>
    std::vector<COMPLEX<T>> FFNT<T>::ReverseRootTable_ffnt()
    {
        std::vector<COMPLEX<T>> reverse_root_table;
        int logn_new = int(log2((n >> 1)));
        for (int i = 0; i < (n >> 1); i++)
        {
            reverse_root_table.push_back(
                root_tables_new[bitreverse(i, logn_new - 1)]);
        }

        return reverse_root_table;
    }

    template <typename T>
    std::vector<COMPLEX<T>> FFNT<T>::InverseReverseRootTable_ffnt()
    {
        std::vector<COMPLEX<T>> inverse_reverse_root_table;
        int logn_new = int(log2((n >> 1)));
        for (int i = 0; i < (n >> 1); i++)
        {
            inverse_reverse_root_table.push_back(
                inverse_root_tables_new[bitreverse(i, logn_new - 1)]);
        }

        return inverse_reverse_root_table;
    }

    template <typename T> std::vector<COMPLEX<T>> FFNT<T>::twist_table_ffnt()
    {
        return root_tables_twist;
    }

    template <typename T> std::vector<COMPLEX<T>> FFNT<T>::untwist_table_ffnt()
    {
        return inverse_root_tables_twist;
    }

    template class FFNT<Float32>;
    template class FFNT<Float64>;

} // namespace gpufft
