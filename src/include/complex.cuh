// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef COMPLEX_ARITHMATIC_H
#define COMPLEX_ARITHMATIC_H

#include <complex>
#include <vector>
#include "cuda_runtime.h"

#ifdef FLOAT_64
typedef std::complex<double> COMPLEX_C;
typedef double FIXED_POINT;
typedef double2 FIXED_POINT2;
#elif defined(FLOAT_32)
typedef std::complex<float> COMPLEX_C;
typedef float FIXED_POINT;
typedef float2 FIXED_POINT2;
#else
#error                                                                         \
    "Please define either USE_FLOAT or USE_DOUBLE to choose the complex type."
#endif

namespace complex_fix_point
{

    // Define the Complex structure
    struct Complex
    {
        FIXED_POINT2 data;
        __host__ __device__ __forceinline__ FIXED_POINT& real()
        {
            return data.x;
        }
        __host__ __device__ __forceinline__ FIXED_POINT& imag()
        {
            return data.y;
        }
        __host__ __device__ __forceinline__ const FIXED_POINT& real() const
        {
            return data.x;
        } //
        __host__ __device__ __forceinline__ const FIXED_POINT& imag() const
        {
            return data.y;
        }

        // Constructor to initialize the complex number
        __host__ __device__ __forceinline__ Complex(FIXED_POINT r,
                                                    FIXED_POINT i)
            : data({r, i})
        {
        }
        __host__ __device__ __forceinline__ Complex() : data({0.0, 0.0}) {}

        // Overload the addition operator (+) within the namespace
        __device__ __forceinline__ Complex operator+(const Complex& other) const
        {
            return Complex(real() + other.real(), imag() + other.imag());
        }

        // Overload the subtraction operator (-) within the namespace
        __device__ __forceinline__ Complex operator-(const Complex& other) const
        {
            return Complex(real() - other.real(), imag() - other.imag());
        }

        // Overload the multiplication operator (*) within the namespace
        __device__ __forceinline__ Complex operator*(const Complex& other) const
        {
            return Complex(real() * other.real() - imag() * other.imag(),
                           real() * other.imag() + imag() * other.real());
        }

        // Overload the division operator (/) within the namespace
        __device__ __forceinline__ Complex operator/(const Complex& other) const
        {
            FIXED_POINT denominator =
                other.real() * other.real() + other.imag() * other.imag();
            return Complex(
                (real() * other.real() + imag() * other.imag()) / denominator,
                (imag() * other.real() - real() * other.imag()) / denominator);
        }

        // Overload the Conjugate of complex number within the namespace
        __device__ __forceinline__ Complex conjugate() const
        {
            return Complex(real(), -imag());
        }

        // Overload the Exponentiation of complex number within the namespace
        __device__ __forceinline__ Complex exp(int& exponent) const
        {
            Complex result(1.0, 0);
            int bits = 32 - __clz(exponent);
            for (int i = bits - 1; i > -1; i--)
            {
                result = result * result;

                if (((exponent >> i) & 1u))
                {
                    result = result * (*this);
                }
            }

            return result;
        }

        // Overload the Exponentiation of complex number within the namespace
        __device__ __forceinline__ Complex
        exp(unsigned long long& exponent) const
        {
            Complex result(1.0, 0);
            int bits = 64 - __clzll(exponent);
            for (int i = bits - 1; i > -1; i--)
            {
                result = result * result;

                if (((exponent >> i) & 1u))
                {
                    result = result * (*this);
                }
            }

            return result;
        }
    };

} // namespace complex_fix_point

typedef complex_fix_point::Complex COMPLEX;

#endif // COMPLEX_ARITHMATIC_H
