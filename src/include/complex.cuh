// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef COMPLEX_ARITHMETIC_H
#define COMPLEX_ARITHMETIC_H

#include <iostream>
#include <complex>
#include <vector>
#include "cuda_runtime.h"

template <typename T>
using fp2 = typename std::conditional<std::is_same<T, float>::value, float2,
                                      double2>::type;

typedef float Float32;
typedef double Float64;

template <typename T>
using Float = typename std::conditional<std::is_same<T, float>::value, Float32,
                                        Float64>::type;

namespace complex_arithmetic
{
    // Define the Complex structure
    template <typename T1> struct ComplexOperations
    {
        fp2<T1> data;

        __host__ __device__ __forceinline__ T1& real() { return data.x; }
        __host__ __device__ __forceinline__ T1& imag() { return data.y; }
        __host__ __device__ __forceinline__ const T1& real() const
        {
            return data.x;
        }
        __host__ __device__ __forceinline__ const T1& imag() const
        {
            return data.y;
        }

        // Constructor to initialize the complex number
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations(T2 r, T2 i)
        {
            data.x = static_cast<T1>(r);
            data.y = static_cast<T1>(i);
        }

        // Constructor to initialize the complex number with real number
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations(T2 r)
        {
            data.x = static_cast<T1>(r);
            data.y = static_cast<T1>(0.0);
        }

        __host__ __device__ __forceinline__ ComplexOperations()
            : data({0.0, 0.0})
        {
        }

        // Complex addition operator (+)
        __host__ __device__ __forceinline__ ComplexOperations
        operator+(const ComplexOperations& other) const
        {
            return ComplexOperations(real() + other.real(),
                                     imag() + other.imag());
        }

        // Complex addition and equal operator (+=)
        __host__ __device__ __forceinline__ ComplexOperations
        operator+=(const ComplexOperations& other) const
        {
            real() += other.real();
            imag() += other.imag();
            return *this;
        }

        // Complex, scalar addition operator (+)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator+(const T2& other) const
        {
            return ComplexOperations(real() + static_cast<T1>(other), imag());
        }

        // Complex, scalar addition and equal operator (+=)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator+=(const T2& other) const
        {
            real() += static_cast<T1>(other);
            return *this;
        }

        // Complex subtraction operator (-)
        __host__ __device__ __forceinline__ ComplexOperations
        operator-(const ComplexOperations& other) const
        {
            return ComplexOperations(real() - other.real(),
                                     imag() - other.imag());
        }

        // Complex subtraction and equal operator (-=)
        __host__ __device__ __forceinline__ ComplexOperations
        operator-=(const ComplexOperations& other) const
        {
            real() -= other.real();
            imag() -= other.imag();
            return *this;
        }

        // Complex, scalar subtraction operator (-)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator-(const T2& other) const
        {
            return ComplexOperations(real() - static_cast<T1>(other), imag());
        }

        // Complex, scalar subtraction and equal operator (-=)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator-=(const T2& other) const
        {
            real() -= static_cast<T1>(other);
            return *this;
        }

        // Complex multiplication operator (*)
        __host__ __device__ __forceinline__ ComplexOperations
        operator*(const ComplexOperations& other) const
        {
            return ComplexOperations(
                real() * other.real() - imag() * other.imag(),
                real() * other.imag() + imag() * other.real());
        }

        // Complex multiplication equal operator (*=)
        __host__ __device__ __forceinline__ ComplexOperations
        operator*=(const ComplexOperations& other) const
        {
            T1 r = real() * other.real() - imag() * other.imag();
            T1 i = real() * other.imag() + imag() * other.real();
            real() = r;
            imag() = i;
            return *this;
        }

        // Complex, scalar multiplication operator (*)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator*(const T2& other) const
        {
            return ComplexOperations(real() * static_cast<T1>(other),
                                     real() * static_cast<T1>(other));
        }

        // Complex, scalar multiplication equal operator (*=)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator*=(const T2& other) const
        {
            real() *= static_cast<T1>(other);
            imag() *= static_cast<T1>(other);
            return *this;
        }

        // Complex division operator (/)
        __host__ __device__ __forceinline__ ComplexOperations
        operator/(const ComplexOperations& other) const
        {
            T1 denominator =
                other.real() * other.real() + other.imag() * other.imag();
            return ComplexOperations(
                (real() * other.real() + imag() * other.imag()) / denominator,
                (imag() * other.real() - real() * other.imag()) / denominator);
        }

        // Complex division operator (/=)
        __host__ __device__ __forceinline__ ComplexOperations
        operator/=(const ComplexOperations& other) const
        {
            T1 denominator =
                other.real() * other.real() + other.imag() * other.imag();
            T1 r =
                (real() * other.real() + imag() * other.imag()) / denominator;
            T1 i =
                (imag() * other.real() - real() * other.imag()) / denominator;
            real() = r;
            imag() = i;
            return *this;
        }

        // Complex, scalar division operator (/)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator/(const T2& other) const
        {
            return ComplexOperations(real() / static_cast<T1>(other),
                                     imag() / static_cast<T1>(other));
        }

        // Complex, scalar division operator (/=)
        template <typename T2>
        __host__ __device__ __forceinline__ ComplexOperations
        operator/=(const T2& other) const
        {
            real() /= static_cast<T1>(other);
            imag() /= static_cast<T1>(other);
            return *this;
        }

        // Complex Conjugate of complex number
        __host__ __device__ __forceinline__ ComplexOperations conjugate() const
        {
            return ComplexOperations(real(), -imag());
        }

        // Complex Negate of complex number
        __host__ __device__ __forceinline__ ComplexOperations negate() const
        {
            return ComplexOperations(-real(), -imag());
        }

        // Complex Square of complex number
        __host__ __device__ __forceinline__ T1 square() const
        {
            return real() * real() + imag() * imag();
        }

        // Complex Inverse of complex number
        __host__ __device__ __forceinline__ ComplexOperations inverse() const
        {
            T1 squared = square();
            ComplexOperations conj = conjugate();
            return ComplexOperations(conj.real() / squared,
                                     conj.imag() / squared);
        }

        // Complex Exponentiation of complex number
        __host__ __device__ __forceinline__ ComplexOperations
        exp(int& exponent) const
        {
            ComplexOperations result(1.0, 0.0);

            if (exponent == 0)
            {
                return result;
            }

            int bits = 32 - __builtin_clz(exponent);
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

        // Complex Exponentiation of complex number
        __host__ __device__ __forceinline__ ComplexOperations
        exp(unsigned long long& exponent) const
        {
            ComplexOperations result(1.0, 0);

            if (exponent == 0ULL)
            {
                return result;
            }

            int bits = 64 - __builtin_clzll(exponent);
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

        friend __host__ std::ostream& operator<<(std::ostream& out,
                                                 const ComplexOperations& c)
        {
            out << c.real();
            if (c.imag() >= 0)
                out << " + " << c.imag() << "i";
            else
                out << " - " << -c.imag() << "i";
            return out;
        }
    };

    template <typename T>
    ComplexOperations<T> exp(const ComplexOperations<T>& input)
    {
        T exp_real = std::exp(input.real());
        return ComplexOperations<T>(exp_real * std::cos(input.imag()),
                                    exp_real * std::sin(input.imag()));
    }

} // namespace complex_arithmetic

template <typename T> using COMPLEX = complex_arithmetic::ComplexOperations<T>;

typedef COMPLEX<Float32> Complex32;
typedef COMPLEX<Float64> Complex64;

#endif // COMPLEX_ARITHMETIC_H
