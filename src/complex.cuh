// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include <thrust/complex.h>

#include <complex>
#include <vector>

#ifndef COMPLEX_ARITHMATIC_H
#define COMPLEX_ARITHMATIC_H

namespace complex_double
{

// Define the Complex structure
struct Complex
{
    double2 data;
    __host__ __device__ __forceinline__ double& real() { return data.x; }
    __host__ __device__ __forceinline__ double& imag() { return data.y; }
    __host__ __device__ __forceinline__ const double& real() const
    {
        return data.x;
    }  //
    __host__ __device__ __forceinline__ const double& imag() const
    {
        return data.y;
    }

    // Constructor to initialize the complex number
    __host__ __device__ __forceinline__ Complex(double r, double i)
        : data({r, i})
    {
    }
    __host__ __device__ __forceinline__ Complex() : data({0.0, 0.0}) {}

    // Overload the addition operator (+) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator+(const Complex& other) const
    {
        return Complex(real() + other.real(), imag() + other.imag());
    }

    // Overload the subtraction operator (-) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator-(const Complex& other) const
    {
        return Complex(real() - other.real(), imag() - other.imag());
    }

    // Overload the multiplication operator (*) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator*(const Complex& other) const
    {
        return Complex(real() * other.real() - imag() * other.imag(),
                       real() * other.imag() + imag() * other.real());
    }

    // Overload the division operator (/) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator/(const Complex& other) const
    {
        double denominator =
            other.real() * other.real() + other.imag() * other.imag();
        return Complex(
            (real() * other.real() + imag() * other.imag()) / denominator,
            (imag() * other.real() - real() * other.imag()) / denominator);
    }
};
}  // namespace complex_double

namespace complex_float
{

// Define the Complex structure
struct Complex
{
    float2 data;

    __host__ __device__ __forceinline__ float& real() { return data.x; }
    __host__ __device__ __forceinline__ float& imag() { return data.y; }
    __host__ __device__ __forceinline__ const float& real() const
    {
        return data.x;
    }  //
    __host__ __device__ __forceinline__ const float& imag() const
    {
        return data.y;
    }

    // Constructor to initialize the complex number
    __host__ __device__ __forceinline__ Complex(float r, float i) : data({r, i})
    {
    }
    __host__ __device__ __forceinline__ Complex() : data({0.0, 0.0}) {}

    // Overload the addition operator (+) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator+(const Complex& other) const
    {
        return Complex(real() + other.real(), imag() + other.imag());
    }

    // Overload the subtraction operator (-) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator-(const Complex& other) const
    {
        return Complex(real() - other.real(), imag() - other.imag());
    }

    // Overload the multiplication operator (*) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator*(const Complex& other) const
    {
        return Complex(real() * other.real() - imag() * other.imag(),
                       real() * other.imag() + imag() * other.real());
    }

    // Overload the division operator (/) within the namespace
    __host__ __device__ __forceinline__ Complex
    operator/(const Complex& other) const
    {
        float denominator =
            other.real() * other.real() + other.imag() * other.imag();
        return Complex(
            (real() * other.real() + imag() * other.imag()) / denominator,
            (imag() * other.real() - real() * other.imag()) / denominator);
    }
};
}  // namespace complex_float

#ifdef FLOAT_64
typedef complex_double::Complex COMPLEX;
typedef std::complex<double> COMPLEX_C;
#elif defined(FLOAT_32)
typedef complex_float::Complex COMPLEX;
typedef std::complex<float> COMPLEX_C;
#elif defined(THRUST_FLOAT_64)
typedef thrust::complex<double> COMPLEX;
typedef std::complex<double> COMPLEX_C;
#elif defined(THRUST_FLOAT_32)
typedef thrust::complex<float> COMPLEX;
typedef std::complex<float> COMPLEX_C;
#else
#error \
    "Please define either USE_FLOAT or USE_DOUBLE to choose the complex type."
#endif

#endif  // COMPLEX_ARITHMATIC_H
