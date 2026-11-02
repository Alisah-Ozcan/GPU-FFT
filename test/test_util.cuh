// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef GPUFFT_TEST_UTIL_H
#define GPUFFT_TEST_UTIL_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "gpufft/fft.cuh"
#include "gpufft/fft_cpu.cuh"

namespace gpufft_test
{
    using namespace gpufft;

    // The references run in double precision for both data types.
    using Ref = COMPLEX<double>;

    template <typename T> class DeviceBuffer
    {
      public:
        explicit DeviceBuffer(size_t count) : count_(count)
        {
            GPUFFT_CUDA_CHECK(
                cudaMalloc(&ptr_, (count > 0 ? count : 1) * sizeof(T)));
        }

        explicit DeviceBuffer(const std::vector<T>& host)
            : DeviceBuffer(host.size())
        {
            upload(host);
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        ~DeviceBuffer() { cudaFree(ptr_); }

        T* get() const { return ptr_; }

        void upload(const std::vector<T>& host)
        {
            GPUFFT_CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T),
                                         cudaMemcpyHostToDevice));
        }

        std::vector<T> download() const
        {
            std::vector<T> host(count_);
            GPUFFT_CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T),
                                         cudaMemcpyDeviceToHost));
            return host;
        }

      private:
        T* ptr_ = nullptr;
        size_t count_;
    };

    inline std::vector<Ref> random_complex(std::mt19937_64& rng, size_t count)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<Ref> values(count);
        for (auto& v : values)
        {
            double re = dist(rng);
            v = Ref(re, dist(rng));
        }
        return values;
    }

    inline std::vector<double> random_real(std::mt19937_64& rng, size_t count)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> values(count);
        for (auto& v : values)
        {
            v = dist(rng);
        }
        return values;
    }

    inline std::vector<int> random_ints(std::mt19937_64& rng, size_t count,
                                        int low, int high)
    {
        std::uniform_int_distribution<int> dist(low, high);
        std::vector<int> values(count);
        for (auto& v : values)
        {
            v = dist(rng);
        }
        return values;
    }

    // Rounds double precision values to T (the input of the GPU).
    template <typename T>
    std::vector<COMPLEX<T>> to_t(const std::vector<Ref>& v)
    {
        std::vector<COMPLEX<T>> result(v.size());
        for (size_t i = 0; i < v.size(); i++)
        {
            result[i] = COMPLEX<T>(static_cast<T>(v[i].real()),
                                   static_cast<T>(v[i].imag()));
        }
        return result;
    }

    template <typename T> std::vector<T> to_t(const std::vector<double>& v)
    {
        return std::vector<T>(v.begin(), v.end());
    }

    // Values of T as double precision (the input of the references).
    template <typename T>
    std::vector<Ref> to_ref(const std::vector<COMPLEX<T>>& v)
    {
        std::vector<Ref> result(v.size());
        for (size_t i = 0; i < v.size(); i++)
        {
            result[i] = Ref(static_cast<double>(v[i].real()),
                            static_cast<double>(v[i].imag()));
        }
        return result;
    }

    template <typename T> std::vector<double> to_ref(const std::vector<T>& v)
    {
        return std::vector<double>(v.begin(), v.end());
    }

    template <typename T>
    std::vector<T> slice(const std::vector<T>& v, size_t begin, size_t count)
    {
        return std::vector<T>(v.begin() + begin, v.begin() + begin + count);
    }

    // ---------------- Root tables ----------------

    // GPU_FFT with X_N_minus: N / 2 entries, exp(2 pi i bitrev(j) / N).
    inline std::vector<Ref> cyclic_table(int logn, bool inverse)
    {
        FFT<double> generator(1 << (logn - 1));
        return inverse ? generator.InverseReverseRootTable()
                       : generator.ReverseRootTable();
    }

    // GPU_FFT with X_N_plus: N entries, exp(pi i bitrev(j) / N).
    inline std::vector<Ref> negacyclic_table(int logn, bool inverse)
    {
        const int n = 1 << logn;
        std::vector<Ref> table(n);
        for (int j = 0; j < n; j++)
        {
            double angle = M_PI * bitreverse(j, logn) / n;
            table[j] = Ref(std::cos(angle),
                           inverse ? -std::sin(angle) : std::sin(angle));
        }
        return table;
    }

    // GPU_Special_FFT: l = 2^logl slots, entry m / 2 + j of the stage with
    // sub-arrays of m elements is Psi[(5^j mod 4m) * (l / m)], where
    // Psi[k] = exp(pi i k / (2l)) (see the comment of GPU_Special_FFT).
    inline std::vector<Ref> special_table(int logl, bool inverse)
    {
        const int l = 1 << logl;
        std::vector<Ref> table(l, Ref(0.0, 0.0));
        for (int logm = 1; logm <= logl; logm++)
        {
            const int m = 1 << logm;
            long long rotation = 1;
            for (int j = 0; j < m / 2; j++)
            {
                const long long k = (rotation % (4 * m)) * (l / m);
                const double angle = M_PI * k / (2.0 * l);
                table[m / 2 + j] =
                    Ref(std::cos(angle),
                        inverse ? -std::sin(angle) : std::sin(angle));
                rotation = (5 * rotation) % (4 * l);
            }
        }
        return table;
    }

    // ---------------- References (natural order in, bit-reversed out) ----

    // Forward transform of GPU_FFT: Cooley-Tukey, 2^logn points.
    inline void reference_forward(std::vector<Ref>& a, int logn,
                                  const std::vector<Ref>& table,
                                  ReductionPolynomial poly)
    {
        const int n = 1 << logn;
        for (int m = 1, t = n >> 1; m < n; m <<= 1, t >>= 1)
        {
            for (int i = 0; i < m; i++)
            {
                const Ref root =
                    table[(poly == ReductionPolynomial::X_N_minus) ? i : m + i];
                for (int j = 2 * i * t; j < 2 * i * t + t; j++)
                {
                    Ref u = a[j];
                    Ref v = a[j + t] * root;
                    a[j] = u + v;
                    a[j + t] = u - v;
                }
            }
        }
    }

    // Inverse transform of GPU_FFT: Gentleman-Sande, then times n_inverse.
    inline void reference_inverse(std::vector<Ref>& a, int logn,
                                  const std::vector<Ref>& table,
                                  ReductionPolynomial poly, Ref n_inverse)
    {
        const int n = 1 << logn;
        for (int m = n >> 1, t = 1; m >= 1; m >>= 1, t <<= 1)
        {
            for (int i = 0; i < m; i++)
            {
                const Ref root =
                    table[(poly == ReductionPolynomial::X_N_minus) ? i : m + i];
                for (int j = 2 * i * t; j < 2 * i * t + t; j++)
                {
                    Ref u = a[j];
                    Ref v = a[j + t];
                    a[j] = u + v;
                    a[j + t] = (u - v) * root;
                }
            }
        }
        for (auto& x : a)
        {
            x = x * n_inverse;
        }
    }

    // GPU_Special_FFT forward: Cooley-Tukey over sub-arrays m = 2 .. l.
    inline void reference_special_forward(std::vector<Ref>& w, int logl,
                                          const std::vector<Ref>& table)
    {
        const int l = 1 << logl;
        for (int m = 2; m <= l; m <<= 1)
        {
            for (int i = 0; i < l; i += m)
            {
                for (int j = 0; j < m / 2; j++)
                {
                    Ref u = w[i + j];
                    Ref v = w[i + j + m / 2] * table[m / 2 + j];
                    w[i + j] = u + v;
                    w[i + j + m / 2] = u - v;
                }
            }
        }
    }

    // GPU_Special_FFT inverse: Gentleman-Sande over m = l .. 2, then times
    // n_inverse.
    inline void reference_special_inverse(std::vector<Ref>& w, int logl,
                                          const std::vector<Ref>& table,
                                          Ref n_inverse)
    {
        const int l = 1 << logl;
        for (int m = l; m >= 2; m >>= 1)
        {
            for (int i = 0; i < l; i += m)
            {
                for (int j = 0; j < m / 2; j++)
                {
                    Ref u = w[i + j];
                    Ref v = w[i + j + m / 2];
                    w[i + j] = u + v;
                    w[i + j + m / 2] = (u - v) * table[m / 2 + j];
                }
            }
        }
        for (auto& x : w)
        {
            x = x * n_inverse;
        }
    }

    // a * b mod X^n - 1 (cyclic) or X^n + 1 (negacyclic), exact.
    inline std::vector<long long> schoolbook(const std::vector<int>& a,
                                             const std::vector<int>& b,
                                             bool negacyclic)
    {
        const size_t n = a.size();
        std::vector<long long> result(n, 0);
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                const long long mult = static_cast<long long>(a[i]) * b[j];
                if (i + j < n)
                {
                    result[i + j] += mult;
                }
                else
                {
                    result[i + j - n] += negacyclic ? -mult : mult;
                }
            }
        }
        return result;
    }

    // ---------------- Reporting ----------------

    class TestReport
    {
      public:
        explicit TestReport(std::string suite) : suite_(std::move(suite)) {}

        // Largest error relative to the largest magnitude of the reference.
        template <typename V>
        void check_close(const std::string& name, const std::vector<V>& got,
                         const std::vector<Ref>& expected, double tolerance)
        {
            cases_++;
            if (got.size() != expected.size())
            {
                fail(name, "size mismatch");
                return;
            }
            double error = 0.0;
            double scale = 0.0;
            size_t worst = 0;
            for (size_t i = 0; i < got.size(); i++)
            {
                const double e = std::hypot(
                    static_cast<double>(got[i].real()) - expected[i].real(),
                    static_cast<double>(got[i].imag()) - expected[i].imag());
                if (!(e <= error))
                {
                    error = e;
                    worst = i;
                }
                scale = std::max(
                    scale, std::hypot(expected[i].real(), expected[i].imag()));
            }
            const double relative = error / (scale > 0.0 ? scale : 1.0);
            if (!(relative <= tolerance))
            {
                fail(name, "relative error " + std::to_string(relative) +
                               " above " + std::to_string(tolerance) +
                               ", worst at index " + std::to_string(worst));
            }
        }

        template <typename V>
        void check_close(const std::string& name, const std::vector<V>& got,
                         const std::vector<double>& expected, double tolerance)
        {
            std::vector<Ref> got_complex(got.size());
            std::vector<Ref> expected_complex(expected.size());
            for (size_t i = 0; i < got.size(); i++)
            {
                got_complex[i] = Ref(static_cast<double>(got[i]), 0.0);
            }
            for (size_t i = 0; i < expected.size(); i++)
            {
                expected_complex[i] = Ref(expected[i], 0.0);
            }
            check_close(name, got_complex, expected_complex, tolerance);
        }

        // Rounds the results to integers and compares them exactly.
        template <typename V>
        void check_rounded(const std::string& name, const std::vector<V>& got,
                           const std::vector<long long>& expected)
        {
            cases_++;
            if (got.size() != expected.size())
            {
                fail(name, "size mismatch");
                return;
            }
            size_t mismatches = 0;
            size_t first = 0;
            for (size_t i = 0; i < got.size(); i++)
            {
                if (std::llround(static_cast<double>(got[i])) != expected[i])
                {
                    if (mismatches == 0)
                    {
                        first = i;
                    }
                    mismatches++;
                }
            }
            if (mismatches > 0)
            {
                fail(name, std::to_string(mismatches) + " of " +
                               std::to_string(got.size()) +
                               " values differ, first at index " +
                               std::to_string(first) + ": got " +
                               std::to_string(static_cast<double>(got[first])) +
                               ", expected " + std::to_string(expected[first]));
            }
        }

        // Compares bit for bit, e.g. with the result of the host API.
        template <typename V>
        void check_equal(const std::string& name, const std::vector<V>& got,
                         const std::vector<V>& expected)
        {
            cases_++;
            if (got.size() != expected.size())
            {
                fail(name, "size mismatch");
                return;
            }
            size_t mismatches = 0;
            size_t first = 0;
            for (size_t i = 0; i < got.size(); i++)
            {
                if (std::memcmp(&got[i], &expected[i], sizeof(V)) != 0)
                {
                    if (mismatches == 0)
                    {
                        first = i;
                    }
                    mismatches++;
                }
            }
            if (mismatches > 0)
            {
                fail(name, std::to_string(mismatches) + " of " +
                               std::to_string(got.size()) +
                               " values differ, first at index " +
                               std::to_string(first));
            }
        }

        void expect(bool condition, const std::string& name,
                    const std::string& message)
        {
            cases_++;
            if (!condition)
            {
                fail(name, message);
            }
        }

        void error(const std::string& name, const std::string& what)
        {
            cases_++;
            fail(name, what);
        }

        int finish() const
        {
            std::cout << suite_ << ": " << cases_ << " cases, " << failures_
                      << " failed" << std::endl;
            return failures_ == 0 ? 0 : 1;
        }

      private:
        void fail(const std::string& name, const std::string& message)
        {
            failures_++;
            std::cout << "FAILED " << suite_ << " " << name << ": " << message
                      << std::endl;
        }

        std::string suite_;
        int cases_ = 0;
        int failures_ = 0;
    };

} // namespace gpufft_test
#endif // GPUFFT_TEST_UTIL_H
