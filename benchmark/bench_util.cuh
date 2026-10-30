// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef BENCH_UTIL_H
#define BENCH_UTIL_H

#include <algorithm>
#include <cstdint>

#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include "gpufft/fft.cuh"

// COMPLEX is used without a namespace so that the benchmarks also build
// against GPU-FFT versions with the complex types in the global namespace.
template <typename V> struct scalar_of
{
    using type = V;
};

template <typename T> struct scalar_of<COMPLEX<T>>
{
    using type = T;
};

template <typename T> __host__ __device__ void set_value(T& out, T re, T)
{
    out = re;
}

template <typename T>
__host__ __device__ void set_value(COMPLEX<T>& out, T re, T im)
{
    out = COMPLEX<T>(re, im);
}

// Uniform values in [-1, 1): real numbers, or complex numbers for COMPLEX<T>.
template <typename V> struct random_functor
{
    std::uint32_t seed;

    __host__ __device__ random_functor(std::uint32_t _seed) : seed(_seed) {}

    __host__ __device__ V operator()(const int n) const
    {
        using T = typename scalar_of<V>::type;
        thrust::default_random_engine rng(seed);
        rng.discard(2 * n);
        thrust::uniform_real_distribution<T> dist(T(-1), T(1));
        T re = dist(rng);
        T im = dist(rng);
        V out;
        set_value(out, re, im);
        return out;
    }
};

template <typename V>
thrust::device_vector<V> random_vector(size_t count, std::uint32_t seed)
{
    thrust::device_vector<V> values(count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(count), values.begin(),
                      random_functor<V>(seed));
    return values;
}

template <typename V> V* ptr(thrust::device_vector<V>& values)
{
    return thrust::raw_pointer_cast(values.data());
}

class BenchStream
{
  public:
    explicit BenchStream(nvbench::state& state)
    {
        cudaStreamCreate(&stream_);
        state.set_cuda_stream(nvbench::make_cuda_stream_view(stream_));
    }

    ~BenchStream()
    {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_;
};

// Around 2^22 coefficients for the "bulk" workload.
inline int bulk_batch(int logn)
{
    return std::max(1, (1 << 22) >> logn);
}

#endif // BENCH_UTIL_H
