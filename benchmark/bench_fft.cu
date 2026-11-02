// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "bench_util.cuh"

using namespace gpufft;

namespace
{
    using DataTypes = nvbench::type_list<Float32, Float64>;

    type direction_of(nvbench::state& state)
    {
        return state.get_string("Direction") == "forward" ? FORWARD : INVERSE;
    }

    int batch_of(nvbench::state& state, int logn)
    {
        return (state.get_string("Workload") == "bulk") ? bulk_batch(logn) : 1;
    }

    template <typename T>
    fft_configuration<T> config(int logn, type dir, ReductionPolynomial poly,
                                cudaStream_t stream)
    {
        fft_configuration<T> cfg{};
        cfg.n_power = logn;
        cfg.fft_type = dir;
        cfg.reduction_poly = poly;
        cfg.mod_inverse = COMPLEX<T>(T(1), T(0));
        cfg.stream = stream;
        return cfg;
    }

    template <typename V>
    void add_memory_traffic(nvbench::state& state, size_t values, size_t roots)
    {
        state.add_element_count(values, "Values");
        state.add_global_memory_reads<V>(values + roots, "Read");
        state.add_global_memory_writes<V>(values, "Write");
    }

    // GPU_FFT in place, complex or real input (forward) / output (inverse).
    template <typename T> void fft(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = batch_of(state, logn);
        const type dir = direction_of(state);
        const ReductionPolynomial poly = (state.get_string("Poly") == "X^N-1")
                                             ? ReductionPolynomial::X_N_minus
                                             : ReductionPolynomial::X_N_plus;
        const bool real_io = (state.get_string("IO") == "real");
        const size_t root_count =
            (poly == ReductionPolynomial::X_N_minus) ? n / 2 : n;

        auto data = random_vector<COMPLEX<T>>(n * batch, 1234);
        auto real = random_vector<T>(real_io ? n * batch : 1, 5678);
        auto roots = random_vector<COMPLEX<T>>(root_count, 4321);
        add_memory_traffic<COMPLEX<T>>(state, n * batch, root_count);

        BenchStream stream(state);
        fft_configuration<T> cfg = config<T>(logn, dir, poly, stream.get());

        state.exec(
            [&](nvbench::launch&)
            {
                if (real_io)
                    GPU_FFT(ptr(real), ptr(data), ptr(roots), cfg, batch,
                            false);
                else
                    GPU_FFT(ptr(data), ptr(roots), cfg, batch, false);
            });
    }

    // GPU_Special_FFT in place (n = number of slots).
    template <typename T>
    void special_fft(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = batch_of(state, logn);

        auto data = random_vector<COMPLEX<T>>(n * batch, 1234);
        auto roots = random_vector<COMPLEX<T>>(n, 4321);
        add_memory_traffic<COMPLEX<T>>(state, n * batch, n);

        BenchStream stream(state);
        fft_configuration<T> cfg =
            config<T>(logn, direction_of(state), ReductionPolynomial::X_N_plus,
                      stream.get());

        state.exec([&](nvbench::launch&)
                   { GPU_Special_FFT(ptr(data), ptr(roots), cfg, batch); });
    }

    // GPU_FFNT of n real coefficients (an FFT of n / 2 complex values).
    template <typename T>
    void ffnt(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = batch_of(state, logn);

        auto real = random_vector<T>(n * batch, 5678);
        auto temp = random_vector<COMPLEX<T>>(n / 2 * batch, 1234);
        auto roots = random_vector<COMPLEX<T>>(n / 2, 4321);
        auto twist = random_vector<COMPLEX<T>>(n / 2, 8765);
        add_memory_traffic<COMPLEX<T>>(state, n / 2 * batch, n);

        BenchStream stream(state);
        fft_configuration<T> cfg =
            config<T>(logn, direction_of(state), ReductionPolynomial::X_N_plus,
                      stream.get());

        state.exec(
            [&](nvbench::launch&) {
                GPU_FFNT(ptr(real), ptr(temp), ptr(twist), ptr(roots), cfg,
                         batch, false);
            });
    }

    // Polynomial multiplication: forward transform of A and B, inverse
    // transform with the fused pointwise product. c2c and real: GPU_FFT of
    // 2^LogN points (X^N - 1); ffnt: GPU_FFNT of 2^LogN real coefficients.
    template <typename T>
    void poly_mult(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = batch_of(state, logn);
        const std::string method = state.get_string("Method");
        const size_t spectrum = (method == "ffnt") ? n / 2 : n;

        auto data = random_vector<COMPLEX<T>>(2 * spectrum * batch, 1234);
        auto real = random_vector<T>(method == "c2c" ? 1 : 2 * n * batch, 5678);
        auto roots = random_vector<COMPLEX<T>>(n, 4321);
        auto inverse_roots = random_vector<COMPLEX<T>>(n, 3412);
        auto twist = random_vector<COMPLEX<T>>(n / 2, 8765);
        add_memory_traffic<COMPLEX<T>>(state, 2 * spectrum * batch, 2 * n);

        BenchStream stream(state);
        const ReductionPolynomial poly = ReductionPolynomial::X_N_minus;
        fft_configuration<T> fwd = config<T>(logn, FORWARD, poly, stream.get());
        fft_configuration<T> inv = config<T>(logn, INVERSE, poly, stream.get());

        state.exec(
            [&](nvbench::launch&)
            {
                if (method == "c2c")
                {
                    GPU_FFT(ptr(data), ptr(roots), fwd, 2 * batch, false);
                    GPU_FFT(ptr(data), ptr(inverse_roots), inv, batch, true);
                }
                else if (method == "real")
                {
                    GPU_FFT(ptr(real), ptr(data), ptr(roots), fwd, 2 * batch,
                            false);
                    GPU_FFT(ptr(real), ptr(data), ptr(inverse_roots), inv,
                            batch, true);
                }
                else
                {
                    GPU_FFNT(ptr(real), ptr(data), ptr(twist), ptr(roots), fwd,
                             2 * batch, false);
                    GPU_FFNT(ptr(real), ptr(data), ptr(twist),
                             ptr(inverse_roots), inv, batch, true);
                }
            });
    }

} // namespace

NVBENCH_BENCH_TYPES(fft, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(1, 28))
    .add_string_axis("Workload", {"single", "bulk"})
    .add_string_axis("Direction", {"forward", "inverse"})
    .add_string_axis("Poly", {"X^N-1", "X^N+1"})
    .add_string_axis("IO", {"complex", "real"})
    .set_timeout(1);

NVBENCH_BENCH_TYPES(special_fft, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(1, 28))
    .add_string_axis("Workload", {"single", "bulk"})
    .add_string_axis("Direction", {"forward", "inverse"})
    .set_timeout(1);

NVBENCH_BENCH_TYPES(ffnt, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(2, 28))
    .add_string_axis("Workload", {"single", "bulk"})
    .add_string_axis("Direction", {"forward", "inverse"})
    .set_timeout(1);

NVBENCH_BENCH_TYPES(poly_mult, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(12, 24))
    .add_string_axis("Workload", {"single", "bulk"})
    .add_string_axis("Method", {"c2c", "real", "ffnt"})
    .set_timeout(1);
