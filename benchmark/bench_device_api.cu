// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Polynomial multiplication modulo X^N + 1 with one operand already
// transformed: a <- IFFT(FFT(a) * b_hat) for about 2^22 values. "fused" runs
// it in one kernel with the device API; "separate" runs GPU_FFT forward and
// GPU_FFT inverse with the fused product (the host API).

#include "bench_util.cuh"
#include "gpufft/descriptor.cuh"

#include <utility>

using namespace gpufft;

namespace
{
    using DataTypes = nvbench::type_list<Float32, Float64>;

    // Sizes that run in a single kernel.
    constexpr int kMaxFusedLogN = 10;

    template <int LogN, typename T, type D>
    using Description =
        decltype(Size<(1 << LogN)>() + Direction<D>() + Precision<T>() +
                 ReductionPoly<X_N_plus>() + Block());

    template <int LogN, typename T>
    __global__ void
    fused_multiply_kernel(COMPLEX<T>* a, const COMPLEX<T>* b_hat,
                          const COMPLEX<T>* forward_roots,
                          const COMPLEX<T>* inverse_roots, int batch)
    {
        using Fwd = Description<LogN, T, FORWARD>;
        using Inv = Description<LogN, T, INVERSE>;
        COMPLEX<T>* shared_memory = Fwd::shared_memory();
        COMPLEX<T>* polys = Fwd::block_data(a);
        const COMPLEX<T>* other = Fwd::block_data(b_hat);
        const int num_polys = Fwd::num_polys(batch);

        Fwd().load(shared_memory, polys, num_polys);
        Fwd().execute(shared_memory, forward_roots);
        Fwd().for_each(
            shared_memory,
            [&](COMPLEX<T>& value, size_t index)
            { value = value * other[index]; },
            num_polys);
        Inv().execute(shared_memory, inverse_roots);
        Inv().store(shared_memory, polys, Inv::n_inv(), num_polys);
    }

    template <int LogN, typename T>
    void launch_fused(COMPLEX<T>* a, const COMPLEX<T>* b_hat,
                      const COMPLEX<T>* forward_roots,
                      const COMPLEX<T>* inverse_roots, int batch,
                      cudaStream_t stream)
    {
        using Fwd = Description<LogN, T, FORWARD>;
        fused_multiply_kernel<LogN, T>
            <<<Fwd::grid(batch), Fwd::block_dim, Fwd::shared_memory_size,
               stream>>>(a, b_hat, forward_roots, inverse_roots, batch);
    }

    template <typename T, int... LogN>
    void launch_fused(int logn, std::integer_sequence<int, LogN...>,
                      COMPLEX<T>* a, const COMPLEX<T>* b_hat,
                      const COMPLEX<T>* forward_roots,
                      const COMPLEX<T>* inverse_roots, int batch,
                      cudaStream_t stream)
    {
        ((logn == LogN + 1
              ? launch_fused<LogN + 1>(a, b_hat, forward_roots, inverse_roots,
                                       batch, stream)
              : void()),
         ...);
    }

    template <typename T>
    void poly_mul(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = bulk_batch(logn);
        const bool fused = state.get_string("Method") == "fused";

        // a (batch polynomials) followed by b_hat, as GPU_FFT with the fused
        // product reads them.
        auto data = random_vector<COMPLEX<T>>(2 * n * batch, 1234);
        auto forward_roots = random_vector<COMPLEX<T>>(n, 4321);
        auto inverse_roots = random_vector<COMPLEX<T>>(n, 8765);
        state.add_element_count(n * batch, "Values");

        BenchStream stream(state);
        fft_configuration<T> cfg{};
        cfg.n_power = logn;
        cfg.fft_type = FORWARD;
        cfg.reduction_poly = X_N_plus;
        cfg.mod_inverse = COMPLEX<T>(T(1) / T(n), T(0));
        cfg.stream = stream.get();
        fft_configuration<T> inverse_cfg = cfg;
        inverse_cfg.fft_type = INVERSE;

        COMPLEX<T>* a = ptr(data);
        COMPLEX<T>* b_hat = a + n * batch;
        state.exec(
            [&](nvbench::launch&)
            {
                if (fused)
                {
                    launch_fused(
                        logn, std::make_integer_sequence<int, kMaxFusedLogN>(),
                        a, b_hat, ptr(forward_roots), ptr(inverse_roots), batch,
                        stream.get());
                }
                else
                {
                    GPU_FFT(a, ptr(forward_roots), cfg, batch, false);
                    GPU_FFT(a, ptr(inverse_roots), inverse_cfg, batch, true);
                }
            });
    }

} // namespace

NVBENCH_BENCH_TYPES(poly_mul, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(1, kMaxFusedLogN))
    .add_string_axis("Method", {"fused", "separate"})
    .set_timeout(1);
