// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Correctness tests of the GPU-FFT host API against double precision CPU
// references.
//
// Usage: fft_tests --bits <32|64> --suite <name|all> --max-logn <N>
// Suites: c2c, real, mult, poly_mult, special, ffnt, validation, large

#include <cstring>
#include <functional>

#include "test_util.cuh"

using namespace gpufft_test;

namespace
{
    const ReductionPolynomial kPolys[] = {ReductionPolynomial::X_N_minus,
                                          ReductionPolynomial::X_N_plus};
    const type kDirections[] = {FORWARD, INVERSE};

    // Smallest and largest log2 sizes of the host API.
    const int kMinLogN = 1;
    const int kMinSpecialLogN = 11;
    const int kMinFFNTLogN = 12;
    const int kMaxLogN = 24;

    // Sizes up to 2^9 transform 2^(9 - logn) polynomials per block.
    const int kLowRingMaxLogN = 9;

    std::string case_name(const std::string& variant, type dir, int logn,
                          ReductionPolynomial poly, int batch)
    {
        return variant + (dir == FORWARD ? " fft" : " ifft") +
               " logn=" + std::to_string(logn) +
               (poly == ReductionPolynomial::X_N_minus ? " X^N-1" : " X^N+1") +
               " batch=" + std::to_string(batch);
    }

    std::string case_name(const std::string& variant, type dir, int logn,
                          int batch)
    {
        return variant + (dir == FORWARD ? " fft" : " ifft") +
               " logn=" + std::to_string(logn) +
               " batch=" + std::to_string(batch);
    }

    // About 2^20 coefficients per case. 37 polynomials cover several grid
    // rows of the batch dimension.
    int batch_for(int logn)
    {
        return std::min(37, std::max(1, (1 << 20) >> logn));
    }

    // More than two blocks of the low ring kernel, the last one not full.
    int low_ring_batch(int logn)
    {
        return 2 * (1 << (kLowRingMaxLogN - logn)) + 3;
    }

    void sync()
    {
        GPUFFT_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<Ref> table_for(int logn, ReductionPolynomial poly, bool inverse)
    {
        return (poly == ReductionPolynomial::X_N_minus)
                   ? cyclic_table(logn, inverse)
                   : negacyclic_table(logn, inverse);
    }

    template <typename T> class FFTTests
    {
      public:
        explicit FFTTests(int max_logn)
            : max_logn_(std::min(max_logn, kMaxLogN)), rng_(2026)
        {
        }

        int run(const std::string& suite)
        {
            if (suite == "c2c")
                return c2c();
            if (suite == "real")
                return real();
            if (suite == "mult")
                return mult();
            if (suite == "poly_mult")
                return poly_mult();
            if (suite == "special")
                return special();
            if (suite == "ffnt")
                return ffnt();
            if (suite == "validation")
                return validation();
            if (suite == "large")
                return large();
            if (suite == "all")
            {
                int result = 0;
                for (const char* s : {"c2c", "real", "mult", "poly_mult",
                                      "special", "ffnt", "validation", "large"})
                {
                    result |= run(s);
                }
                return result;
            }
            throw std::invalid_argument("Unknown suite: " + suite);
        }

      private:
        static constexpr bool kSingle = std::is_same<T, Float32>::value;
        // Relative error bounds, measured errors are about 10x smaller.
        static constexpr double kTolerance = kSingle ? 2e-6 : 1e-13;

        std::string suite_name(const std::string& suite) const
        {
            return std::string(kSingle ? "Float32" : "Float64") + "." + suite;
        }

        fft_configuration<T> config(int logn, type dir,
                                    ReductionPolynomial poly, double n_inverse)
        {
            fft_configuration<T> cfg{};
            cfg.n_power = logn;
            cfg.fft_type = dir;
            cfg.reduction_poly = poly;
            cfg.mod_inverse = COMPLEX<T>(static_cast<T>(n_inverse), T(0));
            cfg.stream = 0;
            return cfg;
        }

        // Runs fn and reports exceptions as failures of the case.
        void guarded(TestReport& report, const std::string& name,
                     const std::function<void()>& fn)
        {
            try
            {
                fn();
            }
            catch (const std::exception& e)
            {
                report.error(name, e.what());
            }
        }

        // ---------------- GPU_FFT, complex in/out ----------------

        void c2c_case(TestReport& report, int logn, type dir,
                      ReductionPolynomial poly, int batch)
        {
            const std::string name = case_name("c2c", dir, logn, poly, batch);
            guarded(report, name,
                    [&]
                    {
                        const size_t n = size_t(1) << logn;
                        const double n_inverse = 1.0 / n;
                        std::vector<Ref> table =
                            table_for(logn, poly, dir == INVERSE);
                        std::vector<COMPLEX<T>> input =
                            to_t<T>(random_complex(rng_, n * batch));

                        DeviceBuffer<COMPLEX<T>> data(input);
                        DeviceBuffer<COMPLEX<T>> roots(to_t<T>(table));
                        GPU_FFT(data.get(), roots.get(),
                                config(logn, dir, poly, n_inverse), batch,
                                false);
                        sync();
                        std::vector<COMPLEX<T>> got = data.download();

                        for (int b = 0; b < batch; b++)
                        {
                            std::vector<Ref> expected =
                                to_ref(slice(input, b * n, n));
                            if (dir == FORWARD)
                                reference_forward(expected, logn, table, poly);
                            else
                                reference_inverse(expected, logn, table, poly,
                                                  Ref(n_inverse, 0.0));
                            report.check_close(
                                name + " poly=" + std::to_string(b),
                                slice(got, b * n, n), expected, kTolerance);
                        }
                    });
        }

        int c2c()
        {
            TestReport report(suite_name("c2c"));
            for (int logn = kMinLogN; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        c2c_case(report, logn, dir, poly, batch_for(logn));
                        if (logn <= kLowRingMaxLogN)
                            c2c_case(report, logn, dir, poly,
                                     low_ring_batch(logn));
                        // The low ring kernels take batches above the grid
                        // limit of the stage kernels.
                        if (logn == kMinLogN)
                            c2c_case(report, logn, dir, poly, 65537);
                    }
            return report.finish();
        }

        // ---------------- GPU_FFT, real in (forward) / out (inverse) -------

        void real_case(TestReport& report, int logn, type dir,
                       ReductionPolynomial poly, int batch)
        {
            const std::string name = case_name("real", dir, logn, poly, batch);
            guarded(report, name,
                    [&]
                    {
                        const size_t n = size_t(1) << logn;
                        const double n_inverse = 1.0 / n;
                        std::vector<Ref> table =
                            table_for(logn, poly, dir == INVERSE);
                        DeviceBuffer<COMPLEX<T>> roots(to_t<T>(table));

                        if (dir == FORWARD)
                        {
                            std::vector<T> input =
                                to_t<T>(random_real(rng_, n * batch));
                            DeviceBuffer<T> real(input);
                            DeviceBuffer<COMPLEX<T>> out(n * batch);
                            GPU_FFT(real.get(), out.get(), roots.get(),
                                    config(logn, dir, poly, n_inverse), batch,
                                    false);
                            sync();
                            std::vector<COMPLEX<T>> got = out.download();
                            for (int b = 0; b < batch; b++)
                            {
                                std::vector<double> values =
                                    to_ref(slice(input, b * n, n));
                                std::vector<Ref> expected(n);
                                for (size_t i = 0; i < n; i++)
                                    expected[i] = Ref(values[i], 0.0);
                                reference_forward(expected, logn, table, poly);
                                report.check_close(
                                    name + " poly=" + std::to_string(b),
                                    slice(got, b * n, n), expected, kTolerance);
                            }
                        }
                        else
                        {
                            std::vector<COMPLEX<T>> input =
                                to_t<T>(random_complex(rng_, n * batch));
                            DeviceBuffer<COMPLEX<T>> in(input);
                            DeviceBuffer<T> real(n * batch);
                            GPU_FFT(real.get(), in.get(), roots.get(),
                                    config(logn, dir, poly, n_inverse), batch,
                                    false);
                            sync();
                            std::vector<T> got = real.download();
                            for (int b = 0; b < batch; b++)
                            {
                                std::vector<Ref> result =
                                    to_ref(slice(input, b * n, n));
                                reference_inverse(result, logn, table, poly,
                                                  Ref(n_inverse, 0.0));
                                std::vector<double> expected(n);
                                for (size_t i = 0; i < n; i++)
                                    expected[i] = result[i].real();
                                report.check_close(
                                    name + " poly=" + std::to_string(b),
                                    slice(got, b * n, n), expected, kTolerance);
                            }
                        }
                    });
        }

        int real()
        {
            TestReport report(suite_name("real"));
            for (int logn = kMinLogN; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        real_case(report, logn, dir, poly, batch_for(logn));
                        if (logn <= kLowRingMaxLogN)
                            real_case(report, logn, dir, poly,
                                      low_ring_batch(logn));
                    }
            return report.finish();
        }

        // ---------------- Pointwise product fused into the inverse --------

        void mult_case(TestReport& report, int logn, ReductionPolynomial poly,
                       bool real_output, int batch)
        {
            const std::string name =
                case_name(real_output ? "mult real" : "mult c2c", INVERSE, logn,
                          poly, batch);
            guarded(report, name,
                    [&]
                    {
                        const size_t n = size_t(1) << logn;
                        const double n_inverse = 1.0 / n;
                        std::vector<Ref> table = table_for(logn, poly, true);
                        // A (batch polynomials) followed by B
                        std::vector<COMPLEX<T>> input =
                            to_t<T>(random_complex(rng_, 2 * n * batch));
                        DeviceBuffer<COMPLEX<T>> data(input);
                        DeviceBuffer<COMPLEX<T>> roots(to_t<T>(table));
                        DeviceBuffer<T> real(n * batch);
                        fft_configuration<T> cfg =
                            config(logn, INVERSE, poly, n_inverse);
                        if (real_output)
                            GPU_FFT(real.get(), data.get(), roots.get(), cfg,
                                    batch, true);
                        else
                            GPU_FFT(data.get(), roots.get(), cfg, batch, true);
                        sync();

                        std::vector<COMPLEX<T>> got_complex = data.download();
                        std::vector<T> got_real = real.download();
                        for (int b = 0; b < batch; b++)
                        {
                            std::vector<Ref> a = to_ref(slice(input, b * n, n));
                            std::vector<Ref> bb =
                                to_ref(slice(input, (batch + b) * n, n));
                            for (size_t i = 0; i < n; i++)
                                a[i] = a[i] * bb[i];
                            reference_inverse(a, logn, table, poly,
                                              Ref(n_inverse, 0.0));
                            const std::string poly_name =
                                name + " poly=" + std::to_string(b);
                            if (real_output)
                            {
                                std::vector<double> expected(n);
                                for (size_t i = 0; i < n; i++)
                                    expected[i] = a[i].real();
                                report.check_close(poly_name,
                                                   slice(got_real, b * n, n),
                                                   expected, kTolerance);
                            }
                            else
                            {
                                report.check_close(poly_name,
                                                   slice(got_complex, b * n, n),
                                                   a, kTolerance);
                            }
                        }
                    });
        }

        int mult()
        {
            TestReport report(suite_name("mult"));
            for (int logn = kMinLogN; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (bool real_output : {false, true})
                    {
                        mult_case(report, logn, poly, real_output,
                                  batch_for(logn));
                        if (logn <= kLowRingMaxLogN)
                            mult_case(report, logn, poly, real_output,
                                      low_ring_batch(logn));
                    }
            return report.finish();
        }

        // ---------------- Exact polynomial products ----------------

        // Coefficients small enough for exact results after rounding.
        std::vector<int> small_poly(size_t n, size_t padded)
        {
            std::vector<int> values =
                random_ints(rng_, n, kSingle ? -1 : -7, kSingle ? 1 : 7);
            values.resize(padded, 0);
            return values;
        }

        // GPU_FFT forward on A and B, inverse with the fused product.
        // Cyclic: n / 2 coefficients zero padded to n (linear product).
        // Negacyclic: n coefficients, product mod X^n + 1.
        void poly_mult_fft_case(TestReport& report, int logn,
                                ReductionPolynomial poly, bool real_io,
                                int batch)
        {
            const std::string name =
                case_name(real_io ? "poly_mult real" : "poly_mult c2c", INVERSE,
                          logn, poly, batch);
            guarded(
                report, name,
                [&]
                {
                    const size_t n = size_t(1) << logn;
                    const bool negacyclic =
                        (poly == ReductionPolynomial::X_N_plus);
                    const size_t coefficients = negacyclic ? n : n / 2;

                    std::vector<std::vector<int>> polys(2 * batch);
                    std::vector<COMPLEX<T>> complex_input(2 * n * batch);
                    std::vector<T> real_input(2 * n * batch);
                    for (int p = 0; p < 2 * batch; p++)
                    {
                        polys[p] = small_poly(coefficients, n);
                        for (size_t i = 0; i < n; i++)
                        {
                            complex_input[p * n + i] =
                                COMPLEX<T>(static_cast<T>(polys[p][i]), T(0));
                            real_input[p * n + i] = static_cast<T>(polys[p][i]);
                        }
                    }

                    DeviceBuffer<COMPLEX<T>> forward_roots(
                        to_t<T>(table_for(logn, poly, false)));
                    DeviceBuffer<COMPLEX<T>> inverse_roots(
                        to_t<T>(table_for(logn, poly, true)));
                    DeviceBuffer<COMPLEX<T>> data(complex_input);
                    DeviceBuffer<T> real(real_input);
                    fft_configuration<T> fwd =
                        config(logn, FORWARD, poly, 1.0 / n);
                    fft_configuration<T> inv =
                        config(logn, INVERSE, poly, 1.0 / n);
                    std::vector<double> got(n * batch);
                    if (real_io)
                    {
                        GPU_FFT(real.get(), data.get(), forward_roots.get(),
                                fwd, 2 * batch, false);
                        GPU_FFT(real.get(), data.get(), inverse_roots.get(),
                                inv, batch, true);
                        sync();
                        std::vector<T> out = real.download();
                        for (size_t i = 0; i < got.size(); i++)
                            got[i] = out[i];
                    }
                    else
                    {
                        GPU_FFT(data.get(), forward_roots.get(), fwd, 2 * batch,
                                false);
                        GPU_FFT(data.get(), inverse_roots.get(), inv, batch,
                                true);
                        sync();
                        std::vector<COMPLEX<T>> out = data.download();
                        for (size_t i = 0; i < got.size(); i++)
                            got[i] = out[i].real();
                    }

                    for (int b = 0; b < batch; b++)
                    {
                        report.check_rounded(
                            name + " poly=" + std::to_string(b),
                            slice(got, b * n, n),
                            schoolbook(polys[b], polys[batch + b], negacyclic));
                    }
                });
        }

        void poly_mult_ffnt_case(TestReport& report, int logn, int batch)
        {
            const std::string name =
                case_name("poly_mult ffnt", INVERSE, logn, batch);
            guarded(report, name,
                    [&]
                    {
                        const size_t n = size_t(1) << logn;
                        FFNT<T> generator(n);
                        std::vector<std::vector<int>> polys(2 * batch);
                        std::vector<T> input(2 * n * batch);
                        for (int p = 0; p < 2 * batch; p++)
                        {
                            polys[p] = small_poly(n, n);
                            for (size_t i = 0; i < n; i++)
                                input[p * n + i] = static_cast<T>(polys[p][i]);
                        }

                        DeviceBuffer<T> data(input);
                        DeviceBuffer<COMPLEX<T>> temp(n * batch);
                        DeviceBuffer<COMPLEX<T>> roots(
                            generator.ReverseRootTable_ffnt());
                        DeviceBuffer<COMPLEX<T>> inverse_roots(
                            generator.InverseReverseRootTable_ffnt());
                        DeviceBuffer<COMPLEX<T>> twist(
                            generator.twist_table_ffnt());
                        DeviceBuffer<COMPLEX<T>> untwist(
                            generator.untwist_table_ffnt());

                        GPU_FFNT(data.get(), temp.get(), twist.get(),
                                 roots.get(),
                                 config(logn, FORWARD,
                                        ReductionPolynomial::X_N_plus, 0.0),
                                 2 * batch, false);
                        GPU_FFNT(data.get(), temp.get(), untwist.get(),
                                 inverse_roots.get(),
                                 config(logn, INVERSE,
                                        ReductionPolynomial::X_N_plus,
                                        1.0 / (n / 2)),
                                 batch, true);
                        sync();
                        std::vector<T> got = data.download();
                        for (int b = 0; b < batch; b++)
                        {
                            report.check_rounded(
                                name + " poly=" + std::to_string(b),
                                slice(got, b * n, n),
                                schoolbook(polys[b], polys[batch + b], true));
                        }
                    });
        }

        int poly_mult()
        {
            TestReport report(suite_name("poly_mult"));
            // The schoolbook reference is quadratic.
            for (int logn = kMinLogN; logn <= std::min(13, max_logn_); logn++)
            {
                for (ReductionPolynomial poly : kPolys)
                    for (bool real_io : {false, true})
                        poly_mult_fft_case(report, logn, poly, real_io, 3);
                if (logn >= kMinFFNTLogN)
                    poly_mult_ffnt_case(report, logn, 3);
            }
            return report.finish();
        }

        // ---------------- GPU_Special_FFT ----------------

        void special_case(TestReport& report, int logl, type dir, int batch)
        {
            const std::string name = case_name("special", dir, logl, batch);
            guarded(
                report, name,
                [&]
                {
                    const size_t l = size_t(1) << logl;
                    const double n_inverse = 1.0 / l;
                    std::vector<Ref> table =
                        special_table(logl, dir == INVERSE);
                    std::vector<COMPLEX<T>> input =
                        to_t<T>(random_complex(rng_, l * batch));
                    DeviceBuffer<COMPLEX<T>> data(input);
                    DeviceBuffer<COMPLEX<T>> roots(to_t<T>(table));
                    GPU_Special_FFT(data.get(), roots.get(),
                                    config(logl, dir,
                                           ReductionPolynomial::X_N_plus,
                                           n_inverse),
                                    batch);
                    sync();
                    std::vector<COMPLEX<T>> got = data.download();
                    for (int b = 0; b < batch; b++)
                    {
                        std::vector<Ref> expected =
                            to_ref(slice(input, b * l, l));
                        if (dir == FORWARD)
                            reference_special_forward(expected, logl, table);
                        else
                            reference_special_inverse(expected, logl, table,
                                                      Ref(n_inverse, 0.0));
                        report.check_close(name + " poly=" + std::to_string(b),
                                           slice(got, b * l, l), expected,
                                           kTolerance);
                    }
                });
        }

        // Inverse after forward with the conjugate table is the identity.
        void special_round_trip(TestReport& report, int logl, int batch)
        {
            const std::string name =
                "special round trip logl=" + std::to_string(logl) +
                " batch=" + std::to_string(batch);
            guarded(report, name,
                    [&]
                    {
                        const size_t l = size_t(1) << logl;
                        std::vector<COMPLEX<T>> input =
                            to_t<T>(random_complex(rng_, l * batch));
                        DeviceBuffer<COMPLEX<T>> data(input);
                        DeviceBuffer<COMPLEX<T>> roots(
                            to_t<T>(special_table(logl, false)));
                        DeviceBuffer<COMPLEX<T>> inverse_roots(
                            to_t<T>(special_table(logl, true)));
                        GPU_Special_FFT(data.get(), roots.get(),
                                        config(logl, FORWARD,
                                               ReductionPolynomial::X_N_plus,
                                               0.0),
                                        batch);
                        GPU_Special_FFT(data.get(), inverse_roots.get(),
                                        config(logl, INVERSE,
                                               ReductionPolynomial::X_N_plus,
                                               1.0 / l),
                                        batch);
                        sync();
                        report.check_close(name, data.download(), to_ref(input),
                                           kTolerance);
                    });
        }

        int special()
        {
            TestReport report(suite_name("special"));
            for (int logl = kMinSpecialLogN; logl <= max_logn_; logl++)
            {
                for (type dir : kDirections)
                    special_case(report, logl, dir, batch_for(logl));
                special_round_trip(report, logl, batch_for(logl));
            }
            return report.finish();
        }

        // ---------------- GPU_FFNT ----------------

        // Reference of the forward FFNT: fold, twist, FFT of n / 2 points.
        static std::vector<Ref> ffnt_forward(const std::vector<double>& real,
                                             int logn, FFNT<double>& generator)
        {
            const size_t half = real.size() / 2;
            std::vector<Ref> twist = generator.twist_table_ffnt();
            std::vector<Ref> folded(half);
            for (size_t j = 0; j < half; j++)
                folded[j] = Ref(real[j], real[j + half]) * twist[j];
            reference_forward(folded, logn - 1,
                              generator.ReverseRootTable_ffnt(),
                              ReductionPolynomial::X_N_minus);
            return folded;
        }

        // Reference of the inverse FFNT: IFFT, untwist, unfold.
        static std::vector<double> ffnt_inverse(std::vector<Ref> spectrum,
                                                int logn,
                                                FFNT<double>& generator)
        {
            const size_t half = spectrum.size();
            reference_inverse(
                spectrum, logn - 1, generator.InverseReverseRootTable_ffnt(),
                ReductionPolynomial::X_N_minus, Ref(1.0 / half, 0.0));
            std::vector<Ref> untwist = generator.untwist_table_ffnt();
            std::vector<double> real(2 * half);
            for (size_t j = 0; j < half; j++)
            {
                Ref value = spectrum[j] * untwist[j];
                real[j] = value.real();
                real[j + half] = value.imag();
            }
            return real;
        }

        void ffnt_case(TestReport& report, int logn, type dir, bool mult,
                       int batch)
        {
            const std::string name =
                case_name(mult ? "ffnt mult" : "ffnt", dir, logn, batch);
            guarded(report, name,
                    [&]
                    {
                        const size_t n = size_t(1) << logn;
                        const size_t half = n / 2;
                        FFNT<double> reference(n);
                        FFNT<T> generator(n);

                        if (dir == FORWARD)
                        {
                            std::vector<T> input =
                                to_t<T>(random_real(rng_, n * batch));
                            DeviceBuffer<T> data(input);
                            DeviceBuffer<COMPLEX<T>> temp(half * batch);
                            DeviceBuffer<COMPLEX<T>> roots(
                                generator.ReverseRootTable_ffnt());
                            DeviceBuffer<COMPLEX<T>> twist(
                                generator.twist_table_ffnt());
                            GPU_FFNT(data.get(), temp.get(), twist.get(),
                                     roots.get(),
                                     config(logn, FORWARD,
                                            ReductionPolynomial::X_N_plus, 0.0),
                                     batch, false);
                            sync();
                            std::vector<COMPLEX<T>> got = temp.download();
                            for (int b = 0; b < batch; b++)
                            {
                                report.check_close(
                                    name + " poly=" + std::to_string(b),
                                    slice(got, b * half, half),
                                    ffnt_forward(to_ref(slice(input, b * n, n)),
                                                 logn, reference),
                                    kTolerance);
                            }
                        }
                        else
                        {
                            // A (batch spectra) followed by B for the product
                            std::vector<COMPLEX<T>> input =
                                to_t<T>(random_complex(rng_, (mult ? 2 : 1) *
                                                                 half * batch));
                            DeviceBuffer<COMPLEX<T>> temp(input);
                            DeviceBuffer<T> data(n * batch);
                            DeviceBuffer<COMPLEX<T>> roots(
                                generator.InverseReverseRootTable_ffnt());
                            DeviceBuffer<COMPLEX<T>> untwist(
                                generator.untwist_table_ffnt());
                            GPU_FFNT(data.get(), temp.get(), untwist.get(),
                                     roots.get(),
                                     config(logn, INVERSE,
                                            ReductionPolynomial::X_N_plus,
                                            1.0 / half),
                                     batch, mult);
                            sync();
                            std::vector<T> got = data.download();
                            for (int b = 0; b < batch; b++)
                            {
                                std::vector<Ref> spectrum =
                                    to_ref(slice(input, b * half, half));
                                if (mult)
                                {
                                    std::vector<Ref> other = to_ref(
                                        slice(input, (batch + b) * half, half));
                                    for (size_t i = 0; i < half; i++)
                                        spectrum[i] = spectrum[i] * other[i];
                                }
                                report.check_close(
                                    name + " poly=" + std::to_string(b),
                                    slice(got, b * n, n),
                                    ffnt_inverse(spectrum, logn, reference),
                                    kTolerance);
                            }
                        }
                    });
        }

        int ffnt()
        {
            TestReport report(suite_name("ffnt"));
            for (int logn = kMinFFNTLogN; logn <= max_logn_; logn++)
            {
                ffnt_case(report, logn, FORWARD, false, batch_for(logn));
                ffnt_case(report, logn, INVERSE, false, batch_for(logn));
                ffnt_case(report, logn, INVERSE, true, batch_for(logn));
            }
            return report.finish();
        }

        // ---------------- Input checks ----------------

        void expect_invalid(TestReport& report, const std::string& name,
                            const std::function<void()>& fn)
        {
            try
            {
                fn();
                report.expect(false, name, "no std::invalid_argument");
            }
            catch (const std::invalid_argument&)
            {
                report.expect(true, name, "");
            }
            catch (const std::exception& e)
            {
                report.error(name, std::string("unexpected: ") + e.what());
            }
        }

        int validation()
        {
            TestReport report(suite_name("validation"));
            COMPLEX<T>* c = nullptr;
            T* r = nullptr;
            auto cfg = [&](int logn) {
                return config(logn, FORWARD, ReductionPolynomial::X_N_minus,
                              1.0);
            };

            for (int logn : {kMinLogN - 1, kMaxLogN + 1})
            {
                const std::string n = " logn=" + std::to_string(logn);
                expect_invalid(report, "c2c" + n,
                               [&] { GPU_FFT(c, c, cfg(logn), 1, false); });
                expect_invalid(report, "real" + n,
                               [&] { GPU_FFT(r, c, c, cfg(logn), 1, false); });
            }
            for (int logn : {kMinFFNTLogN - 1, kMaxLogN + 1})
            {
                expect_invalid(report, "ffnt logn=" + std::to_string(logn),
                               [&]
                               { GPU_FFNT(r, c, c, c, cfg(logn), 1, false); });
            }
            for (int logn : {kMinSpecialLogN - 1, kMaxLogN + 1})
            {
                expect_invalid(report, "special logn=" + std::to_string(logn),
                               [&] { GPU_Special_FFT(c, c, cfg(logn), 1); });
            }

            // Non-positive, above the grid limit, and batch * N >= 2^31.
            const int batches[][2] = {{kMinFFNTLogN, 0},
                                      {kMinFFNTLogN, -1},
                                      {kMinFFNTLogN, 65536},
                                      {24, 128}};
            for (const auto& entry : batches)
            {
                const int logn = entry[0];
                const int batch = entry[1];
                const std::string n = " logn=" + std::to_string(logn) +
                                      " batch=" + std::to_string(batch);
                expect_invalid(report, "c2c" + n,
                               [&] { GPU_FFT(c, c, cfg(logn), batch, false); });
                expect_invalid(report, "real" + n,
                               [&]
                               { GPU_FFT(r, c, c, cfg(logn), batch, false); });
                expect_invalid(report, "special" + n,
                               [&]
                               { GPU_Special_FFT(c, c, cfg(logn), batch); });
                expect_invalid(
                    report, "ffnt" + n,
                    [&] { GPU_FFNT(r, c, c, c, cfg(logn), batch, false); });
            }
            return report.finish();
        }

        // ---------------- Sizes above --max-logn ----------------

        int large()
        {
            TestReport report(suite_name("large"));
            for (int logn = max_logn_ + 1; logn <= kMaxLogN; logn++)
            {
                for (ReductionPolynomial poly : kPolys)
                {
                    for (type dir : kDirections)
                    {
                        c2c_case(report, logn, dir, poly, 1);
                        real_case(report, logn, dir, poly, 1);
                    }
                    mult_case(report, logn, poly, false, 1);
                }
                for (type dir : kDirections)
                    special_case(report, logn, dir, 1);
                ffnt_case(report, logn, FORWARD, false, 1);
                ffnt_case(report, logn, INVERSE, true, 1);
            }
            return report.finish();
        }

        int max_logn_;
        std::mt19937_64 rng_;
    };

} // namespace

int main(int argc, char* argv[])
{
    int bits = 64;
    std::string suite = "all";
    int max_logn = 20;

    for (int i = 1; i + 1 < argc; i += 2)
    {
        if (std::strcmp(argv[i], "--bits") == 0)
            bits = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--suite") == 0)
            suite = argv[i + 1];
        else if (std::strcmp(argv[i], "--max-logn") == 0)
            max_logn = std::atoi(argv[i + 1]);
        else
        {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 2;
        }
    }

    try
    {
        if (bits == 32)
            return FFTTests<Float32>(max_logn).run(suite);
        if (bits == 64)
            return FFTTests<Float64>(max_logn).run(suite);
        std::cerr << "--bits must be 32 or 64" << std::endl;
        return 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
