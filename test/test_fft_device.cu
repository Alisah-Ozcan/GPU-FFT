// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests of the device API (gpufft/descriptor.cuh). The device API runs the
// butterflies of the host API in the same order, so its results are
// compared bit for bit with GPU_FFT; the root tables are compared with the
// tables of the host API tests.

#include <utility>

#include "gpufft/descriptor.cuh"
#include "test_util.cuh"

using namespace gpufft_test;

namespace
{
    template <typename T, int LogN, ReductionPolynomial R, type D>
    using Desc = decltype(Size<(1 << LogN)>() + Direction<D>() +
                          Precision<T>() + ReductionPoly<R>() + Block());

    // Base + BlockDim<Threads>, or Base for Threads == 0.
    template <typename Base, int Threads> struct with_threads
    {
        using type = decltype(Base() + BlockDim<Threads>());
    };

    template <typename Base> struct with_threads<Base, 0>
    {
        using type = Base;
    };

    // Base + BatchesPerBlock<Batches>, or Base for Batches == 0.
    template <typename Base, int Batches> struct with_batches
    {
        using type = decltype(Base() + BatchesPerBlock<Batches>());
    };

    template <typename Base> struct with_batches<Base, 0>
    {
        using type = Base;
    };

    template <typename T, int LogN, ReductionPolynomial R, type D,
              int Threads = 0, int Batches = 0>
    using DescWith = typename with_batches<
        typename with_threads<Desc<T, LogN, R, D>, Threads>::type,
        Batches>::type;

    template <typename T>
    fft_configuration<T> host_config(int logn, type dir,
                                     ReductionPolynomial poly)
    {
        fft_configuration<T> cfg{};
        cfg.n_power = logn;
        cfg.fft_type = dir;
        cfg.reduction_poly = poly;
        cfg.mod_inverse = COMPLEX<T>(T(1) / T(1 << logn), T(0));
        cfg.stream = 0;
        return cfg;
    }

    void sync()
    {
        GPUFFT_CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename T> std::string type_name()
    {
        return std::is_same<T, Float32>::value ? "Float32" : "Float64";
    }

    std::string case_name(const std::string& kind, int logn,
                          ReductionPolynomial poly, int threads, int batch)
    {
        return kind + " logn=" + std::to_string(logn) +
               (poly == X_N_minus ? " X^N-1" : " X^N+1") +
               (threads ? " threads=" + std::to_string(threads) : "") +
               " batch=" + std::to_string(batch);
    }

    std::string dir_name(type dir)
    {
        return dir == FORWARD ? " fft" : " ifft";
    }

    // More than two blocks of the low ring kernel, the last one not full.
    int batch_for(int logn)
    {
        return (logn <= kLowRingMaxLogN)
                   ? 2 * (1 << (kLowRingMaxLogN - logn)) + 3
                   : (logn <= 20 ? 3 : 1);
    }

    // Runs fn and reports exceptions as failures of the case.
    template <typename F>
    void guarded(TestReport& report, const std::string& name, F fn)
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

    // ---------------- Kernels ----------------

    // Polynomial product in one kernel: forward FFT of a, product with
    // b_hat = FFT(b), inverse FFT.
    template <typename Fwd, typename Inv, typename T>
    __global__ void
    fused_multiply_kernel(COMPLEX<T>* a, const COMPLEX<T>* b_hat,
                          const COMPLEX<T>* roots,
                          const COMPLEX<T>* inverse_roots, int batch)
    {
        COMPLEX<T>* shared_memory = Fwd::shared_memory();
        COMPLEX<T>* polys = Fwd::block_data(a);
        const COMPLEX<T>* other = Fwd::block_data(b_hat);
        const int num_polys = Fwd::num_polys(batch);

        Fwd().load(shared_memory, polys, num_polys);
        Fwd().execute(shared_memory, roots);
        Fwd().for_each(
            shared_memory,
            [&](COMPLEX<T>& value, size_t index)
            { value = value * other[index]; },
            num_polys);
        Inv().execute(shared_memory, inverse_roots);
        Inv().store(shared_memory, polys, Inv::n_inv(), num_polys);
    }

    // The same with the thread's entries: the values of b_hat are read into
    // registers before the forward FFT.
    template <typename Fwd, typename Inv, typename T>
    __global__ void
    fused_multiply_entries_kernel(COMPLEX<T>* a, const COMPLEX<T>* b_hat,
                                  const COMPLEX<T>* roots,
                                  const COMPLEX<T>* inverse_roots, int batch)
    {
        COMPLEX<T>* shared_memory = Fwd::shared_memory();
        COMPLEX<T>* polys = Fwd::block_data(a);
        const COMPLEX<T>* other = Fwd::block_data(b_hat);
        const int num_polys = Fwd::num_polys(batch);

        const ThreadEntries entries = Fwd().entries(num_polys);
        Fwd().load(shared_memory, polys, num_polys);
        COMPLEX<T> other_values[ThreadEntries::count];
        if (entries.valid)
            for (int k = 0; k < ThreadEntries::count; k++)
                other_values[k] = other[entries.index[k]];
        Fwd().execute(shared_memory, roots);
        if (entries.valid)
            for (int k = 0; k < ThreadEntries::count; k++)
            {
                COMPLEX<T>& value = shared_memory[entries.position[k]];
                value = value * other_values[k];
            }
        __syncthreads();
        Inv().execute(shared_memory, inverse_roots);
        Inv().store(shared_memory, polys, Inv::n_inv(), num_polys);
    }

    // The product of a and b (not transformed) with b in a second shared
    // memory buffer; launched with 2 * shared_memory_size bytes.
    template <typename Fwd, typename Inv, typename T>
    __global__ void fused_multiply_two_buffers_kernel(
        COMPLEX<T>* a, const COMPLEX<T>* b, const COMPLEX<T>* roots,
        const COMPLEX<T>* inverse_roots, int batch)
    {
        COMPLEX<T>* a_shared = Fwd::shared_memory();
        COMPLEX<T>* b_shared =
            a_shared + Fwd::shared_memory_size / sizeof(COMPLEX<T>);
        COMPLEX<T>* polys = Fwd::block_data(a);
        const int num_polys = Fwd::num_polys(batch);

        Fwd().load(a_shared, polys, num_polys);
        Fwd().load(b_shared, Fwd::block_data(b), num_polys);
        Fwd().execute(a_shared, roots);
        Fwd().execute(b_shared, roots);
        Fwd().for_each(
            a_shared,
            [&](COMPLEX<T>& value, size_t index)
            { value = value * b_shared[index]; },
            num_polys);
        Inv().execute(a_shared, inverse_roots);
        Inv().store(a_shared, polys, Inv::n_inv(), num_polys);
    }

    // One stage of a transform, as FFT::launch runs it.
    template <typename Stage, type D, typename T, typename TIn, typename TOut>
    __global__ void stage_kernel(const TIn* in, TOut* out,
                                 const COMPLEX<T>* roots, COMPLEX<T> n_inverse)
    {
        COMPLEX<T>* shared_memory = Stage::shared_memory();
        Stage().load(shared_memory, in);
        Stage().execute(shared_memory, roots);
        if constexpr (D == INVERSE && Stage::last)
            Stage().store(shared_memory, out, n_inverse);
        else
            Stage().store(shared_memory, out);
    }

    // The last forward stage, the product with b_hat and the first inverse
    // stage in one kernel.
    template <typename Last, typename First, typename T>
    __global__ void fused_stage_kernel(COMPLEX<T>* a, const COMPLEX<T>* b_hat,
                                       const COMPLEX<T>* roots,
                                       const COMPLEX<T>* inverse_roots,
                                       COMPLEX<T> n_inverse)
    {
        COMPLEX<T>* shared_memory = Last::shared_memory();
        Last().load(shared_memory, a);
        Last().execute(shared_memory, roots);
        Last().for_each(shared_memory, [&](COMPLEX<T>& value, size_t index)
                        { value = value * b_hat[index]; });
        First().execute(shared_memory, inverse_roots);
        if constexpr (First::last)
            First().store(shared_memory, a, n_inverse);
        else
            First().store(shared_memory, a);
    }

    // ---------------- Tests ----------------

    template <typename T> class DeviceTests
    {
      public:
        DeviceTests() : rng_(2026) {}

        int run()
        {
            int result = 0;
            result |= twiddles();
            result |= launch();
            result |= fused();
            result |= fused_stages();
            result |= validation();
            return result;
        }

      private:
        static constexpr double kTwiddleTolerance =
            std::is_same<T, Float32>::value ? 1e-7 : 1e-15;
        // The host tables can differ from the references by one ulp,
        // depending on whether the compiler merges sin and cos into sincos.
        static constexpr double kHostTwiddleTolerance =
            std::is_same<T, Float32>::value ? 1e-7 : 4e-16;

        std::string suite_name(const std::string& suite) const
        {
            return type_name<T>() + ".device." + suite;
        }

        std::vector<COMPLEX<T>> random(size_t count)
        {
            return to_t<T>(random_complex(rng_, count));
        }

        // a * b with the host API: forward FFT of both, inverse FFT with the
        // product. Also returns b_hat = FFT(b).
        template <int LogN, ReductionPolynomial R>
        std::vector<COMPLEX<T>> host_product(const std::vector<COMPLEX<T>>& a,
                                             const std::vector<COMPLEX<T>>& b,
                                             int batch,
                                             std::vector<COMPLEX<T>>& b_hat)
        {
            const size_t count = a.size();
            std::vector<COMPLEX<T>> both(a);
            both.insert(both.end(), b.begin(), b.end());
            DeviceBuffer<COMPLEX<T>> data(both);
            DeviceBuffer<COMPLEX<T>> roots(
                Desc<T, LogN, R, FORWARD>::make_twiddles_host());
            DeviceBuffer<COMPLEX<T>> inverse_roots(
                Desc<T, LogN, R, INVERSE>::make_twiddles_host());
            GPU_FFT(data.get(), roots.get(), host_config<T>(LogN, FORWARD, R),
                    2 * batch, false);
            sync();
            std::vector<COMPLEX<T>> spectra = data.download();
            b_hat.assign(spectra.begin() + count, spectra.end());
            GPU_FFT(data.get(), inverse_roots.get(),
                    host_config<T>(LogN, INVERSE, R), batch, true);
            sync();
            std::vector<COMPLEX<T>> product = data.download();
            product.resize(count);
            return product;
        }

        // ---------------- Root tables ----------------

        template <int LogN, ReductionPolynomial R, type D>
        void twiddle_case(TestReport& report)
        {
            using FFT = Desc<T, LogN, R, D>;
            const std::string name =
                case_name("twiddles" + dir_name(D), LogN, R, 0, 1);
            guarded(report, name,
                    [&]
                    {
                        const std::vector<COMPLEX<T>> host =
                            FFT::make_twiddles_host();
                        const std::vector<Ref> reference =
                            (R == X_N_minus)
                                ? cyclic_table(LogN, D == INVERSE)
                                : negacyclic_table(LogN, D == INVERSE);
                        report.check_close(name + " host", host, reference,
                                           kHostTwiddleTolerance);

                        DeviceBuffer<COMPLEX<T>> table(FFT::twiddle_table_size);
                        FFT::make_twiddles(table.get());
                        sync();
                        report.check_close(name + " device", table.download(),
                                           reference, kTwiddleTolerance);
                    });
        }

        template <int LogN> void twiddle_size(TestReport& report)
        {
            twiddle_case<LogN, X_N_minus, FORWARD>(report);
            twiddle_case<LogN, X_N_minus, INVERSE>(report);
            twiddle_case<LogN, X_N_plus, FORWARD>(report);
            twiddle_case<LogN, X_N_plus, INVERSE>(report);
        }

        template <int... LogN>
        void twiddle_sizes(TestReport& report,
                           std::integer_sequence<int, LogN...>)
        {
            (twiddle_size<LogN>(report), ...);
        }

        int twiddles()
        {
            TestReport report(suite_name("twiddles"));
            twiddle_sizes(report, std::integer_sequence<int, 1, 2, 3, 5, 9, 10,
                                                        11, 12, 16, 20, 24>());
            return report.finish();
        }

        // ---------------- FFT::launch ----------------

        // FFT::launch against GPU_FFT, complex or real input (forward) or
        // output (inverse), out of place and in place.
        template <int LogN, ReductionPolynomial R, type D, bool RealIO,
                  int Threads = 0>
        void launch_case(TestReport& report, int batch)
        {
            using FFT = DescWith<T, LogN, R, D, Threads>;
            const std::string name = case_name(
                std::string(RealIO ? "launch real" : "launch") + dir_name(D),
                LogN, R, Threads, batch);
            guarded(
                report, name,
                [&]
                {
                    const size_t count = size_t(batch) << LogN;
                    DeviceBuffer<COMPLEX<T>> roots(FFT::make_twiddles_host());
                    const fft_configuration<T> cfg = host_config<T>(LogN, D, R);
                    const std::vector<COMPLEX<T>> input = random(count);

                    if constexpr (!RealIO)
                    {
                        DeviceBuffer<COMPLEX<T>> expected(input);
                        DeviceBuffer<COMPLEX<T>> in(input);
                        DeviceBuffer<COMPLEX<T>> out(count);
                        GPU_FFT(expected.get(), roots.get(), cfg, batch, false);
                        FFT::launch(static_cast<const COMPLEX<T>*>(in.get()),
                                    out.get(), roots.get(), batch);
                        sync();
                        const std::vector<COMPLEX<T>> reference =
                            expected.download();
                        report.check_equal(name, out.download(), reference);
                        FFT::launch(in.get(), in.get(), roots.get(), batch);
                        sync();
                        report.check_equal(name + " in place", in.download(),
                                           reference);
                    }
                    else if constexpr (D == FORWARD)
                    {
                        const std::vector<T> real =
                            to_t<T>(random_real(rng_, count));
                        DeviceBuffer<T> in(real);
                        DeviceBuffer<COMPLEX<T>> expected(count);
                        DeviceBuffer<COMPLEX<T>> out(count);
                        GPU_FFT(in.get(), expected.get(), roots.get(), cfg,
                                batch, false);
                        FFT::launch(static_cast<const T*>(in.get()), out.get(),
                                    roots.get(), batch);
                        sync();
                        report.check_equal(name, out.download(),
                                           expected.download());
                    }
                    else
                    {
                        // Both overwrite the input with intermediate values.
                        DeviceBuffer<COMPLEX<T>> host_in(input);
                        DeviceBuffer<COMPLEX<T>> in(input);
                        DeviceBuffer<T> expected(count);
                        DeviceBuffer<T> out(count);
                        GPU_FFT(expected.get(), host_in.get(), roots.get(), cfg,
                                batch, false);
                        FFT::launch(in.get(), out.get(), roots.get(), batch);
                        sync();
                        report.check_equal(name, out.download(),
                                           expected.download());
                    }
                });
        }

        template <int LogN, int Threads = 0>
        void launch_size(TestReport& report, int batch)
        {
            launch_case<LogN, X_N_minus, FORWARD, false, Threads>(report,
                                                                  batch);
            launch_case<LogN, X_N_minus, INVERSE, false, Threads>(report,
                                                                  batch);
            launch_case<LogN, X_N_plus, FORWARD, false, Threads>(report, batch);
            launch_case<LogN, X_N_plus, INVERSE, false, Threads>(report, batch);
            launch_case<LogN, X_N_minus, FORWARD, true, Threads>(report, batch);
            launch_case<LogN, X_N_plus, INVERSE, true, Threads>(report, batch);
        }

        template <int... LogN>
        void launch_sizes(TestReport& report,
                          std::integer_sequence<int, LogN...>)
        {
            (launch_size<LogN>(report, batch_for(LogN)), ...);
        }

        int launch()
        {
            TestReport report(suite_name("launch"));
            launch_sizes(
                report,
                std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                      12, 13, 16, 17, 20, 23, 24>());
            // The low ring kernel takes batches above the grid limit of the
            // stage kernels.
            launch_case<1, X_N_plus, FORWARD, false>(report, 65537);
            // Every thread count of BlockDim<T>.
            launch_size<11, 1024>(report, 3);
            launch_size<12, 128>(report, 3);
            launch_size<16, 128>(report, 3);
            launch_size<16, 512>(report, 3);
            launch_size<16, 1024>(report, 3);
            launch_size<20, 256>(report, 3);
            launch_size<24, 128>(report, 1);
            launch_size<24, 1024>(report, 1);
            return report.finish();
        }

        // ---------------- Single kernel with own work ----------------

        // Variant 0: for_each, 1: entries, 2: a second shared memory buffer.
        template <int LogN, ReductionPolynomial R, int Variant, int Threads = 0,
                  int Batches = 0>
        void fused_case(TestReport& report, int batch)
        {
            using Fwd = DescWith<T, LogN, R, FORWARD, Threads, Batches>;
            using Inv = DescWith<T, LogN, R, INVERSE, Threads, Batches>;
            static_assert(Fwd::num_stages == 1 && Inv::num_stages == 1);
            const char* variants[] = {"fused for_each", "fused entries",
                                      "fused two buffers"};
            const std::string name =
                case_name(variants[Variant], LogN, R, Threads, batch) +
                (Batches ? " polys/block=" + std::to_string(Batches) : "");
            guarded(
                report, name,
                [&]
                {
                    const size_t count = size_t(batch) << LogN;
                    const std::vector<COMPLEX<T>> a = random(count);
                    const std::vector<COMPLEX<T>> b = random(count);
                    std::vector<COMPLEX<T>> b_hat;
                    const std::vector<COMPLEX<T>> expected =
                        host_product<LogN, R>(a, b, batch, b_hat);

                    DeviceBuffer<COMPLEX<T>> data(a);
                    DeviceBuffer<COMPLEX<T>> other(Variant == 2 ? b : b_hat);
                    DeviceBuffer<COMPLEX<T>> roots(Fwd::make_twiddles_host());
                    DeviceBuffer<COMPLEX<T>> inverse_roots(
                        Inv::make_twiddles_host());
                    if constexpr (Variant == 0)
                        fused_multiply_kernel<Fwd, Inv, T>
                            <<<Fwd::grid(batch), Fwd::block_dim,
                               Fwd::shared_memory_size>>>(
                                data.get(), other.get(), roots.get(),
                                inverse_roots.get(), batch);
                    else if constexpr (Variant == 1)
                        fused_multiply_entries_kernel<Fwd, Inv, T>
                            <<<Fwd::grid(batch), Fwd::block_dim,
                               Fwd::shared_memory_size>>>(
                                data.get(), other.get(), roots.get(),
                                inverse_roots.get(), batch);
                    else
                    {
                        // Above 48 KB the kernel needs the attribute.
                        auto kernel =
                            fused_multiply_two_buffers_kernel<Fwd, Inv, T>;
                        GPUFFT_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            int(2 * Fwd::shared_memory_size)));
                        kernel<<<Fwd::grid(batch), Fwd::block_dim,
                                 2 * Fwd::shared_memory_size>>>(
                            data.get(), other.get(), roots.get(),
                            inverse_roots.get(), batch);
                    }
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                    sync();
                    report.check_equal(name, data.download(), expected);
                });
        }

        template <int LogN, int Threads = 0>
        void fused_size(TestReport& report, int batch)
        {
            fused_case<LogN, X_N_minus, 0, Threads>(report, batch);
            fused_case<LogN, X_N_plus, 0, Threads>(report, batch);
            fused_case<LogN, X_N_plus, 1, Threads>(report, batch);
            fused_case<LogN, X_N_minus, 2, Threads>(report, batch);
        }

        template <int... LogN>
        void fused_sizes(TestReport& report,
                         std::integer_sequence<int, LogN...>)
        {
            (fused_size<LogN>(report, batch_for(LogN)), ...);
        }

        int fused()
        {
            TestReport report(suite_name("fused"));
            fused_sizes(
                report,
                std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>());
            fused_size<11, 1024>(report, 3);
            // BatchesPerBlock other than the default: blocks of 5, 5 and 2
            // polynomials.
            fused_case<4, X_N_plus, 0, 0, 5>(report, 12);
            fused_case<4, X_N_minus, 1, 0, 5>(report, 12);
            return report.finish();
        }

        // ---------------- Stage kernels with own work ----------------

        // The stages of a * b with the last forward and the first inverse
        // stage fused around the product with b_hat.
        template <int LogN, ReductionPolynomial R, int Threads = 0>
        void fused_stages_case(TestReport& report, int batch)
        {
            using Fwd = DescWith<T, LogN, R, FORWARD, Threads>;
            using Inv = DescWith<T, LogN, R, INVERSE, Threads>;
            static_assert(Fwd::num_stages > 1 &&
                          Fwd::num_stages == Inv::num_stages);
            constexpr int stages = Fwd::num_stages;
            const std::string name =
                case_name("fused stages", LogN, R, Threads, batch);
            guarded(
                report, name,
                [&]
                {
                    const size_t count = size_t(batch) << LogN;
                    const std::vector<COMPLEX<T>> a = random(count);
                    const std::vector<COMPLEX<T>> b = random(count);
                    std::vector<COMPLEX<T>> b_hat_host;
                    const std::vector<COMPLEX<T>> expected =
                        host_product<LogN, R>(a, b, batch, b_hat_host);

                    DeviceBuffer<COMPLEX<T>> data(a);
                    DeviceBuffer<COMPLEX<T>> b_hat(b_hat_host);
                    DeviceBuffer<COMPLEX<T>> roots(Fwd::make_twiddles_host());
                    DeviceBuffer<COMPLEX<T>> inverse_roots(
                        Inv::make_twiddles_host());
                    const COMPLEX<T> n_inverse = Inv::n_inv();

                    Fwd::for_each_stage(
                        [&](auto I)
                        {
                            if constexpr (I + 1 < stages)
                            {
                                using Stage = typename Fwd::template stage<I>;
                                stage_kernel<Stage, FORWARD, T>
                                    <<<Stage::grid(batch), Stage::block_dim,
                                       Stage::shared_memory_size>>>(
                                        data.get(), data.get(), roots.get(),
                                        n_inverse);
                            }
                        });
                    using Last = typename Fwd::template stage<stages - 1>;
                    using First = typename Inv::template stage<0>;
                    fused_stage_kernel<Last, First, T>
                        <<<Last::grid(batch), Last::block_dim,
                           Last::shared_memory_size>>>(
                            data.get(), b_hat.get(), roots.get(),
                            inverse_roots.get(), n_inverse);
                    Inv::for_each_stage(
                        [&](auto I)
                        {
                            if constexpr (I > 0)
                            {
                                using Stage = typename Inv::template stage<I>;
                                stage_kernel<Stage, INVERSE, T>
                                    <<<Stage::grid(batch), Stage::block_dim,
                                       Stage::shared_memory_size>>>(
                                        data.get(), data.get(),
                                        inverse_roots.get(), n_inverse);
                            }
                        });
                    GPUFFT_CUDA_CHECK(cudaGetLastError());
                    sync();
                    report.check_equal(name, data.download(), expected);
                });
        }

        int fused_stages()
        {
            TestReport report(suite_name("fused_stages"));
            fused_stages_case<11, X_N_plus>(report, 3);
            fused_stages_case<12, X_N_minus>(report, 3);
            fused_stages_case<12, X_N_plus>(report, 3);
            fused_stages_case<16, X_N_plus>(report, 3);
            fused_stages_case<16, X_N_minus, 128>(report, 3);
            fused_stages_case<16, X_N_plus, 1024>(report, 3);
            fused_stages_case<17, X_N_plus>(report, 2);
            fused_stages_case<20, X_N_minus, 512>(report, 2);
            fused_stages_case<24, X_N_plus>(report, 1);
            return report.finish();
        }

        // ---------------- Input checks ----------------

        template <typename F>
        void expect_invalid(TestReport& report, const std::string& name, F fn)
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
            using Small = Desc<T, 4, X_N_plus, FORWARD>;
            using Large = Desc<T, 12, X_N_plus, FORWARD>;
            using Largest = Desc<T, 24, X_N_plus, FORWARD>;
            expect_invalid(report, "batch 0",
                           [&] { Small::launch(c, c, c, 0); });
            expect_invalid(report, "batch -1",
                           [&] { Small::launch(c, c, c, -1); });
            expect_invalid(report, "stages batch 65536",
                           [&] { Large::launch(c, c, c, 65536); });
            expect_invalid(report, "batch * N = 2^31",
                           [&] { Largest::launch(c, c, c, 128); });
            return report.finish();
        }

        std::mt19937_64 rng_;
    };

} // namespace

int main()
{
    try
    {
        int result = DeviceTests<Float32>().run();
        result |= DeviceTests<Float64>().run();
        return result;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
