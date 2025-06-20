// Copyright 2023-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "fft.cuh"
#include "fft_cpu.cuh"
#include "bench_util.cuh"

using namespace std;
using namespace gpufft;

// typedef Float32 BenchmarkDataType; // Use for 32-bit benchmark
typedef Float64 BenchmarkDataType; // Use for 64-bit benchmark

void GPU_FFT_Forward_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;

    thrust::device_vector<COMPLEX<BenchmarkDataType>> inout_data(2 * ring_size *
                                                                 batch_count);
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(2 * ring_size * batch_count),
        inout_data.begin(), random_functor<COMPLEX<BenchmarkDataType>>(1234));

    thrust::device_vector<COMPLEX<BenchmarkDataType>> root_table_data(
        ring_size);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((ring_size)),
                      root_table_data.begin(),
                      random_functor<COMPLEX<BenchmarkDataType>>(1234));

    state.add_global_memory_reads<BenchmarkDataType>(
        (2 * ring_size * batch_count) + (ring_size * batch_count),
        "Read Memory Size");
    state.add_global_memory_writes<BenchmarkDataType>(ring_size * batch_count,
                                                      "Write Memory Size");
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    fft_configuration<BenchmarkDataType> cfg_fft{};
    cfg_fft.n_power = (static_cast<int>(ring_size_logN) + 1);
    cfg_fft.fft_type = FORWARD;
    cfg_fft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_fft.zero_padding = false;
    cfg_fft.stream = stream;

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_FFT(thrust::raw_pointer_cast(inout_data.data()),
                    thrust::raw_pointer_cast(root_table_data.data()), cfg_fft,
                    static_cast<int>(batch_count), false);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_FFT_Forward_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);

void GPU_FFT_Inverse_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;

    thrust::device_vector<COMPLEX<BenchmarkDataType>> inout_data(2 * ring_size *
                                                                 batch_count);
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(2 * ring_size * batch_count),
        inout_data.begin(), random_functor<COMPLEX<BenchmarkDataType>>(1234));

    thrust::device_vector<COMPLEX<BenchmarkDataType>> root_table_data(
        ring_size);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((ring_size)),
                      root_table_data.begin(),
                      random_functor<COMPLEX<BenchmarkDataType>>(1234));

    state.add_global_memory_reads<BenchmarkDataType>(
        (2 * ring_size * batch_count) + (ring_size * batch_count),
        "Read Memory Size");
    state.add_global_memory_writes<BenchmarkDataType>(ring_size * batch_count,
                                                      "Write Memory Size");
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    COMPLEX<BenchmarkDataType> mod_inverse(1.0, 1.0);
    fft_configuration<BenchmarkDataType> cfg_fft{};
    cfg_fft.n_power = (static_cast<int>(ring_size_logN) + 1);
    cfg_fft.fft_type = INVERSE;
    cfg_fft.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg_fft.zero_padding = false;
    cfg_fft.mod_inverse = mod_inverse;
    cfg_fft.stream = stream;

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_FFT(thrust::raw_pointer_cast(inout_data.data()),
                    thrust::raw_pointer_cast(root_table_data.data()), cfg_fft,
                    static_cast<int>(batch_count), false);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_FFT_Inverse_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);