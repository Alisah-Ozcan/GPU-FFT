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

void GPU_FFT_Poly_Mult_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;

    /*

    /

    /////////////////////////////////////////////////////////////////////////

    fft_configuration<TestDataType> cfg_fft = {.n_power = logn, .fft_type =
    FORWARD, .zero_padding = false, .stream = 0}; GPU_FFNT(Forward_InOut_Datas,
    Temp_Datas, Twist_Table_Device, Root_Table_Device, cfg_fft, batch * 2,
    false);

    TestDataType n_inverse_new = 1.0 / (n>>1);
    fft_configuration<TestDataType> cfg_ifft = {.n_power = logn,
                                  .fft_type = INVERSE,
                                  .zero_padding = false,
                                  .mod_inverse =
    COMPLEX<TestDataType>(n_inverse_new, 0.0), .stream = 0};
    GPU_FFNT(Forward_InOut_Datas, Temp_Datas, Untwist_Table_Device,
    Inverse_Root_Table_Device, cfg_ifft, batch, true);

    */

    thrust::device_vector<BenchmarkDataType> inout_data(ring_size *
                                                        (batch_count * 2));
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(ring_size * (batch_count * 2)),
        inout_data.begin(), random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<COMPLEX<BenchmarkDataType>> temp_data(
        (ring_size >> 1) * (batch_count * 2));
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>((ring_size >> 1) * (batch_count * 2)),
        temp_data.begin(), random_functor<COMPLEX<BenchmarkDataType>>(1234));

    thrust::device_vector<COMPLEX<BenchmarkDataType>> root_table_data(
        (ring_size >> 1));
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((ring_size >> 1)),
                      root_table_data.begin(),
                      random_functor<COMPLEX<BenchmarkDataType>>(1234));

    thrust::device_vector<COMPLEX<BenchmarkDataType>> twist_table_data(
        (ring_size >> 1));
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((ring_size >> 1)),
                      twist_table_data.begin(),
                      random_functor<COMPLEX<BenchmarkDataType>>(1234));

    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    COMPLEX<BenchmarkDataType> mod_inverse(1.0, 1.0);

    fft_configuration<BenchmarkDataType> cfg_fft = {
        .n_power = (static_cast<int>(ring_size_logN)),
        .fft_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .stream = stream};

    fft_configuration<BenchmarkDataType> cfg_ifft = {
        .n_power = (static_cast<int>(ring_size_logN)),
        .fft_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = mod_inverse,
        .stream = stream};

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_FFNT(thrust::raw_pointer_cast(inout_data.data()),
                     thrust::raw_pointer_cast(temp_data.data()),
                     thrust::raw_pointer_cast(twist_table_data.data()),
                     thrust::raw_pointer_cast(root_table_data.data()), cfg_fft,
                     static_cast<int>(batch_count) * 2, false);

            GPU_FFNT(thrust::raw_pointer_cast(inout_data.data()),
                     thrust::raw_pointer_cast(temp_data.data()),
                     thrust::raw_pointer_cast(twist_table_data.data()),
                     thrust::raw_pointer_cast(root_table_data.data()), cfg_ifft,
                     static_cast<int>(batch_count), true);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_FFT_Poly_Mult_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);