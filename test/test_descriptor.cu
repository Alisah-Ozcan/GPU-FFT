// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Compile-time tests of the FFT description operators. The checks are
// static_asserts; the test passes if this file compiles.

#include "gpufft/descriptor.cuh"

#include <cstdio>

using namespace gpufft;

using FwdFFT =
    decltype(Size<4096>() + Direction<FORWARD>() + Precision<Float64>() +
             ReductionPoly<X_N_minus>() + Block());

static_assert(std::is_same<
              FwdFFT,
              FFTDescription<Size<4096>, Direction<FORWARD>, Precision<Float64>,
                             ReductionPoly<X_N_minus>, Block>>::value);

// Descriptions add up with operators on either side and with each other.
using Left = decltype(Size<1024>() + Direction<INVERSE>());
using Right = decltype(Precision<Float32>() + Block());
static_assert(std::is_same<decltype(Left() + Right()),
                           FFTDescription<Size<1024>, Direction<INVERSE>,
                                          Precision<Float32>, Block>>::value);
static_assert(std::is_same<decltype(BlockDim<512>() + Left()),
                           FFTDescription<BlockDim<512>, Size<1024>,
                                          Direction<INVERSE>>>::value);

static_assert(Size<4096>::value == 4096 && Size<4096>::log_value == 12);
static_assert(Size<2>::log_value == 1 && Size<(1 << 28)>::log_value == 28);
static_assert(BlockDim<256, 2>::x == 256 && BlockDim<256, 2>::y == 2 &&
              BlockDim<256, 2>::z == 1);

static_assert(is_fft_operator<Size<8>>::value);
static_assert(is_fft_operator<Block>::value);
static_assert(!is_fft_operator<int>::value);
static_assert(!is_fft_operator<FwdFFT>::value);
static_assert(is_fft_description<FwdFFT>::value);
static_assert(!is_fft_description<Size<8>>::value);

static_assert(has_operator<FFTOperator::Size, FwdFFT>::value);
static_assert(has_operator<FFTOperator::Block, FwdFFT>::value);
static_assert(!has_operator<FFTOperator::BlockDim, FwdFFT>::value);

static_assert(std::is_same<get_or_default_t<FFTOperator::Precision, FwdFFT,
                                            Precision<Float32>>,
                           Precision<Float64>>::value);
static_assert(
    std::is_same<get_or_default_t<FFTOperator::BlockDim, FwdFFT, BlockDim<512>>,
                 BlockDim<512>>::value);

constexpr bool same_dim(dim3 d, unsigned x, unsigned y, unsigned z)
{
    return d.x == x && d.y == y && d.z == z;
}

template <typename T, typename = void> struct has_block_dim : std::false_type
{
};

template <typename T>
struct has_block_dim<T, std::void_t<decltype(T::block_dim)>> : std::true_type
{
};

// A partial description has no members.
static_assert(!has_block_dim<decltype(Size<256>() + Block())>::value);

// 2^12 forward, Float64: 3 levels in blocks (64, 4), then 9 levels in a
// block (256, 1), as GPU_FFT.
static_assert(FwdFFT::size == 4096 && FwdFFT::log_size == 12);
static_assert(FwdFFT::direction == FORWARD);
static_assert(std::is_same<FwdFFT::real_type, Float64>::value);
static_assert(std::is_same<FwdFFT::value_type, Complex64>::value);
static_assert(FwdFFT::num_stages == 2 && FwdFFT::batches_per_block == 1);
static_assert(same_dim(FwdFFT::stage<0>::block_dim, 64, 4, 1));
static_assert(same_dim(FwdFFT::stage<0>::grid(3), 8, 1, 3));
static_assert(FwdFFT::stage<0>::shared_memory_size == 512 * sizeof(Complex64));
static_assert(same_dim(FwdFFT::stage<1>::block_dim, 256, 1, 1));
static_assert(same_dim(FwdFFT::stage<1>::grid(3), 1, 8, 3));
static_assert(!FwdFFT::stage<0>::last && FwdFFT::stage<1>::last);
static_assert(!has_block_dim<FwdFFT>::value);
static_assert(FwdFFT::twiddle_table_size == 2048);

// The inverse runs the stages in the opposite order.
using InvFFT =
    decltype(Size<4096>() + Direction<INVERSE>() + Precision<Float64>() +
             ReductionPoly<X_N_plus>() + Block());
static_assert(same_dim(InvFFT::stage<0>::block_dim, 256, 1, 1));
static_assert(same_dim(InvFFT::stage<1>::block_dim, 64, 4, 1));
static_assert(InvFFT::twiddle_table_size == 4096);

// 2^8: low ring kernel, 2 polynomials per block by default.
using Small = decltype(Size<256>() + Direction<FORWARD>() +
                       Precision<Float32>() + Block());
static_assert(Small::num_stages == 1 && Small::batches_per_block == 2);
static_assert(same_dim(Small::block_dim, 128, 2, 1));
static_assert(Small::shared_memory_size == 512 * sizeof(Complex32));
static_assert(same_dim(Small::grid(5), 3, 1, 1));

// 256 threads for every low ring size by default.
static_assert(decltype(Size<2>() + Direction<FORWARD>() +
                       Precision<Float32>())::batches_per_block == 256);
static_assert(decltype(Size<512>() + Direction<FORWARD>() +
                       Precision<Float32>())::batches_per_block == 1);

using SmallBatches = decltype(Small() + BatchesPerBlock<4>());
static_assert(SmallBatches::batches_per_block == 4);
static_assert(same_dim(SmallBatches::block_dim, 128, 4, 1));
static_assert(SmallBatches::shared_memory_size == 1024 * sizeof(Complex32));
static_assert(same_dim(SmallBatches::grid(5), 2, 1, 1));

using SmallBlockDim = decltype(Small() + BlockDim<128, 8>());
static_assert(SmallBlockDim::batches_per_block == 8);
static_assert(same_dim(SmallBlockDim::block_dim, 128, 8, 1));

// 2^10: a single stage kernel of 512 threads, one polynomial per block. The
// single kernel API indexes blocks by blockIdx.x; the stage API, like the
// host API, by blockIdx.z.
using Mid = decltype(Size<1024>() + Direction<FORWARD>() +
                     Precision<Float64>() + Block());
static_assert(Mid::num_stages == 1 && Mid::batches_per_block == 1);
static_assert(same_dim(Mid::block_dim, 512, 1, 1));
static_assert(same_dim(Mid::grid(7), 7, 1, 1));
static_assert(same_dim(Mid::stage<0>::grid(7), 1, 1, 7));
static_assert(Mid::shared_memory_size == 1024 * sizeof(Complex64));

// For Size above 2^9, BlockDim<T> sets the threads of every kernel:
// 2^16 with 128 threads runs 4 + 4 levels in blocks (16, 8), then 8 levels
// in a block (128, 1).
using Threads128 =
    decltype(Size<(1 << 16)>() + Direction<FORWARD>() + Precision<Float64>() +
             ReductionPoly<X_N_plus>() + Block() + BlockDim<128>());
static_assert(Threads128::block_threads == 128 && Threads128::num_stages == 3);
static_assert(Threads128::batches_per_block == 1);
static_assert(same_dim(Threads128::stage<0>::block_dim, 16, 8, 1));
static_assert(same_dim(Threads128::stage<0>::grid(3), 256, 1, 3));
static_assert(same_dim(Threads128::stage<1>::block_dim, 16, 8, 1));
static_assert(same_dim(Threads128::stage<1>::grid(3), 16, 16, 3));
static_assert(same_dim(Threads128::stage<2>::block_dim, 128, 1, 1));
static_assert(same_dim(Threads128::stage<2>::grid(3), 1, 256, 3));
static_assert(Threads128::stage<2>::shared_memory_size ==
              256 * sizeof(Complex64));
static_assert(FwdFFT::block_threads == 0 && Mid::block_threads == 0);

// 2^24 with 128 threads runs 5 + 5 + 6 levels, then 8 levels in 2^16
// groups, which do not fit grid y: the last forward and the first inverse
// stage swap the grid axes.
using Large128 = decltype(Size<(1 << 24)>() + Direction<FORWARD>() +
                          Precision<Float32>() + Block() + BlockDim<128>());
using LargeInverse128 =
    decltype(Size<(1 << 24)>() + Direction<INVERSE>() + Precision<Float32>() +
             Block() + BlockDim<128>());
static_assert(Large128::num_stages == 4);
static_assert(!Large128::stage<2>::swap_grid_axes);
static_assert(Large128::stage<3>::swap_grid_axes);
static_assert(same_dim(Large128::stage<3>::grid(2), 65536, 1, 2));
static_assert(LargeInverse128::stage<0>::swap_grid_axes);
static_assert(!FwdFFT::stage<1>::swap_grid_axes);

// With the default 256 threads the last forward stage swaps the grid axes
// from 2^25: 5 + 5 + 6 levels, then 9 levels in 2^16 groups.
using Fwd25 = decltype(Size<(1 << 25)>() + Direction<FORWARD>() +
                       Precision<Float64>() + Block());
static_assert(Fwd25::num_stages == 4 && Fwd25::stage<3>::swap_grid_axes);
static_assert(same_dim(Fwd25::stage<3>::grid(1), 65536, 1, 1));
static_assert(!decltype(Size<(1 << 24)>() + Direction<FORWARD>() +
                        Precision<Float64>() +
                        Block())::stage<2>::swap_grid_axes);

// 2^11 with 1024 threads fits a single kernel; 2^10 with 128 threads does
// not.
static_assert(decltype(Size<2048>() + Direction<INVERSE>() +
                       Precision<Float32>() + Block() +
                       BlockDim<1024>())::num_stages == 1);
static_assert(decltype(Mid() + BlockDim<128>())::num_stages == 2);
static_assert(decltype(Mid() + BlockDim<512>())::num_stages == 1);

// Stages per size and T.
template <int LogN, int Threads>
constexpr int stages_with =
    decltype(Size<(1 << LogN)>() + Direction<FORWARD>() + Precision<Float64>() +
             Block() + BlockDim<Threads>())::num_stages;
static_assert(stages_with<16, 128> == 3 && stages_with<16, 256> == 2 &&
              stages_with<16, 512> == 2 && stages_with<16, 1024> == 2);
static_assert(stages_with<20, 128> == 3 && stages_with<20, 256> == 3 &&
              stages_with<20, 512> == 3 && stages_with<20, 1024> == 2);
static_assert(stages_with<24, 128> == 4 && stages_with<24, 256> == 4 &&
              stages_with<24, 512> == 3 && stages_with<24, 1024> == 3);
static_assert(stages_with<28, 128> == 5 && stages_with<28, 256> == 4 &&
              stages_with<28, 512> == 4 && stages_with<28, 1024> == 3);

// The default plans are the plans of 256 threads, and of 512 threads for
// 2^10.
constexpr bool same_config(const KernelConfig& a, const KernelConfig& b)
{
    return a.griddim_x == b.griddim_x && a.griddim_y == b.griddim_y &&
           a.blockdim_x == b.blockdim_x && a.blockdim_y == b.blockdim_y &&
           a.shared_memory == b.shared_memory &&
           a.shared_index == b.shared_index && a.logm == b.logm && a.k == b.k &&
           a.outer_iteration_count == b.outer_iteration_count &&
           a.not_last_kernel == b.not_last_kernel;
}

template <typename T> constexpr bool default_plans_use_threads()
{
    for (int log_n = 10; log_n <= 28; log_n++)
    {
        const int threads = (log_n == 10) ? 512 : 256;
        const type directions[] = {FORWARD, INVERSE};
        for (type direction : directions)
        {
            const StagePlan defaults = stage_plan<T>(direction, log_n);
            const StagePlan plan = stage_plan<T>(direction, log_n, threads);
            if (plan.count != defaults.count)
            {
                return false;
            }
            for (int i = 0; i < plan.count; i++)
            {
                if (!same_config(plan.stage[i], defaults.stage[i]))
                {
                    return false;
                }
            }
        }
    }
    return true;
}
static_assert(default_plans_use_threads<Float32>());
static_assert(default_plans_use_threads<Float64>());

// The levels of every plan add up to the size and every grid fits.
template <typename T> constexpr bool plans_cover_all_levels()
{
    for (int threads = 128; threads <= 1024; threads *= 2)
    {
        for (int log_n = 10; log_n <= 28; log_n++)
        {
            if (threads > (1 << log_n) / 2)
            {
                continue;
            }
            const StagePlan plan = stage_plan<T>(FORWARD, log_n, threads);
            int levels = 0;
            for (int i = 0; i < plan.count; i++)
            {
                const KernelConfig& c = plan.stage[i];
                if (c.logm != levels ||
                    c.blockdim_x * c.blockdim_y != threads ||
                    c.griddim_y > 65535 ||
                    static_cast<long long>(c.griddim_x) * c.griddim_y *
                            threads * 2 !=
                        (1LL << log_n))
                {
                    return false;
                }
                levels += c.outer_iteration_count;
            }
            if (levels != log_n)
            {
                return false;
            }
        }
    }
    return true;
}
static_assert(plans_cover_all_levels<Float32>());
static_assert(plans_cover_all_levels<Float64>());

// The last forward stage and the first inverse stage of every multi-kernel
// size have the same launch parameters and thread mapping, so a kernel can
// run both without storing in between (see FFTStage::for_each).
template <typename T> constexpr bool last_forward_matches_first_inverse()
{
    for (int threads = 0; threads <= 1024;
         threads = threads ? 2 * threads : 128)
    {
        for (int log_n = 11; log_n <= 28; log_n++)
        {
            if (threads > (1 << log_n) / 2)
            {
                continue;
            }
            const StagePlan forward = stage_plan<T>(FORWARD, log_n, threads);
            const StagePlan inverse = stage_plan<T>(INVERSE, log_n, threads);
            const KernelConfig a = forward.stage[forward.count - 1];
            const KernelConfig b = inverse.stage[0];
            if (a.griddim_x != b.griddim_x || a.griddim_y != b.griddim_y ||
                a.blockdim_x != b.blockdim_x || a.blockdim_y != b.blockdim_y ||
                a.shared_memory != b.shared_memory || a.logm != b.k ||
                a.outer_iteration_count != b.outer_iteration_count ||
                forward.swap_grid_axes[forward.count - 1] !=
                    inverse.swap_grid_axes[0])
            {
                return false;
            }
        }
    }
    return true;
}
static_assert(last_forward_matches_first_inverse<Float32>());
static_assert(last_forward_matches_first_inverse<Float64>());

// The special FFT runs the inverse plan forward and the forward plan
// inverse, so its last forward stage and its first inverse stage share their
// layout as well.
using FwdSpecial =
    decltype(Size<4096>() + Direction<FORWARD>() + Precision<Float64>() +
             Transform<TransformType::Special>() + Block());
using InvSpecial =
    decltype(Size<4096>() + Direction<INVERSE>() + Precision<Float64>() +
             Transform<TransformType::Special>() + Block());
static_assert(FwdSpecial::transform == TransformType::Special);
static_assert(FwdSpecial::num_stages == 2);
static_assert(same_dim(FwdSpecial::stage<0>::block_dim, 256, 1, 1));
static_assert(same_dim(FwdSpecial::stage<1>::block_dim, 64, 4, 1));
static_assert(same_dim(InvSpecial::stage<0>::block_dim, 64, 4, 1));
static_assert(same_dim(InvSpecial::stage<0>::grid(3),
                       FwdSpecial::stage<1>::grid(3).x,
                       FwdSpecial::stage<1>::grid(3).y, 3));
static_assert(FwdSpecial::twiddle_table_size == 4096);
static_assert(decltype(Size<512>() + Direction<FORWARD>() +
                       Precision<Float32>() +
                       Transform<TransformType::Special>())::num_stages == 1);

// The FFNT of Size real values runs an FFT of Size / 2 values.
using FwdFFNT =
    decltype(Size<8192>() + Direction<FORWARD>() + Precision<Float64>() +
             Transform<TransformType::FFNT>() + Block());
static_assert(FwdFFNT::num_stages == 2);
static_assert(same_dim(FwdFFNT::stage<0>::block_dim, 64, 4, 1));
static_assert(same_dim(FwdFFNT::stage<1>::block_dim, 256, 1, 1));
static_assert(FwdFFNT::twiddle_table_size == 2048 &&
              FwdFFNT::twist_table_size == 4096);
using SmallFFNT =
    decltype(Size<1024>() + Direction<INVERSE>() + Precision<Float32>() +
             Transform<TransformType::FFNT>() + Block());
static_assert(SmallFFNT::num_stages == 1 && SmallFFNT::batches_per_block == 1);
static_assert(same_dim(SmallFFNT::block_dim, 256, 1, 1));
static_assert(SmallFFNT::shared_memory_size == 512 * sizeof(Complex32));
static_assert(decltype(Size<4>() + Direction<FORWARD>() + Precision<Float32>() +
                       Transform<TransformType::FFNT>())::batches_per_block ==
              256);
static_assert(decltype(FwdFFNT() + BlockDim<1024>())::num_stages == 2 &&
              decltype(Size<4096>() + Direction<FORWARD>() +
                       Precision<Float64>() + Transform<TransformType::FFNT>() +
                       BlockDim<1024>())::num_stages == 1);

// Operator values are usable in device code.
__global__ void use_in_kernel(int* out)
{
    out[0] = Size<4096>::log_value + BlockDim<256>::x;
}

int main()
{
    std::printf("descriptor: all compile-time checks passed\n");
    return 0;
}
