# Changelog

## v1.0.0

GPU-FFT now runs compile-time kernels for sizes 2^1 to 2^28, has a device API for running the FFT inside your own kernels, and builds the root tables of all its transforms. Code written for v0.0.0 needs the changes listed under [Migrating from v0.0.0](#migrating-from-v000).

### Added

- Device API (`gpufft/descriptor.cuh`): an FFT is described at compile time as a sum of operators (`Size`, `Direction`, `Precision`, `ReductionPoly`, `Block`, `BatchesPerBlock`, `BlockDim`) and runs inside your own kernels with `load`, `execute` and `store`, per stage for sizes above 2^10. Its results are the same as those of `GPU_FFT`, bit for bit. See the README section "Device API".
  - `for_each` and `entries()` give each thread's values in shared memory for element-wise work, e.g. a pointwise product between the forward and the inverse FFT in one kernel.
  - The last forward stage and the first inverse stage share their layout, so they can run in one kernel.
  - `FFT::launch` runs the whole transform from the host, with complex or real input (forward) and output (inverse).
  - `BlockDim<T>` sets the threads per kernel (128, 256, 512 or 1024) for sizes from 2^10; the stages are planned at compile time.
- Root tables (`gpufft/twiddles.cuh`), built on the host or by a kernel: `make_fft_twiddles` for `GPU_FFT`, `make_ffnt_twiddles` and `make_ffnt_twist` for `GPU_FFNT`, `make_special_fft_twiddles` for `GPU_Special_FFT` (whose table had to be built by the caller), each with a `_host` variant, and `fft_twiddle_table_size`. A description has `make_twiddles_host`, `make_twiddles`, `twiddle_table_size` and `n_inv`.
- `GPU_FFT` for sizes 2^1 to 2^11. Sizes up to 2^10 run in a single kernel; up to 2^9 a block transforms several polynomials and the batch may exceed 65535.
- Sizes up to 2^28 for all transforms (was 2^24).
- Installed `gpufft/config.cuh` with the version (`GPUFFT_VERSION_MAJOR`, `GPUFFT_VERSION_MINOR`, `GPUFFT_VERSION_PATCH`, `GPUFFT_VERSION`) and the CUDA architectures.
- CMake package: `find_package(GPUFFT 1.0)` checks the version (same major version, at least the requested one); `GPUFFT_CUDA_ARCHITECTURES` holds the architectures the library was compiled for.
- Input checks that throw `std::invalid_argument`: sizes out of range and invalid batch sizes.
- CTest correctness tests (`GPUFFT_BUILD_TESTS`; the sizes 2^25 to 2^28 under the label `huge`), the benchmarks `benchmark_fft` and `benchmark_device_api`, a special FFT example, and examples that take `[logn] [batch] [32|64]` from the command line.

### Changed

- All types and functions are in the `gpufft` namespace. The complex types (`COMPLEX`, `Complex32`, `Complex64`, `Float32`, `Float64`, ...) and `complex_arithmetic` are also available in the global namespace unless `GPUFFT_NO_GLOBAL_ALIASES` is defined.
- The host API (`GPU_FFT`, `GPU_FFNT`, `GPU_Special_FFT`) runs kernels whose launch parameters are fixed at compile time for each size. The results for the sizes of v0.0.0 are the same as before, bit for bit; the kernel tables are no longer part of the public header.
- `gpufft/fft.cuh` no longer includes the CPU reference; include `gpufft/fft_cpu.cuh` for `FFT`, `FFNT` and `bitreverse`.
- The CPU classes `FFT` and `FFNT` compute their root tables in double precision for both data types.
- `CooleyTukeyUnit` and `GentlemanSandeUnit` take the root as a const reference.
- The version is `1.0.0` and follows the release tags. Installed paths and the library name carry `<major>.<minor>`: `include/GPUFFT-1.0`, `lib/cmake/GPUFFT-1.0`, `libfft-1.0.a`.
- Code that uses GPU-FFT needs no relocatable device code (`CUDA_SEPARABLE_COMPILATION`).

### Removed

- The unused `zero_padding` field of `fft_configuration`.
- The kernels declared in `fft.cuh` (`ForwardCore`, `InverseCore`, `Special_ForwardCore`, `Special_InverseCore`, `ForwardFFNTCoreUnique`, `ForwardFFNTCoreRegular`, `InverseFFNTCoreRegular`, `InverseFFNTCoreUnique`) and the kernel tables. Use `GPU_FFT`, `GPU_FFNT`, `GPU_Special_FFT` or the device API.
- `customAssert`.
- The separate benchmark executables; `benchmark_fft` covers them.

### Fixed

- `COMPLEX` times a real scalar returned (re·s, re·s) instead of (re·s, im·s), so the CPU `FFT::ifft` was wrong for complex input.
- The compound assignment operators of `COMPLEX` (`+=`, `-=`, `*=`, `/=`) did not compile.
- The `COMPLEX` constructor did not compile for real and imaginary parts of different types, and `COMPLEX::exp` did not accept a literal exponent.
- A warp level race in the last forward stages of `GPU_FFT`, `GPU_FFNT` and the inverse `GPU_Special_FFT`.
- The CPU `FFNT::MULT` rounded negative results incorrectly.
- Batch sizes above 65535, non-positive batch sizes and batch · N ≥ 2^31 failed at launch, did nothing or overflowed addresses; they now throw.
- The examples crashed from size 2^17, accepted unsupported sizes, overflowed negative coefficients and did not free device memory.

### Migrating from v0.0.0

- Qualify names with `gpufft::` or add `using namespace gpufft;`. The complex types keep working without it unless `GPUFFT_NO_GLOBAL_ALIASES` is defined.
- Remove `zero_padding` from `fft_configuration`. Code that initializes it by position (`{n_power, fft_type, reduction_poly, zero_padding, mod_inverse, stream}`) must drop that value; code that sets the fields by name only drops the assignment.
- Include `gpufft/fft_cpu.cuh` where `FFT`, `FFNT` or `bitreverse` are used.
- `COMPLEX` times a real scalar now also scales the imaginary part. Check code that relied on the old result.
- Replace direct launches of the kernels of `fft.cuh` with `GPU_FFT`, `GPU_FFNT`, `GPU_Special_FFT` or the device API.
- The root tables of v0.0.0 keep working. `make_special_fft_twiddles_host` builds the table of `GPU_Special_FFT`, which callers built themselves before; its entries agree with the usual construction up to rounding.
- Use `find_package(GPUFFT 1.0 CONFIG REQUIRED)` and link `GPUFFT::fft`; the installed library is `libfft-1.0.a`.

## v0.0.0

The state before the rewrite: `GPU_FFT`, `GPU_FFNT` and `GPU_Special_FFT` for 32-bit and 64-bit data with kernel tables chosen at run time, sizes 2^12 to 2^24 (2^11 to 2^24 for the special FFT), as on the `main` branch up to commit `b743607`.
