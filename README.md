# GPU-FFT
Welcome to the GPU-FFT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Fast Fourier Transform (FFT) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

NTT variant of GPU-FFT is available: https://github.com/Alisah-Ozcan/GPU-NTT

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.26
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Build & Install

Configure + build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

Install:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
cmake --install build
```

Notes:
- If you install to a system location (default: `/usr/local`), you may need `sudo` or set `-DCMAKE_INSTALL_PREFIX=/your/prefix`.
- If you omit `-DCMAKE_CUDA_ARCHITECTURES=...`, GPU-FFT defaults to `80;86;89;90`.
- If CMake cannot find `nvcc`, set `CUDACXX=/path/to/nvcc` or pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
- If you change compilers/toolchains, prefer a clean configure: `cmake --fresh -S . -B build`.

### Compile Options

GPU-FFT uses C++17/CUDA17 and applies per-configuration compile flags.

- Build type (single-config generators like Makefiles/Ninja): `-DCMAKE_BUILD_TYPE=Release|Debug|RelWithDebInfo|MinSizeRel`
- Optimization/debug defaults:
  - `Release`: `-O3 -DNDEBUG`
  - `RelWithDebInfo`: `-O3 -g -DNDEBUG`
  - `Debug`: `-g` (and CUDA adds line info)
- Extra warnings: `-DGPUFFT_ENABLE_WARNINGS=ON`

Example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGPUFFT_ENABLE_WARNINGS=ON
cmake --build build --parallel
```

### Testing & Benchmarking

#### Examples (CPU & GPU)

The examples take the ring size, the batch size and the data type (`32` for `Float32`, `64` for `Float64`) from the command line; the defaults are `12 1 64`. They multiply random polynomials and compare the rounded results with the schoolbook product; `Float32` runs use smaller coefficients so that the products stay exact. The special FFT example compares the forward transform with a CPU implementation and checks that the inverse undoes it.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUFFT_BUILD_EXAMPLES=ON
cmake --build build --parallel

./build/bin/example/cpu_fft_example [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
./build/bin/example/gpu_fft_C_C_example [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
./build/bin/example/gpu_fft_R_R_example [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
./build/bin/example/cpu_ffnt_example [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
./build/bin/example/gpu_ffnt_R_R_example [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
./build/bin/example/gpu_special_fft_example [SLOT_COUNT_IN_LOG2] [BATCH_SIZE] [32|64]
# Example: ./build/bin/example/gpu_fft_R_R_example 12 1 32
```

#### Tests

The tests compare every GPU-FFT host API variant (complex and real input/output, X^N-1 and X^N+1, the fused multiplication, the special FFT and the FFNT) against double precision CPU references for 32-bit and 64-bit data, and check small polynomial products exactly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUFFT_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

- `-DGPUFFT_TEST_MAX_LOGN=<N>` sets the largest size covered by the regular suites (default `20`).
- The `large` suites cover the sizes above it, up to 2^24, with one polynomial each.
- The `huge` suites (CTest label `huge`) cover 2^25 to 2^28: they compare random outputs with direct sums and check that the special FFT and the FFNT round trip. They need up to 12 GB of GPU memory and take several minutes; `ctest -LE huge` skips them.
- A single suite can be run directly: `./build/bin/test/fft_tests --bits 64 --suite special --max-logn 16`

#### Benchmarks

Benchmarks use [nvbench](https://github.com/NVIDIA/nvbench), which requires CMake >= 3.30.4. `benchmark_fft` runs every benchmark for both 32-bit and 64-bit data, with one polynomial (`single`) or about 2^22 values (`bulk`; one polynomial from `LogN` 22):
- `fft`: `GPU_FFT` for `LogN` 1–28, forward and inverse, X^N-1 and X^N+1, complex or real input/output.
- `special_fft`: `GPU_Special_FFT` for `LogN` 11–28 (the number of slots).
- `ffnt`: `GPU_FFNT` for `LogN` 12–28 (the number of real coefficients).
- `poly_mult`: polynomial multiplication, the forward transform of both operands and the inverse transform with the fused pointwise product, with `GPU_FFT` (`c2c`, `real`) or `GPU_FFNT` (`ffnt`).

`benchmark_device_api` holds `poly_mul`: polynomial multiplication modulo X^N + 1 with one operand already transformed, a ← IFFT(FFT(a) · b̂), for about 2^22 values and `LogN` 1–10. `Method=fused` runs it in one kernel with the device API, `Method=separate` runs `GPU_FFT` forward and `GPU_FFT` inverse with the fused product.

Configure + build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUFFT_BUILD_BENCHMARKS=ON
cmake --build build --parallel
```

Run benchmarks:

```bash
./build/bin/benchmark/benchmark_fft --disable-blocking-kernel
# Only one benchmark / configuration:
./build/bin/benchmark/benchmark_fft --disable-blocking-kernel -b fft -a Data=F64 -a LogN=16
```

Compare two runs (e.g. before and after a change):

```bash
./build/bin/benchmark/benchmark_fft --disable-blocking-kernel --json before.json
# ... change and rebuild ...
./build/bin/benchmark/benchmark_fft --disable-blocking-kernel --json after.json
python3 benchmark/compare_results.py before.json after.json --threshold 3
```

`compare_results.py` has no dependencies; it lists the configurations whose GPU time changed by more than the threshold and the measured noise, and exits with status 1 if any configuration got slower. The `single` workload takes only a few microseconds and is dominated by launch latency, so its run-to-run noise is high; use `--skip Workload=single` to compare only the stable workloads.

## Using GPU-FFT in a downstream CMake project

Make sure GPU-FFT is installed before integrating it into your project. The installed GPU-FFT library provides a set of config files that make it easy to integrate GPU-FFT into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(GPUFFT 1.0 CONFIG REQUIRED)   # any installed 1.x, at least 1.0
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) GPUFFT::fft CUDA::cudart)
# ...
```

- Neither relocatable device code (`CUDA_SEPARABLE_COMPILATION`, `-rdc`) nor device link-time optimization is needed.
- The version follows the release tags (`v1.0.0`). `find_package(GPUFFT 1.0 ...)` accepts an installed 1.x of at least 1.0 and rejects other major versions; `GPUFFT_VERSION` holds the version found. In code, `gpufft/config.cuh` defines `GPUFFT_VERSION_MAJOR`, `GPUFFT_VERSION_MINOR`, `GPUFFT_VERSION_PATCH` and `GPUFFT_VERSION` (`"1.0.0"`).
- Installed paths and the library name carry `<major>.<minor>` (`include/GPUFFT-1.0`, `lib/cmake/GPUFFT-1.0`, `libfft-1.0.a`), so a patch release replaces the previous one and different minor versions can be installed side by side.
- `GPUFFT_CUDA_ARCHITECTURES` holds the GPU architectures the installed library was compiled for; the installed `gpufft/config.cuh` records them too.

All GPU-FFT types and functions are in the `gpufft` namespace. For compatibility, the complex types (`COMPLEX`, `Complex32`, `Complex64`, `Float32`, `Float64`, ...) and `complex_arithmetic` are also available in the global namespace; define `GPUFFT_NO_GLOBAL_ALIASES` to disable them.

## Host API

`gpufft/fft.cuh` declares the transforms and `gpufft/twiddles.cuh` builds their root tables. All functions work on device memory, run on `cfg.stream` and throw `std::invalid_argument` for an invalid size or batch. `fft_configuration<T>` holds the log2 size (`n_power`), the direction (`fft_type`, `FORWARD` or `INVERSE`), the reduction polynomial (`X_N_minus` for X^N - 1, `X_N_plus` for X^N + 1), the scale of the inverse (`mod_inverse`) and the stream.

| Function | `cfg.n_power` | Data |
|---|---|---|
| `GPU_FFT(data, roots, cfg, batch, multiplication)` | 1–28 | `batch` polynomials of 2^n_power complex values, transformed in place. |
| `GPU_FFT(real, data, roots, cfg, batch, multiplication)` | 1–28 | Forward: real values `real` (zero imaginary parts) to complex values `data`. Inverse: `data` to its real parts in `real`; `data` is overwritten. |
| `GPU_FFNT(real, temp, twist, roots, cfg, batch, multiplication)` | 12–28 | Negacyclic transform of 2^n_power real values ([eprint 2021/480](https://eprint.iacr.org/2021/480)): forward from `real` to 2^(n_power - 1) complex values per polynomial in `temp`, inverse from `temp` to `real`. |
| `GPU_Special_FFT(data, roots, cfg, batch)` | 11–28 | The FFT of the CKKS canonical embedding on 2^n_power slots, in place. |

- The forward transform takes the values in natural order and returns them in bit-reversed order; the inverse takes them in bit-reversed order, returns them in natural order and multiplies them by `cfg.mod_inverse` (1 / 2^n_power to undo the forward transform, 1 / 2^(n_power - 1) for `GPU_FFNT`).
- With `multiplication`, the inverse first multiplies each polynomial element-wise by a second operand that follows the `batch` polynomials (at `data + batch * 2^n_power`, for `GPU_FFNT` at `temp + batch * 2^(n_power - 1)`). A polynomial product is then a forward transform of both operands and one inverse transform.
- `batch` must be positive with `batch * 2^n_power` below 2^31, and at most 65535 for sizes above 2^9 (and always for `GPU_FFNT` and `GPU_Special_FFT`).
- `GPU_FFT` with real data is not a real-to-complex FFT: it transforms the real values as complex values with zero imaginary parts and keeps all 2^n_power outputs.
- `GPU_Special_FFT` runs the loop of the algorithm in `fft.cuh` on the values as given, without its bit reversal step; the inverse with the inverse table and `mod_inverse` = 1 / 2^n_power undoes the forward transform.

The root tables:

| Transform | Host | Entries |
|---|---|---|
| `GPU_FFT` | `make_fft_twiddles_host<T>(n_power, poly, direction)` | 2^(n_power - 1) for X^N - 1, 2^n_power for X^N + 1 (`fft_twiddle_table_size`) |
| `GPU_FFNT` | `make_ffnt_twiddles_host<T>(n_power, direction)`; `make_ffnt_twist_host<T>(n_power, direction)` for `twist` (twist forward, untwist inverse) | 2^(n_power - 2); 2^(n_power - 1) |
| `GPU_Special_FFT` | `make_special_fft_twiddles_host<T>(log_slots, direction)` | 2^log_slots; the table also serves fewer slots |

The functions without `_host` (`make_fft_twiddles(table, n_power, poly, direction, stream)`, ...) write the same tables with a kernel to device memory. The host computes the entries in double precision and rounds them to `T`; the kernel uses `sincospi`, so both agree up to rounding.

```cpp
#include "gpufft/fft.cuh"
#include "gpufft/twiddles.cuh"
using namespace gpufft;

// a: batch polynomials of N = 2^12 complex values in device memory.
std::vector<Complex64> roots = make_fft_twiddles_host<Float64>(12, X_N_plus, FORWARD);
// ... copy roots to device memory (roots_device) ...
fft_configuration<Float64> cfg{12, FORWARD, X_N_plus, Complex64(1.0 / 4096, 0.0), stream};
GPU_FFT(a, roots_device, cfg, batch, false);
```

## Device API

Besides the host functions, GPU-FFT can run the FFT inside your own kernels. Other work on the data, such as a pointwise product, then runs in the same kernel without an extra round trip through global memory. An FFT is described at compile time as a sum of operators:

```cpp
#include "gpufft/descriptor.cuh"
using namespace gpufft;

using FwdFFT = decltype(Size<1024>() + Direction<FORWARD>() + Precision<Float64>() +
                        ReductionPoly<X_N_plus>() + Block());
using InvFFT = decltype(Size<1024>() + Direction<INVERSE>() + Precision<Float64>() +
                        ReductionPoly<X_N_plus>() + Block());
```

| Operator | Required | Meaning |
|---|---|---|
| `Size<N>` | yes | FFT size, a power of two in [2, 2^28]. |
| `Direction<FORWARD>`, `Direction<INVERSE>` | yes | |
| `Precision<Float32>`, `Precision<Float64>` | yes | Type of the real and imaginary parts; the values are `COMPLEX<T>` (`FFT::value_type`). `load` also reads real values and the inverse `store` can write the real parts. |
| `ReductionPoly<X_N_minus>`, `ReductionPoly<X_N_plus>` | for `execute` | X^N - 1 or X^N + 1. |
| `Block` | no | The FFT is computed by a thread block. |
| `BatchesPerBlock<B>` | no | Polynomials per block, only for N up to 2^9. The default gives 256 threads per block. |
| `BlockDim<N / 2, B>` | no | For N up to 2^9: same as `BatchesPerBlock<B>`. |
| `BlockDim<T>` | no | For N from 2^10: T threads in every kernel, T = 128, 256, 512 or 1024 and at most N / 2. The default is 256 (512 for N = 2^10), the kernels `GPU_FFT` runs. |

Each operator may appear once; invalid descriptions fail to compile with a `static_assert` message. The device functions live in headers and are inlined into your kernel, so the device API needs neither relocatable device code (`-rdc`) nor link-time optimization. The results are the same as those of `GPU_FFT`, bit for bit.

### Root tables and 1 / N

A description with `ReductionPoly` builds its root table, the table `GPU_FFT` takes:

```cpp
std::vector<Complex64> roots = FwdFFT::make_twiddles_host();  // FwdFFT::twiddle_table_size entries
FwdFFT::make_twiddles(roots_device, stream);                  // the same table, built by a kernel
Complex64 n_inverse = InvFFT::n_inv();                        // 1 / N, host and device
```

`FFT::twiddle_table_size` is N / 2 for X^N - 1 and N for X^N + 1; the inverse description builds the conjugate table.

### Whole transform: `FFT::launch`

When there is no own work to fuse, a description with `ReductionPoly` runs the whole transform from the host:

```cpp
FwdFFT::launch(in, out, roots, batch, stream);   // stream defaults to 0
```

- `in` holds `batch` polynomials one after another; `out` (which may be `in`) receives the same result as `GPU_FFT` with the same root table. The inverse multiplies by `n_inv()`.
- The forward input may be real (`T*`, zero imaginary parts) and the inverse output may be real (the real parts). An inverse with real output and more than one stage keeps its intermediate values in `in`, like `GPU_FFT`.
- `batch` must be positive with `batch * N` below 2^31, and at most 65535 when the transform has more than one stage; otherwise `launch` throws `std::invalid_argument`.

### Sizes up to 2^10: one kernel

Block `b` transforms the `B = FFT::batches_per_block` polynomials that start at `data + b * B * N`:

```cpp
__global__ void poly_mul(Complex64* a, const Complex64* b_hat, const Complex64* roots,
                         const Complex64* inverse_roots, int batch)
{
    Complex64* shared_memory = FwdFFT::shared_memory();
    Complex64* polys = FwdFFT::block_data(a);           // a + blockIdx.x * B * N
    const Complex64* other = FwdFFT::block_data(b_hat);
    const int num_polys = FwdFFT::num_polys(batch);     // B, fewer in a last block

    FwdFFT().load(shared_memory, polys, num_polys);
    FwdFFT().execute(shared_memory, roots);
    FwdFFT().for_each(shared_memory, [&](Complex64& value, size_t index) {
        value = value * other[index];
    }, num_polys);
    InvFFT().execute(shared_memory, inverse_roots);
    InvFFT().store(shared_memory, polys, InvFFT::n_inv(), num_polys);
}

poly_mul<<<FwdFFT::grid(batch), FwdFFT::block_dim, FwdFFT::shared_memory_size>>>(
    a, b_hat, roots, inverse_roots, batch);
```

- `FFT::shared_memory()` is the kernel's dynamic shared memory; launch with `FFT::shared_memory_size` bytes.
- Shared memory holds the block's polynomials one after another (polynomial `j` at `shared_memory + j * N`), in the same order as in global memory. After the forward `execute` they are in the order of the `GPU_FFT` output.
- `FFT().for_each(shared_memory, f, num_polys)` calls `f(value, index)` for every entry, spread over the threads; `index` is the entry's position in the array `load` reads. It ends with a block barrier.
- `execute` ends with a block barrier. Code that writes shared memory between `execute` and `store` (or another `execute`) without `for_each` must end with `__syncthreads()`.
- The inverse `execute` leaves N · IFFT(x) in shared memory; the inverse `store(shared_memory, data, n_inverse, num_polys)` multiplies by `n_inverse` while writing.
- `num_polys` (at most B) is the number of polynomials of a last block that is not full.

### Larger sizes: one kernel per stage

Larger transforms run as `FFT::num_stages` kernels. Stage kernels transform polynomial `blockIdx.z` of the batch:

```cpp
using BigFFT = decltype(Size<(1 << 16)>() + Direction<FORWARD>() + Precision<Float64>() +
                        ReductionPoly<X_N_plus>() + Block());

template <int I>
__global__ void my_stage(const Complex64* in, Complex64* out, const Complex64* roots)
{
    using Stage = BigFFT::stage<I>;
    Complex64* shared_memory = Stage::shared_memory();

    Stage().load(shared_memory, in);
    Stage().execute(shared_memory, roots);
    Stage().store(shared_memory, out);
}

// Stage 0 reads `in`; every stage writes `out`, which the later stages read and
// write in place.
BigFFT::for_each_stage([&](auto I) {
    using Stage = BigFFT::stage<I>;
    my_stage<I><<<Stage::grid(batch), Stage::block_dim, Stage::shared_memory_size>>>(
        I == 0 ? in : out, out, roots);
});
```

- The last inverse stage (`BigFFT::stage<I>::last`) stores with `store(shared_memory, out, n_inverse)`, as complex values or their real parts.
- A stage holds a strided part of each polynomial in shared memory. `Stage().for_each(shared_memory, f)` calls `f(value, index)` for each entry of the thread, where `index` is the entry's position in `data`, so element-wise work with other arrays can run inside a stage kernel. It ends with a block barrier.
- The last forward stage and the first inverse stage have the same launch parameters and keep the same values in shared memory, so they can share a kernel with element-wise work in between, e.g. for a product a · b with b̂ = FFT(b):

```cpp
using BigInvFFT = decltype(Size<(1 << 16)>() + Direction<INVERSE>() + Precision<Float64>() +
                           ReductionPoly<X_N_plus>() + Block());
using Last = BigFFT::stage<BigFFT::num_stages - 1>;
using First = BigInvFFT::stage<0>;

Complex64* shared_memory = Last::shared_memory();
Last().load(shared_memory, a);
Last().execute(shared_memory, roots);
Last().for_each(shared_memory, [&](Complex64& value, size_t index) {
    value = value * b_hat[index];
});
First().execute(shared_memory, inverse_roots);
First().store(shared_memory, a);  // with n_inverse if First::last
// launched with Last::grid(batch), Last::block_dim, Last::shared_memory_size
```

### Threads per kernel: `BlockDim<T>`

`BlockDim<T>` gives every stage kernel T threads instead of the default, e.g. to match the block size of your own work in the same kernel. Every thread owns two entries, so a kernel holds 2T values in shared memory and runs at most log2(T) + 1 levels: the last forward stage runs log2(T) + 1 levels, the levels before it are spread evenly over as few stages as possible with at most log2(T) - 1 levels each. `num_stages`, `stage<I>`, `for_each_stage`, `launch` and the single kernel API (when `num_stages` is 1, e.g. N = 2^11 with T = 1024) work as without `BlockDim`.

| N | T = 128 | T = 256 | T = 512 | T = 1024 |
|---|---|---|---|---|
| 2^16 | 3 | 2 | 2 | 2 |
| 2^20 | 3 | 3 | 3 | 2 |
| 2^24 | 4 | 4 | 3 | 3 |
| 2^28 | 5 | 4 | 4 | 3 |

### Entries of a thread

Every thread of a single kernel or a stage kernel owns two entries of shared memory: `load` fills them, `store` writes them back, and they keep their positions in between. `FFT().entries(num_polys)` (single kernel) and `Stage().entries()` return them as a `ThreadEntries`: entry `k` (`k < ThreadEntries::count`, which is 2) is `shared_memory[position[k]]` and belongs at `data[index[k]]`, the index that `for_each` passes. In a last single kernel block that is not full, the threads of the missing polynomials have `valid == false`. `store` writes the thread's own entries, so changing them needs no barrier before `store`; before another `execute`, end with `__syncthreads()`.

### Own use of shared memory

`execute` works on any buffer of `shared_memory_size` bytes with the layout of `load`:

- **Without `load` or `store`.** In a single kernel polynomial `j` of the block is at `shared_memory + j * N` in the order of `data`; in a stage kernel write through `for_each` or `entries`. If threads write other threads' entries, call `__syncthreads()` before `execute`.
- **More shared memory.** The FFT uses only the first `shared_memory_size` bytes; launch with more and use the rest from `shared_memory() + shared_memory_size / sizeof(value_type)`. Above 48 KB per block, CUDA needs `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` first.
- **A second buffer.** A second `shared_memory_size` area can hold another polynomial with its own `load` and `execute`; an entry has the same `position` in both buffers.

[`test/test_fft_device.cu`](test/test_fft_device.cu) runs all of these against `GPU_FFT`.

## How to Cite GPU-FFT

Please use the below BibTeX, to cite GPU-FFT in academic papers.

```
@misc{cryptoeprint:2023/1410,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {Two Algorithms for Fast GPU Implementation of NTT},
      howpublished = {Cryptology ePrint Archive, Paper 2023/1410},
      year = {2023},
      note = {\url{https://eprint.iacr.org/2023/1410}},
      url = {https://eprint.iacr.org/2023/1410}
}
```

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)
