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
find_package(GPUFFT CONFIG REQUIRED)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) GPUFFT::fft CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ...
```

All GPU-FFT types and functions are in the `gpufft` namespace. For compatibility, the complex types (`COMPLEX`, `Complex32`, `Complex64`, `Float32`, `Float64`, ...) and `complex_arithmetic` are also available in the global namespace; define `GPUFFT_NO_GLOBAL_ALIASES` to disable them.

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
