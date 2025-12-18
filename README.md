# GPU-FFT
Welcome to the GPU-FFT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Fast Fourier Transform (FFT) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

NTT variant of GPU-FFT is available: https://github.com/Alisah-Ozcan/GPU-NTT

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Build & Install

To build:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -B./build
$ cmake --build ./build/ --parallel
```

To install:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -B./build
$ cmake --build ./build/ --parallel
$ sudo cmake --install build
```

### Testing & Benchmarking

#### CPU & GPU FTT Testing & Benchmarking

Choose one of data type which is upper line of the benchmark files:
- typedef Float32 BenchmarkDataType;
- typedef Float64 BenchmarkDataType;

To run examples:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D GPUFFT_BUILD_EXAMPLES=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/example/cpu_fft_example <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/gpu_fft_C_C_example <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/gpu_fft_R_R_example <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/cpu_ffnt_example <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/gpu_ffnt_R_R_example <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/example/gpu_fft_R_R_example 12 1
```

To run benchmarks:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D GPUFFT_BUILD_BENCHMARKS=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/benchmark/gpu_fft_C_C_mult_benchmark --disable-blocking-kernel
$ ./build/bin/benchmark/gpu_fft_R_R_mult_benchmark --disable-blocking-kernel
$ ./build/bin/benchmark/gpu_fft_benchmark --disable-blocking-kernel
$ ./build/bin/benchmark/gpu_ffnt_R_R_mult_benchmark --disable-blocking-kernel
```

## Using GPU-FFT in a downstream CMake project

Make sure GPU-FFT is installed before integrating it into your project. The installed GPU-FFT library provides a set of config files that make it easy to integrate GPU-FFT into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(GPUFFT)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) GPUFFT::fft CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ...
```

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
