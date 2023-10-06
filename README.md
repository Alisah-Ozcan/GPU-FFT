# GPU-FFT
Welcome to the GPU-FFT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Fast Fourier Transform (FFT) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Testing & Benchmarking

Four different float data type supported. They represented as numbers:

- COPLEX_DATA_TYPE=0 -> THRUST_FLOAT_32(32 bit)
- COPLEX_DATA_TYPE=1 -> THRUST_FLOAT_64(64 bit)
- COPLEX_DATA_TYPE=2 -> FLOAT_32(32 bit)
- COPLEX_DATA_TYPE=3 -> FLOAT_64(64 bit)


#### Testing CPU & GPU FTT with Schoolbook Polynomial Multiplication

To build tests:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./cmake-build
$ cmake --build ./cmake-build/ --target fft_test --parallel
```

To run tests:

```bash
$ ./cmake-build/fft_test <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./cmake-build/fft_test 12 1
```

#### Benchmarking GPU FFT

To build tests:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./cmake-build
$ cmake --build ./cmake-build-debug/ --target fft_bench --parallel
```

To run tests:

```bash
$ ./cmake-build-debug/fft_bench <RING_SIZE_IN_LOG2> <BATCH_SIZE>
```

#### Benchmarking Polynomial Multiplication with using FFT

To build tests:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./cmake-build
$ cmake --build ./cmake-build-debug/ --target polymult_bench --parallel
```

To run tests:

```bash
$ ./cmake-build-debug/polymult_bench <RING_SIZE_IN_LOG2> <BATCH_SIZE>
```
