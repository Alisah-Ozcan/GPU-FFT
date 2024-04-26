# GPU-FFT
Welcome to the GPU-FFT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Fast Fourier Transform (FFT) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

NTT variant of GPU-FFT is available: https://github.com/Alisah-Ozcan/GPU-NTT

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [GCC](https://gcc.gnu.org/)
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
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To run tests:

```bash
$ ./build/bin/gpu_fft_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/gpu_fft_examples 12 1
```

#### Benchmarking GPU FFT

To build tests:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To run tests:

```bash
$ ./build/bin/benchmark_fft <RING_SIZE_IN_LOG2> <BATCH_SIZE>
```

#### Benchmarking Polynomial Multiplication with using FFT

To build tests:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To run tests:

```bash
$ ./build/bin/benchmark_polymult <RING_SIZE_IN_LOG2> <BATCH_SIZE>
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