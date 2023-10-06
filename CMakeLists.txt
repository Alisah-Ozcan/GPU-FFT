cmake_minimum_required(VERSION 3.2)

set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(FFT_Library LANGUAGES C CXX CUDA ASM)

if(NOT DEFINED COPLEX_DATA_TYPE)
    set(COPLEX_DATA_TYPE 0)
endif()

# FFT_Library
add_library(FFT_Library src/fft.cu src/common.cu src/fft_cpu.cu)

if(NOT "${CMAKE_CUDA_BYTE_ORDER}" STREQUAL "LITTLE_ENDIAN")
    message(FATAL_ERROR "Target architecture must be little-endian")
endif()

set_target_properties(FFT_Library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------- #

add_executable(fft_test test/test_fft.cu)
add_executable(fft_bench test/bench_fft.cu)
add_executable(polymult_bench test/bench_polymult.cu)


if(COPLEX_DATA_TYPE EQUAL 0)
    add_compile_definitions(THRUST_FLOAT_32)
    target_compile_definitions(fft_test PRIVATE THRUST_FLOAT_32)
    target_compile_definitions(fft_bench PRIVATE THRUST_FLOAT_32)
    target_compile_definitions(polymult_bench PRIVATE THRUST_FLOAT_32)
elseif(COPLEX_DATA_TYPE EQUAL 1)
    add_compile_definitions(THRUST_FLOAT_64)
    target_compile_definitions(fft_test PRIVATE THRUST_FLOAT_64)
    target_compile_definitions(fft_bench PRIVATE THRUST_FLOAT_64)
    target_compile_definitions(polymult_bench PRIVATE THRUST_FLOAT_64)
elseif(COPLEX_DATA_TYPE EQUAL 2)
    add_compile_definitions(FLOAT_32)
    target_compile_definitions(fft_test PRIVATE FLOAT_32)
    target_compile_definitions(fft_bench PRIVATE FLOAT_32)
    target_compile_definitions(polymult_bench PRIVATE FLOAT_32)
elseif(COPLEX_DATA_TYPE EQUAL 3)
    add_compile_definitions(FLOAT_64)
    target_compile_definitions(fft_test PRIVATE FLOAT_64)
    target_compile_definitions(fft_bench PRIVATE FLOAT_64)
    target_compile_definitions(polymult_bench PRIVATE FLOAT_64)
else()
    message(FATAL_ERROR "Please use valid COPLEX_DATA_TYPE! (0,1,2,3)")
endif()

target_link_libraries(fft_test PRIVATE FFT_Library)
set_target_properties(fft_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fft_bench PRIVATE FFT_Library)
set_target_properties(fft_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(polymult_bench PRIVATE FFT_Library)
set_target_properties(polymult_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
        
# ---------------------------------------------------------------------------- #

