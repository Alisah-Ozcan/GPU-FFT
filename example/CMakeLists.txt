# Copyright 2023-2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples)

set(EXECUTABLES
    cpu_fft_example test_fft_cpu.cu
    gpu_fft_C_C_example test_fft_gpu_C_C.cu
    gpu_fft_R_R_example test_fft_gpu_R_R.cu
    cpu_ffnt_example test_ffnt_cpu.cu
    gpu_ffnt_R_R_example test_ffnt_gpu_R_R.cu
)

function(add_benchmark exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE fft CUDA::cudart)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
    )
endfunction()

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR EXECUTABLES_COUNT "${EXECUTABLES_LENGTH} / 2")
math(EXPR EXECUTABLES_COUNT_LOOP "${EXECUTABLES_COUNT} - 1")

foreach(i RANGE 0 ${EXECUTABLES_COUNT_LOOP})
    math(EXPR index1 "${i} * 2")
    math(EXPR index2 "${i} * 2 + 1")
    list(GET EXECUTABLES ${index1} exe)
    list(GET EXECUTABLES ${index2} source)
    add_benchmark(${exe} ${source})
endforeach()

add_compile_definitions(${DEFINITION})

foreach(i RANGE 0 ${num_executables}-1)
    math(EXPR index1 "${i} * 2")
    list(GET EXECUTABLES ${index1} exe)
    target_compile_definitions(${exe} PRIVATE ${DEFINITION})
endforeach()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        