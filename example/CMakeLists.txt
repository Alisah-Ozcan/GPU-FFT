set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION})

if(NOT DEFINED COPLEX_DATA_TYPE)
    set(COPLEX_DATA_TYPE 0)
endif()

set(EXECUTABLES
    cpu_fft_examples test_fft_cpu.cu
    gpu_fft_examples test_fft_gpu.cu
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

if(COPLEX_DATA_TYPE EQUAL 0)
    set(DEFINITION FLOAT_64)
elseif(COPLEX_DATA_TYPE EQUAL 1)
    set(DEFINITION FLOAT_32)
else()
    message(FATAL_ERROR "Please use valid COPLEX_DATA_TYPE! (0,1)")
endif()

add_compile_definitions(${DEFINITION})

foreach(i RANGE 0 ${num_executables}-1)
    math(EXPR index1 "${i} * 2")
    list(GET EXECUTABLES ${index1} exe)
    target_compile_definitions(${exe} PRIVATE ${DEFINITION})
endforeach()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        