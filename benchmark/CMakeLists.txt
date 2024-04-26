set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION})

if(NOT DEFINED COPLEX_DATA_TYPE)
    set(COPLEX_DATA_TYPE 0)
endif()

add_executable(benchmark_fft bench_fft.cu)
add_executable(benchmark_polymult bench_polymult.cu)

if(COPLEX_DATA_TYPE EQUAL 0)
    add_compile_definitions(THRUST_FLOAT_32)
    target_compile_definitions(benchmark_fft PRIVATE THRUST_FLOAT_32)
    target_compile_definitions(benchmark_polymult PRIVATE THRUST_FLOAT_32)
elseif(COPLEX_DATA_TYPE EQUAL 1)
    add_compile_definitions(THRUST_FLOAT_64)
    target_compile_definitions(benchmark_fft PRIVATE THRUST_FLOAT_64)
    target_compile_definitions(benchmark_polymult PRIVATE THRUST_FLOAT_64)
elseif(COPLEX_DATA_TYPE EQUAL 2)
    add_compile_definitions(FLOAT_32)
    target_compile_definitions(benchmark_fft PRIVATE FLOAT_32)
    target_compile_definitions(benchmark_polymult PRIVATE FLOAT_32)
elseif(COPLEX_DATA_TYPE EQUAL 3)
    add_compile_definitions(FLOAT_64)
    target_compile_definitions(benchmark_fft PRIVATE FLOAT_64)
    target_compile_definitions(benchmark_polymult PRIVATE FLOAT_64)
else()
    message(FATAL_ERROR "Please use valid COPLEX_DATA_TYPE! (0,1,2,3)")
endif()

target_link_libraries(benchmark_fft PRIVATE fft)

set_target_properties(benchmark_fft PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(benchmark_fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(benchmark_polymult PRIVATE fft)

set_target_properties(benchmark_polymult PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(benchmark_polymult PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)