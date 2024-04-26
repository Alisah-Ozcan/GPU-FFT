set(LIB_SOURCE_FILES
    common_fft.cu
    fft_cpu.cu
    fft.cu
)

add_library(fft STATIC ${LIB_SOURCE_FILES})

if(NOT DEFINED COPLEX_DATA_TYPE)
    set(COPLEX_DATA_TYPE 0)
endif()

if(COPLEX_DATA_TYPE EQUAL 0)
    add_compile_definitions(THRUST_FLOAT_32)
    target_compile_definitions(fft PRIVATE THRUST_FLOAT_32)
elseif(COPLEX_DATA_TYPE EQUAL 1)
    add_compile_definitions(THRUST_FLOAT_64)
    target_compile_definitions(fft PRIVATE THRUST_FLOAT_64)
elseif(COPLEX_DATA_TYPE EQUAL 2)
    add_compile_definitions(FLOAT_32)
    target_compile_definitions(fft PRIVATE FLOAT_32)
elseif(COPLEX_DATA_TYPE EQUAL 3)
    add_compile_definitions(FLOAT_64)
    target_compile_definitions(fft PRIVATE FLOAT_64)
else()
    message(FATAL_ERROR "Please use valid COPLEX_DATA_TYPE! (0,1,2,3)")
endif()

set_target_properties(fft PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(fft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)