# Copyright 2023-2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(LIB_SOURCE_FILES
    lib/fft.cu
    lib/fft_cpu.cu
)

set(LIB_HEADER_FILES
    include/gpufft/common_fft.cuh
    include/gpufft/complex.cuh
    include/gpufft/descriptor.cuh
    include/gpufft/fft.cuh
    include/gpufft/fft_device.cuh
    include/gpufft/fft_cpu.cuh
    include/gpufft/twiddles.cuh
)

add_library(fft STATIC ${LIB_SOURCE_FILES})
add_library(GPUFFT::fft ALIAS fft)
if(CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(fft PUBLIC cxx_std_17)
else()
    message(WARNING
        "CMake does not know C++ compile features for ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION}; falling back to -std=c++17"
    )
    target_compile_options(fft PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
endif()

if(CMAKE_CUDA_COMPILE_FEATURES)
    target_compile_features(fft PRIVATE cuda_std_17)
else()
    target_compile_options(fft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
endif()
target_compile_options(fft PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -g -DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    $<$<CONFIG:Debug>:-g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
)
option(GPUFFT_ENABLE_WARNINGS "Enable extra warnings for GPUFFT" OFF)
if(GPUFFT_ENABLE_WARNINGS)
    target_compile_options(fft PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()

target_link_libraries(fft PUBLIC CUDA::cudart)

# gpufft/config.cuh holds the version and the architectures for code that uses
# the headers.
set(GPUFFT_CONFIG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
configure_file(${PROJECT_SOURCE_DIR}/cmake/config.cuh.in
               ${GPUFFT_CONFIG_INCLUDE_DIR}/gpufft/config.cuh)

set_target_properties(fft PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  OUTPUT_NAME "fft-${GPUFFT_VERSION_DIR}"
)

target_include_directories(
    fft
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${GPUFFT_CONFIG_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include/GPUFFT-${GPUFFT_VERSION_DIR}>
)

set_target_properties(fft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

install(TARGETS fft
  EXPORT ${GPUFFT_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${INCLUDES_INSTALL_DIR}
)
install(
    FILES ${GPUFFT_CONFIG_INCLUDE_DIR}/gpufft/config.cuh
    DESTINATION ${INCLUDES_INSTALL_DIR}/gpufft
)
