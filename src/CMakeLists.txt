# Copyright 2023-2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(LIB_SOURCE_FILES
    lib/common_fft.cu
    lib/fft.cu
    lib/fft_cpu.cu
)

set(LIB_HEADER_FILES
    include/common_fft.cuh
    include/complex.cuh
    include/fft.cuh
    include/fft_cpu.cuh
)

add_library(fft STATIC ${LIB_SOURCE_FILES})
add_library(GPUFFT::fft ALIAS fft)
target_compile_features(fft PUBLIC cxx_std_17)
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

set_target_properties(fft PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  OUTPUT_NAME "fft-${PROJECT_VERSION}"
)

target_include_directories(
    fft
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/GPUFFT-${PROJECT_VERSION}>
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
