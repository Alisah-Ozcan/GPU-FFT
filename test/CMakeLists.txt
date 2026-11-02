# Copyright 2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/test)

set(GPUFFT_TEST_MAX_LOGN 20 CACHE STRING
    "Largest log2 size covered by the GPU-FFT tests")

add_executable(fft_tests test_fft.cu)
target_link_libraries(fft_tests PRIVATE fft CUDA::cudart)
set_target_properties(fft_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

add_executable(descriptor_tests test_descriptor.cu)
target_link_libraries(descriptor_tests PRIVATE fft CUDA::cudart)
set_target_properties(descriptor_tests PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_test(NAME descriptor COMMAND descriptor_tests)

add_executable(fft_device_tests test_fft_device.cu)
target_link_libraries(fft_device_tests PRIVATE fft CUDA::cudart)
set_target_properties(fft_device_tests PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
add_test(NAME fft_device COMMAND fft_device_tests)

foreach(bits 32 64)
    foreach(suite c2c real mult poly_mult special ffnt twiddles validation
                  large)
        add_test(
            NAME fft_${bits}_${suite}
            COMMAND fft_tests --bits ${bits} --suite ${suite}
                    --max-logn ${GPUFFT_TEST_MAX_LOGN}
        )
    endforeach()
    # The large suites use about 2 GB of GPU memory each; run them one at a
    # time.
    set_tests_properties(fft_${bits}_large PROPERTIES
        RESOURCE_LOCK fft_large)
endforeach()
