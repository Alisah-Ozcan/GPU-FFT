// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common_fft.cuh"

namespace gpufft
{

    void customAssert(bool condition, const std::string& errorMessage)
    {
        if (!condition)
        {
            std::cerr << "Custom assertion failed: " << errorMessage
                      << std::endl;
            assert(condition);
        }
    }

    float calculate_mean(const float array[], int size)
    {
        float sum = 0.0;
        for (int i = 0; i < size; ++i)
        {
            sum += array[i];
        }
        return sum / size;
    }

    float calculate_standard_deviation(const float array[], int size)
    {
        float mean = calculate_mean(array, size);
        float sum_squared_diff = 0.0;

        for (int i = 0; i < size; ++i)
        {
            float diff = array[i] - mean;
            sum_squared_diff += diff * diff;
        }

        float variance = sum_squared_diff / size;
        return std::sqrt(variance);
    }

    float find_best_average(const float array[], int array_size,
                            int num_elements)
    {
        if (num_elements <= 0 || num_elements > array_size)
        {
            std::cerr << "Invalid number of elements." << std::endl;
            return 0.0;
        }

        float max_average = 0.0;

        for (int i = 0; i <= array_size - num_elements; ++i)
        {
            float sum = 0.0;
            for (int j = i; j < i + num_elements; ++j)
            {
                sum += array[j];
            }
            float average = sum / num_elements;
            max_average = std::max(max_average, average);
        }

        return max_average;
    }

    float find_min_average(const float array[], int array_size,
                           int num_elements)
    {
        if (num_elements <= 0 || num_elements > array_size)
        {
            std::cerr << "Invalid number of elements." << std::endl;
            return 0.0;
        }

        float min_average = std::numeric_limits<float>::max();

        for (int i = 0; i <= array_size - num_elements; ++i)
        {
            float sum = 0.0;
            for (int j = i; j < i + num_elements; ++j)
            {
                sum += array[j];
            }
            float average = sum / num_elements;
            min_average = std::min(min_average, average);
        }

        return min_average;
    }

    __global__ void GPU_ACTIVITY(unsigned long long* output,
                                 unsigned long long fix_num)
    {
        int idx = blockIdx.x + blockDim.x + threadIdx.x;

        output[idx] = fix_num;
    }

    __host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                    unsigned long long fix_num)
    {
        GPU_ACTIVITY<<<64, 512>>>(output, fix_num);
    }
} // namespace gpufft
