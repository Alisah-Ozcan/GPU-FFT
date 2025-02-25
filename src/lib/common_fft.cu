// Copyright 2023-2025 Alişah Özcan
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

} // namespace gpufft
