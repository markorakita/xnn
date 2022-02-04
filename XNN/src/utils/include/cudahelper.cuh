// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda helper functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel launcher macros.
#ifdef __INTELLISENSE__
    #define LAUNCH_KERNEL(kernel, gridDimensions, blockDimensions) kernel
    #define LAUNCH_KERNEL_ASYNC(kernel, gridDimensions, blockDimensions, stream) kernel
#else
    #define LAUNCH_KERNEL(kernel, gridDimensions, blockDimensions) kernel<<<gridDimensions, blockDimensions>>>
    #define LAUNCH_KERNEL_ASYNC(kernel, gridDimensions, blockDimensions, stream) kernel<<<gridDimensions, blockDimensions, 0, stream>>>
#endif

using namespace std;

// Gets size of available memory on current device, in bytes.
size_t GetSizeOfAvailableGpuMemory();