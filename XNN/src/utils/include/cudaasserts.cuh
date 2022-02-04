// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <driver_types.h>

// Macro to check errors in CUDA operations.
// Asserts in both debug and ship if CUDA operation failed, can't allow this to happen.
#define CudaAssert(cudaStatus) _CudaAssert(cudaStatus, __FILE__, __FUNCTION__, __LINE__)

using namespace std;

// Asserts in both debug and ship if CUDA operation failed, can't allow this to happen.
// Should never be called directly, use macro!
void _CudaAssert(cudaError_t cudaStatus, const char* file, const char* function, int line);