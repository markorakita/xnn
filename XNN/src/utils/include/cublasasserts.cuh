// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda cuBLAS assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <cublas_v2.h>

// Macro to check errors in CUDA CUBLAS library operations.
// Asserts in both debug and ship if CUDA CUBLAS library operation failed, can't allow this to happen.
#define CudaCublasAssert(cublasStatus) _CudaCublasAssert(cublasStatus, __FILE__, __FUNCTION__, __LINE__)

using namespace std;

// Asserts in both debug and ship if CUDA CUBLAS library operation failed, can't allow this to happen.
// Should never be called directly, use macro!
void _CudaCublasAssert(cublasStatus_t cublasStatus, const char* file, const char* function, int line);