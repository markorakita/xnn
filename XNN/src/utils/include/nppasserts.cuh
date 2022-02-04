// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda NPP assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <nppdefs.h>

// Macro to check errors in CUDA NPP library operations.
// Asserts in both debug and ship if CUDA NPP library operation failed, can't allow this to happen.
#define CudaNppAssert(nppStatus) _CudaNppAssert(nppStatus, __FILE__, __FUNCTION__, __LINE__)

using namespace std;

// Asserts in both debug and ship if CUDA NPP library operation failed, can't allow this to happen.
// Should never be called directly, use macro!
void _CudaNppAssert(NppStatus nppStatus, const char* file, const char* function, int line);