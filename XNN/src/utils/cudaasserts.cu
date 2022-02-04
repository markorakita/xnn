// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/cudaasserts.cuh"

#include <iostream>
#include <mutex>

#include <cuda_runtime.h>

#include "include/asserts.cuh"
#include "include/consolehelper.cuh"
#include "include/utils.cuh"

void _CudaAssert(cudaError_t cudaStatus, const char* file, const char* function, int line)
{
	if (cudaStatus != cudaSuccess)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
			cout << endl << "CUDA operation failed with status: " << cudaGetErrorName(cudaStatus) << " (" << cudaGetErrorString(cudaStatus) << ")" << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
			ConsoleHelper::RevertConsoleColors();
		}
		exit(EXIT_FAILURE);
	}
}