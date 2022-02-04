// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda cuBLAS assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/cublasasserts.cuh"

#include <iostream>
#include <mutex>

#include "include/asserts.cuh"
#include "include/consolehelper.cuh"
#include "include/utils.cuh"

void _CudaCublasAssert(cublasStatus_t cublasStatus, const char* file, const char* function, int line)
{
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
			cout << endl << "CUDA CUBLAS operation failed with status: " << cublasStatus << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
			ConsoleHelper::RevertConsoleColors();
		}
		exit(EXIT_FAILURE);
	}
}