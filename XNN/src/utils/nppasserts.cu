// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda NPP assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/nppasserts.cuh"

#include <iostream>
#include <mutex>

#include "include/asserts.cuh"
#include "include/consolehelper.cuh"
#include "include/utils.cuh"

void _CudaNppAssert(NppStatus nppStatus, const char* file, const char* function, int line)
{
	if (nppStatus != NPP_SUCCESS)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
			cout << endl << "CUDA NPP operation failed with status: " << nppStatus << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
			ConsoleHelper::RevertConsoleColors();
		}
		exit(EXIT_FAILURE);
	}
}