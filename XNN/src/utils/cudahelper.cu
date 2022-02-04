// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Cuda helper functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/cudahelper.cuh"

#include "include/cudaasserts.cuh"

size_t GetSizeOfAvailableGpuMemory()
{
	size_t free, total;
	CudaAssert(cudaMemGetInfo(&free, &total));

	return free;
}