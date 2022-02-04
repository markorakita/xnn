// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network activation functions.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/activationfunctions.cuh"

#include "../utils/include/asserts.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/cudahelper.cuh"
#include "../utils/include/utils.cuh"

/*
	ReLU activations are calculated as: activation = max(0, preactivation)
*/
__global__ void ApplyReLUActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex] >= 0.0f ? preactivations[activationIndex] : 0.0f;
	}
}

/*
	ELU activations are calculated as: activation = max(0, preactivation) + min(0, alpha * (exp(preactivation) - 1))
*/
__global__ void ApplyELUActivation(float* preactivations, uint numPreactivations, float* activations, float activationAlpha)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex] >= 0.0f ? preactivations[activationIndex] :
			activationAlpha * (__expf(preactivations[activationIndex]) - 1.f);
	}
}

/*
	LeakyReLU activations are calculated as: activation = max(0, preactivation) + min (0, alpha * preactivation)
*/
__global__ void ApplyLeakyReLUActivation(float* preactivations, uint numPreactivations, float* activations, float activationAlpha)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex] >= 0.0f ? preactivations[activationIndex] : activationAlpha * preactivations[activationIndex];
	}
}

/*
	Sigmoid activations are calculated as: activation = 1 / (1 + exp(-preactivation))
*/
__global__ void ApplySigmoidActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex] >= 0.f ? __fdividef(1.0f, 1.0f + __expf(-preactivations[activationIndex])) :
			(1.f - __fdividef(1.0f, 1.0f + __expf(preactivations[activationIndex])));
	}
}

/*
	Tanh activations are calculated as: activation = tanh(preactivation)
	= (exp(preactivation) - exp(-preactivation)) / (exp(preactivation) + exp(-preactivation))
	= (exp(2 * preactivation) - 1) / (exp(2 * preactivation) + 1)
	= (1 - exp(-2 * preactivation)) / (1 + exp(-2 * preactivation))
*/
__global__ void ApplyTanhActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex] >= 0.f ? (__fdividef(2.0f, 1.0f + __expf(-2.f * preactivations[activationIndex])) - 1.f) :
			(1.f - __fdividef(2.0f, 1.0f + __expf(2.f * preactivations[activationIndex])));
	}
}

void ApplyActivation(ActivationType activationType, float activationAlpha, float* preactivations, uint numPreactivations, float* activations,
	cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;

	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numPreactivations, c_numThreadsPerBlock)));
	if (activationType == ActivationType::Linear)
	{
		// Linear activations are calculated as: activation = preactivation
		CudaAssert(cudaMemcpyAsync(activations, preactivations, numPreactivations * sizeof(float), cudaMemcpyDeviceToDevice, deviceCalculationStream));
	}
	else if (activationType == ActivationType::ReLU)
	{
		LAUNCH_KERNEL_ASYNC(ApplyReLUActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else if (activationType == ActivationType::ELU)
	{
		LAUNCH_KERNEL_ASYNC(ApplyELUActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations,
			activationAlpha);
	}
	else if (activationType == ActivationType::LeakyReLU)
	{
		LAUNCH_KERNEL_ASYNC(ApplyLeakyReLUActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations,
			activationAlpha);
	}
	else if (activationType == ActivationType::Sigmoid)
	{
		LAUNCH_KERNEL_ASYNC(ApplySigmoidActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else if (activationType == ActivationType::Tanh)
	{
		LAUNCH_KERNEL_ASYNC(ApplyTanhActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else
	{
		ShipAssert(false, "Unknown activation type!");
	}
	CudaAssert(cudaGetLastError());
}

__global__ void CalculateReLUActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activations[activationIndex] > 0.0f ? activationGradients[activationIndex] : 0.0f;
	}
}

__global__ void CalculateELUActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients,
	float activationAlpha)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activations[activationIndex] > 0.0f ? activationGradients[activationIndex] :
			activationGradients[activationIndex] * (activations[activationIndex] + activationAlpha);
	}
}

__global__ void CalculateLeakyReLUActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients,
	float activationAlpha)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activations[activationIndex] > 0.0f ? activationGradients[activationIndex] :
			activationGradients[activationIndex] * activationAlpha;
	}
}

__global__ void CalculateSigmoidActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activationGradients[activationIndex] * activations[activationIndex] * (1.0f - activations[activationIndex]);
	}
}

__global__ void CalculateTanhActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activationGradients[activationIndex] * (1.0f - activations[activationIndex] * activations[activationIndex]);
	}
}

void CalculatePreactivationGradients(ActivationType activationType, float activationAlpha, float* activationGradients, float* activations, uint numActivations,
	float* preactivationGradients, cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;

	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numActivations, c_numThreadsPerBlock)));
	if (activationType == ActivationType::Linear)
	{
		CudaAssert(cudaMemcpyAsync(preactivationGradients, activationGradients, numActivations * sizeof(float), cudaMemcpyDeviceToDevice, deviceCalculationStream));
	}
	else if (activationType == ActivationType::ReLU)
	{
		LAUNCH_KERNEL_ASYNC(CalculateReLUActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else if (activationType == ActivationType::ELU)
	{
		LAUNCH_KERNEL_ASYNC(CalculateELUActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients, activationAlpha);
	}
	else if (activationType == ActivationType::LeakyReLU)
	{
		LAUNCH_KERNEL_ASYNC(CalculateLeakyReLUActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients, activationAlpha);
	}
	else if (activationType == ActivationType::Sigmoid)
	{
		LAUNCH_KERNEL_ASYNC(CalculateSigmoidActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else if (activationType == ActivationType::Tanh)
	{
		LAUNCH_KERNEL_ASYNC(CalculateTanhActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else
	{
		ShipAssert(false, "Unknown activation type!");
	}
	CudaAssert(cudaGetLastError());
}