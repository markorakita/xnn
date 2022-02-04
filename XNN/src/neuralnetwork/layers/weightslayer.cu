// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract neural network layer with weights.
// Created: 01/22/2021.
// ----------------------------------------------------------------------------------------------------

#include "include/weightslayer.cuh"

#include "../include/activationfunctions.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

WeightsLayer::WeightsLayer(uint indexInTier, size_t weightsBufferSize, uint numWeightsPerNeuron, float weightsUpdateMomentum, float weightsUpdateDecay,
	float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor,
	size_t biasesBufferSize, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
	float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType, float activationAlpha,
	curandState* curandStatesBuffer)
{
	m_indexInTier = indexInTier;

	m_weightsUpdateMomentum = weightsUpdateMomentum;
	m_weightsUpdateDecay = weightsUpdateDecay;
	m_weightsUpdateLearningRateProgressStep = weightsUpdateLearningRateProgressStep;
	m_weightsUpdateStartingLearningRate = weightsUpdateStartingLearningRate;
	m_weightsUpdateLearningRateUpdateFactor = weightsUpdateLearningRateUpdateFactor;

	m_biasesUpdateMomentum = biasesUpdateMomentum;
	m_biasesUpdateDecay = biasesUpdateDecay;
	m_biasesUpdateLearningRateProgressStep = biasesUpdateLearningRateProgressStep;
	m_biasesUpdateStartingLearningRate = biasesUpdateStartingLearningRate;
	m_biasesUpdateLearningRateUpdateFactor = biasesUpdateLearningRateUpdateFactor;

	m_weightsBufferSize = weightsBufferSize;
	m_biasesBufferSize = biasesBufferSize;

	m_numWeightsPerNeuron = numWeightsPerNeuron;

	m_curandStatesBuffer = curandStatesBuffer;
	m_activationType = activationType;
	m_activationAlpha = activationAlpha;

	m_weightsBuffer = NULL;
	m_weightsGradientsBuffer = NULL;
	m_weightsUpdateBuffer = NULL;
	m_biasesBuffer = NULL;
	m_biasesGradientsBuffer = NULL;
	m_biasesUpdateBuffer = NULL;
}

void WeightsLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(m_indexInTier));

	// Allocating weights buffers.
	CudaAssert(cudaMalloc<float>(&m_weightsBuffer, m_weightsBufferSize));
	m_memoryConsumptionSize += m_weightsBufferSize;

	// Allocating biases buffer.
	CudaAssert(cudaMalloc<float>(&m_biasesBuffer, m_biasesBufferSize));
	m_memoryConsumptionSize += m_biasesBufferSize;

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating weights gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_weightsGradientsBuffer, m_weightsBufferSize));
		m_memoryConsumptionSize += m_weightsBufferSize;

		// Allocating weights update buffer.
		CudaAssert(cudaMalloc<float>(&m_weightsUpdateBuffer, m_weightsBufferSize));
		m_memoryConsumptionSize += m_weightsBufferSize;
		InitializeBufferToConstant(m_weightsUpdateBuffer, (uint)(m_weightsBufferSize / sizeof(float)), 0.f);

		// Allocating biases gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_biasesGradientsBuffer, m_biasesBufferSize));
		m_memoryConsumptionSize += m_biasesBufferSize;

		// Allocating biases update buffer.
		CudaAssert(cudaMalloc<float>(&m_biasesUpdateBuffer, m_biasesBufferSize));
		m_memoryConsumptionSize += m_biasesBufferSize;
		InitializeBufferToConstant(m_biasesUpdateBuffer, (uint)(m_biasesBufferSize / sizeof(float)), 0.f);
	}

	// Because buffers initialization is done on calculations stream.
	SynchronizeCalculations();

	CudaAssert(cudaSetDevice(0));
}

WeightsLayer::~WeightsLayer()
{
	if (m_weightsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_weightsBuffer));
	}
	if (m_weightsGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_weightsGradientsBuffer));
	}
	if (m_weightsUpdateBuffer != NULL)
	{
		CudaAssert(cudaFree(m_weightsUpdateBuffer));
	}
	if (m_biasesBuffer != NULL)
	{
		CudaAssert(cudaFree(m_biasesBuffer));
	}
	if (m_biasesGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_biasesGradientsBuffer));
	}
	if (m_biasesUpdateBuffer != NULL)
	{
		CudaAssert(cudaFree(m_biasesUpdateBuffer));
	}
}

void WeightsLayer::InitializeWeightsToConstant(float initialValue)
{
	InitializeBufferToConstant(m_weightsBuffer, (uint)(m_weightsBufferSize / sizeof(float)), initialValue);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeWeightsFromUniformDistribution(float rangeStart, float rangeEnd)
{
	InitializeBufferFromUniformDistribution(m_weightsBuffer, (uint)(m_weightsBufferSize / sizeof(float)), rangeStart, rangeEnd, m_curandStatesBuffer);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeWeightsFromNormalDistribution(float mean, float stDev)
{
	InitializeBufferFromNormalDistribution(m_weightsBuffer, (uint)(m_weightsBufferSize / sizeof(float)), mean, stDev, m_curandStatesBuffer);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeWeightsXavier()
{
	float stDev = (float)sqrt(6.0 / (m_numWeightsPerNeuron + m_activationNumChannels * m_activationDataSize));

	InitializeBufferFromNormalDistribution(m_weightsBuffer, (uint)(m_weightsBufferSize / sizeof(float)), 0.f, stDev, m_curandStatesBuffer);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeWeightsHe()
{
	// TODO: support also ELU and LeakyReLU here
	float stDev = (float)sqrt((m_activationType == ActivationType::ReLU ? 2.0 : 1.0) / m_numWeightsPerNeuron);

	InitializeBufferFromNormalDistribution(m_weightsBuffer, (uint)(m_weightsBufferSize / sizeof(float)), 0.f, stDev, m_curandStatesBuffer);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeBiasesToConstant(float initialValue)
{
	InitializeBufferToConstant(m_biasesBuffer, (uint)(m_biasesBufferSize / sizeof(float)), initialValue);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeBiasesFromUniformDistribution(float rangeStart, float rangeEnd)
{
	InitializeBufferFromUniformDistribution(m_biasesBuffer, (uint)(m_biasesBufferSize / sizeof(float)), rangeStart, rangeEnd, m_curandStatesBuffer);

	SynchronizeCalculations();
}

void WeightsLayer::InitializeBiasesFromNormalDistribution(float mean, float stDev)
{
	InitializeBufferFromNormalDistribution(m_biasesBuffer, (uint)(m_biasesBufferSize / sizeof(float)), mean, stDev, m_curandStatesBuffer);

	SynchronizeCalculations();
}

void WeightsLayer::CopyWeightsFromHost(float* hostWeightsBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_weightsBuffer, hostWeightsBuffer, m_weightsBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void WeightsLayer::CopyWeightsUpdateFromHost(float* hostWeightsUpdateBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_weightsUpdateBuffer, hostWeightsUpdateBuffer, m_weightsBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void WeightsLayer::CopyBiasesFromHost(float* hostBiasesBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_biasesBuffer, hostBiasesBuffer, m_biasesBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void WeightsLayer::CopyBiasesUpdateFromHost(float* hostBiasesUpdateBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_biasesUpdateBuffer, hostBiasesUpdateBuffer, m_biasesBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void WeightsLayer::CopyWeightsFromLayer(WeightsLayer* standardLayer)
{
	if (m_indexInTier == standardLayer->GetIndexInTier())
	{
		CudaAssert(cudaMemcpyAsync(m_weightsBuffer, standardLayer->GetWeightsBuffer(), m_weightsBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
	}
	else
	{
		CudaAssert(cudaMemcpyPeerAsync(m_weightsBuffer, m_indexInTier, standardLayer->GetWeightsBuffer(), standardLayer->GetIndexInTier(),
			m_weightsBufferSize, m_deviceMemoryStream));
	}
	SynchronizeMemoryOperations();
}

void WeightsLayer::CopyBiasesFromLayer(WeightsLayer* standardLayer)
{
	if (m_indexInTier == standardLayer->GetIndexInTier())
	{
		CudaAssert(cudaMemcpyAsync(m_biasesBuffer, standardLayer->GetBiasesBuffer(), m_biasesBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
	}
	else
	{
		CudaAssert(cudaMemcpyPeerAsync(m_biasesBuffer, m_indexInTier, standardLayer->GetBiasesBuffer(), standardLayer->GetIndexInTier(),
			m_biasesBufferSize, m_deviceMemoryStream));
	}
	SynchronizeMemoryOperations();
}

/*
	Updates layer parameters by applying momentum to last update, learning rate to gradients, and decay to parameters.
*/
__global__ void ApplyParamatersUpdate(float* paramsBuffer, float* gradientsBuffer, float* updatesBuffer, uint numElements,
	float updateMomentum, float learningRate, float updateDecay)
{
	for (uint elementIndex = blockIdx.x * blockDim.x + threadIdx.x; elementIndex < numElements; elementIndex += gridDim.x * blockDim.x)
	{
		updatesBuffer[elementIndex] = updateMomentum * updatesBuffer[elementIndex] + learningRate * (gradientsBuffer[elementIndex] -
			updateDecay * paramsBuffer[elementIndex]);
		paramsBuffer[elementIndex] += updatesBuffer[elementIndex];
	}
}

void WeightsLayer::UpdateLayerParameters(float learningProgress)
{
	// Updating weights.
	const uint numBlocks = 128;
	const uint numThreadsPerBlock = 128;
	float weightsUpdateProgressSteps = floorf(learningProgress / m_weightsUpdateLearningRateProgressStep);
	const float weightsLearningRate = m_weightsUpdateStartingLearningRate * powf(m_weightsUpdateLearningRateUpdateFactor, weightsUpdateProgressSteps);
	const uint weightsGradientsBufferLength = (uint)(m_weightsBufferSize / sizeof(float));
	dim3 blockDimensions(numThreadsPerBlock);
	dim3 gridDimensions(min(numBlocks, DivideUp(weightsGradientsBufferLength, numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(ApplyParamatersUpdate, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_weightsBuffer, m_weightsGradientsBuffer,
		m_weightsUpdateBuffer, weightsGradientsBufferLength, m_weightsUpdateMomentum, weightsLearningRate, m_weightsUpdateDecay);
	CudaAssert(cudaGetLastError());

	// Updating biases.
	float biasesUpdateProgressSteps = floorf(learningProgress / m_biasesUpdateLearningRateProgressStep);
	const float biasesLearningRate = m_biasesUpdateStartingLearningRate * powf(m_biasesUpdateLearningRateUpdateFactor, biasesUpdateProgressSteps);
	const uint biasesGradientsBufferLength = (uint)(m_biasesBufferSize / sizeof(float));
	blockDimensions = dim3(numThreadsPerBlock);
	gridDimensions = dim3(min(numBlocks, DivideUp(biasesGradientsBufferLength, numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(ApplyParamatersUpdate, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_biasesBuffer, m_biasesGradientsBuffer,
		m_biasesUpdateBuffer, biasesGradientsBufferLength, m_biasesUpdateMomentum, biasesLearningRate, m_biasesUpdateDecay);
	CudaAssert(cudaGetLastError());

	SynchronizeCalculations();
}