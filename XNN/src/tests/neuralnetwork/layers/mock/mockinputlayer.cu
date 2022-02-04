// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network input layer, used in tests.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockinputlayer.cuh"

#include <chrono>

#include <curand_kernel.h>

#include "../../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"
#include "../../../../utils/include/cudahelper.cuh"

MockInputLayer::MockInputLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, curandState* curandStatesBuffer)
{
	// Hack to avoid casting to InputLayer inside layers CommonLoadInputs function.
	m_layerType = LayerType::Standard;
	m_parallelismMode = ParallelismMode::Model;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_deviceCalculationStream = 0;
	m_deviceMemoryStream = 0;
	m_curandStatesBuffer = curandStatesBuffer;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = false;

	m_inputBufferSize = m_activationBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);

	m_holdsActivationGradients = false;
}

void MockInputLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(0));

	// Allocating activation data buffer.
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));
}

void MockInputLayer::GenerateActivationFromUniformDistribution(float rangeStart, float rangeEnd)
{
	const uint activationBufferLength = (uint)(m_activationBufferSize / sizeof(float));
	InitializeBufferFromUniformDistribution(m_activationDataBuffer, activationBufferLength, rangeStart, rangeEnd, m_curandStatesBuffer);
	SynchronizeCalculations();
}

/*
	Initializes buffer with random values sampled from uniform whole integer distribution.
*/
__global__ void InitializeBufferUniformInt(float* buffer, const uint bufferLength, int rangeStart, int rangeEnd, curandState* curandStatesBuffer)
{
	const uint bufferOffset = blockIdx.x * blockDim.x + threadIdx.x;
	const float rangeSpan = rangeEnd - rangeStart + 1.f;

	// Saving state to register for efficiency.
	curandState localCurandState = curandStatesBuffer[bufferOffset];

	for (uint bufferIndex = bufferOffset; bufferIndex < bufferLength; bufferIndex += gridDim.x * blockDim.x)
	{
		buffer[bufferIndex] = rangeStart + ceilf(rangeSpan * curand_uniform(&localCurandState)) - 1.f;
	}

	// Copying state back to global memory.
	// We need to do this since each generation of random number changes the state of the generator.
	curandStatesBuffer[bufferOffset] = localCurandState;
}

void MockInputLayer::GenerateActivationFromUniformIntDistribution(int rangeStart, int rangeEnd)
{
	const uint activationBufferLength = (uint)(m_activationBufferSize / sizeof(float));
	dim3 blockDimensions(NeuralNet::c_numCurandThreadsPerBlock);
	dim3 gridDimensions(NeuralNet::c_numCurandBlocks);
	LAUNCH_KERNEL_ASYNC(InitializeBufferUniformInt, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_activationDataBuffer,
		activationBufferLength, rangeStart, rangeEnd, m_curandStatesBuffer);
	CudaAssert(cudaGetLastError());

	SynchronizeCalculations();
}

void MockInputLayer::GenerateActivationFromNormalDistribution(float mean, float stDev)
{
	const uint activationBufferLength = (uint)(m_activationBufferSize / sizeof(float));
	InitializeBufferFromNormalDistribution(m_activationDataBuffer, activationBufferLength, mean, stDev, m_curandStatesBuffer);
	SynchronizeCalculations();
}