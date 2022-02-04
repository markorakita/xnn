// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/dropoutlayer.cuh"

#include "../include/matrixoperations.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

DropoutLayer::DropoutLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, curandState* curandStatesBuffer,
	uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, float dropProbability,
	bool useHostDropoutFilter, bool holdsActivationGradients)
{
	m_layerType = LayerType::Dropout;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = indexInTier;
	m_tierSize = tierSize;
	m_curandStatesBuffer = curandStatesBuffer;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_inputBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = m_inputBufferSize;

	m_dropProbability = dropProbability;
	m_useHostDropoutFilter = useHostDropoutFilter;
	m_dropoutFilterSize = m_inputBufferSize;

	m_holdsActivationGradients = holdsActivationGradients;

	m_dropoutFilter = NULL;
}

void DropoutLayer::Reinitialize(uint newInputDataCount)
{
	Layer::Reinitialize(newInputDataCount);

	m_dropoutFilterSize = m_inputBufferSize;
}

void DropoutLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(m_indexInTier));

	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;
	}

	// Allocating activation data buffers.
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));
	m_memoryConsumptionSize += m_activationBufferSize;

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;

		// Allocating dropout filter buffer.
		CudaAssert(cudaMalloc<float>(&m_dropoutFilter, m_dropoutFilterSize));
		m_memoryConsumptionSize += m_dropoutFilterSize;

		// Allocating activation gradients buffer.
		if (m_holdsActivationGradients)
		{
			CudaAssert(cudaMalloc<float>(&m_activationGradientsBuffer, m_activationBufferSize));
			m_memoryConsumptionSize += m_activationBufferSize;
		}
	}

	CudaAssert(cudaSetDevice(0));
}

DropoutLayer::~DropoutLayer()
{
	if (m_dropoutFilter != NULL)
	{
		CudaAssert(cudaFree(m_dropoutFilter));
	}
}

void DropoutLayer::CopyDropoutFilterFromHost(float* hostDropoutFilter)
{
	CudaAssert(cudaMemcpyAsync(m_dropoutFilter, hostDropoutFilter, m_dropoutFilterSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

/*
	Drops filter values which are not above the drop probability, setting others to 1.
*/
__global__ void DropFilterValues(float* dropoutFilter, const uint dropoutFilterLength, float dropProbability)
{
	for (uint dropoutFilterIndex = blockIdx.x * blockDim.x + threadIdx.x; dropoutFilterIndex < dropoutFilterLength; dropoutFilterIndex += gridDim.x * blockDim.x)
	{
		dropoutFilter[dropoutFilterIndex] = dropoutFilter[dropoutFilterIndex] > dropProbability ? 1.0f : 0.0f;
	}
}

void DropoutLayer::CreateDropoutFilter()
{
	// Filling dropout filter with random values.
	const uint dropoutFilterLength = (uint)(m_dropoutFilterSize / sizeof(float));
	InitializeBufferFromUniformDistribution(m_dropoutFilter, dropoutFilterLength, 0.f, 1.f, m_curandStatesBuffer);

	// Dropping filter values which are not above the drop probability.
	const uint numBlocks = 128;
	const uint numThreadsPerBlock = 128;
	dim3 dropBlockDimensions(numThreadsPerBlock);
	dim3 dropGridDimensions(min(numBlocks, DivideUp(dropoutFilterLength, numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(DropFilterValues, dropGridDimensions, dropBlockDimensions, m_deviceCalculationStream)(m_dropoutFilter, dropoutFilterLength, m_dropProbability);
	CudaAssert(cudaGetLastError());
}

void DropoutLayer::ApplyDropoutFilter()
{
	CalculateElementWiseProduct(m_inputDataBuffer, m_dropoutFilter, (uint)(m_dropoutFilterSize / sizeof(float)), m_activationDataBuffer, m_deviceCalculationStream);
}

void DropoutLayer::DoForwardProp(PropagationMode propagationMode)
{
	if (propagationMode == PropagationMode::Train)
	{
		if (!m_useHostDropoutFilter)
		{
			CreateDropoutFilter();
		}
		ApplyDropoutFilter();
	}
	else
	{
		// Scaling inputs by probability that they will not be dropped, which is a reasonable approximation to taking the geometric mean
		// of the predictive distributions produced by the exponentially-many dropout networks.
		CalculateElementWiseScale(m_inputDataBuffer, 1.0f - m_dropProbability, (uint)(m_inputBufferSize / sizeof(float)), m_activationDataBuffer, m_deviceCalculationStream);
	}
}

void DropoutLayer::DoBackwardProp()
{
	CalculateElementWiseProduct(m_activationGradientsBuffer, m_dropoutFilter, (uint)(m_dropoutFilterSize / sizeof(float)), m_inputGradientsBuffer, m_deviceCalculationStream);
}