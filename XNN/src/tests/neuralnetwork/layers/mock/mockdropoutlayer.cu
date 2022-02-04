// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network dropout layer, used in tests.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockdropoutlayer.cuh"

#include <chrono>

#include <cuda_runtime.h>

#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"

MockDropoutLayer::MockDropoutLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability,
	curandState* curandStatesBuffer)
	:
	DropoutLayer(ParallelismMode::Model, 0, 0, curandStatesBuffer, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, true,
		dropProbability, false, true)
{
}

void MockDropoutLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating dropout filter buffer.
	CudaAssert(cudaMallocHost<float>(&m_dropoutFilter, m_dropoutFilterSize));

	// Allocating activation data buffers.
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

		// Allocating activation gradients buffer.
		if (m_holdsActivationGradients)
		{
			CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
		}
	}
}

MockDropoutLayer::~MockDropoutLayer()
{
	if (m_holdsInputData && m_inputDataBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
		m_inputDataBuffer = NULL;
	}
	if (m_inputGradientsBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_inputGradientsBuffer));
		m_inputGradientsBuffer = NULL;
	}
	if (m_dropoutFilter != NULL)
	{
		CudaAssert(cudaFreeHost(m_dropoutFilter));
		m_dropoutFilter = NULL;
	}
	if (m_activationDataBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_activationDataBuffer));
		m_activationDataBuffer = NULL;
	}
	if (m_holdsActivationGradients && m_activationGradientsBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_activationGradientsBuffer));
		m_activationGradientsBuffer = NULL;
	}
}

void MockDropoutLayer::LoadInputs()
{
	ShipAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockDropoutLayer::LoadActivationGradients()
{
	ShipAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");

	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockDropoutLayer::CreateDropoutFilter()
{
	// Filling dropout filter with random values.
	float* deviceBuffer;
	CudaAssert(cudaMalloc<float>(&deviceBuffer, m_dropoutFilterSize));

	uint dropoutFilterLength = (uint)(m_dropoutFilterSize / sizeof(float));
	InitializeBufferFromUniformDistribution(deviceBuffer, dropoutFilterLength, 0.f, 1.f, m_curandStatesBuffer);
	SynchronizeCalculations();

	CudaAssert(cudaMemcpy(m_dropoutFilter, deviceBuffer, m_dropoutFilterSize, cudaMemcpyDeviceToHost));
	CudaAssert(cudaFree(deviceBuffer));

	// Dropping filter values which are not above the drop probability.
	for (uint i = 0; i < dropoutFilterLength; ++i)
	{
		m_dropoutFilter[i] = m_dropoutFilter[i] > m_dropProbability ? 1.0f : 0.0f;
	}
}

void MockDropoutLayer::ApplyDropoutFilter()
{
	size_t dropoutFilterLength = m_dropoutFilterSize / sizeof(float);
	for (size_t i = 0; i < dropoutFilterLength; ++i)
	{
		m_activationDataBuffer[i] = m_inputDataBuffer[i] * m_dropoutFilter[i];
	}
}

void MockDropoutLayer::DoForwardProp(PropagationMode propagationMode)
{
	CreateDropoutFilter();
	ApplyDropoutFilter();
}

void MockDropoutLayer::DoBackwardProp()
{
	size_t dropoutFilterLength = m_dropoutFilterSize / sizeof(float);
	for (size_t i = 0; i < dropoutFilterLength; ++i)
	{
		m_inputGradientsBuffer[i] = m_activationGradientsBuffer[i] * m_dropoutFilter[i];
	}
}