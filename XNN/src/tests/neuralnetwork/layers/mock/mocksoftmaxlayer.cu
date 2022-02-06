// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network softmax layer, used in tests.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mocksoftmaxlayer.cuh"

#include <cuda_runtime.h>

#include "../../../../neuralnetwork/layers/include/outputlayer.cuh"
#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"

MockSoftMaxLayer::MockSoftMaxLayer(uint inputDataSize, uint inputDataCount)
	:
	SoftMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, true)
{
}

void MockSoftMaxLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input activations maximums buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputActivationsMaxBuffer, m_inputDataCount * sizeof(float)));

	// Allocating input activations maximums buffer.
	CudaAssert(cudaMallocHost<float>(&m_exponentialsSumBuffer, m_inputDataCount * sizeof(float)));

	// Allocating activation data buffers.
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));
	}
}

MockSoftMaxLayer::~MockSoftMaxLayer()
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
	if (m_inputActivationsMaxBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_inputActivationsMaxBuffer));
		m_inputActivationsMaxBuffer = NULL;
	}
	if (m_exponentialsSumBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_exponentialsSumBuffer));
		m_exponentialsSumBuffer = NULL;
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
	if (m_NLLsBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_NLLsBuffer));
		m_NLLsBuffer = NULL;
	}
}

void MockSoftMaxLayer::LoadInputs()
{
	ShipAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockSoftMaxLayer::LoadActivationGradients()
{
	ShipAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");

	CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockSoftMaxLayer::StabilizeInputs()
{
	// Finding maximums of input activations.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		m_inputActivationsMaxBuffer[dataIndex] = m_inputDataBuffer[dataIndex];
		for (uint activationIndex = 1; activationIndex < m_activationDataSize; ++activationIndex)
		{
			m_inputActivationsMaxBuffer[dataIndex] = max(m_inputActivationsMaxBuffer[dataIndex], m_inputDataBuffer[activationIndex * m_inputDataCount + dataIndex]);
		}
	}

	// Substracting maximums of input activations from all the input activations.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint activationIndex = 0; activationIndex < m_activationDataSize; ++activationIndex)
		{
			m_activationDataBuffer[activationIndex * m_inputDataCount + dataIndex] = m_inputDataBuffer[activationIndex * m_inputDataCount + dataIndex] -
				m_inputActivationsMaxBuffer[dataIndex];
		}
	}
}

void MockSoftMaxLayer::CalculateSoftMaximums()
{
	// Computing the exponentials.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint activationIndex = 0; activationIndex < m_activationDataSize; ++activationIndex)
		{
			m_activationDataBuffer[activationIndex * m_inputDataCount + dataIndex] = exp(m_activationDataBuffer[activationIndex * m_inputDataCount + dataIndex]);
		}
	}

	// Computing sum of the exponentials.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		m_exponentialsSumBuffer[dataIndex] = m_activationDataBuffer[dataIndex];
		for (uint activationIndex = 1; activationIndex < m_activationDataSize; ++activationIndex)
		{
			m_exponentialsSumBuffer[dataIndex] += m_activationDataBuffer[activationIndex * m_inputDataCount + dataIndex];
		}
	}

	// Dividing exponentials with their sum to get soft maximums.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint activationIndex = 0; activationIndex < m_activationDataSize; ++activationIndex)
		{
			m_activationDataBuffer[activationIndex * m_inputDataCount + dataIndex] /= m_exponentialsSumBuffer[dataIndex];
		}
	}
}

void MockSoftMaxLayer::CalculateNegativeLogLikelihoods(uint* dataLabels)
{
	if (m_NLLsBuffer == NULL)
	{
		CudaAssert(cudaMallocHost<float>(&m_NLLsBuffer, m_inputDataCount * sizeof(float)));
	}

	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		m_NLLsBuffer[dataIndex] = m_inputActivationsMaxBuffer[dataIndex] + logf(m_exponentialsSumBuffer[dataIndex]) -
			m_inputDataBuffer[dataLabels[dataIndex] * m_inputDataCount + dataIndex];
	}
}

void MockSoftMaxLayer::DoForwardProp(PropagationMode propagationMode)
{
	StabilizeInputs();
	CalculateSoftMaximums();

	if (m_nextLayers[0]->GetLayerType() == LayerType::Output)
	{
		OutputLayer* outputLayer = static_cast<OutputLayer*>(m_nextLayers[0]);
		if (outputLayer->GetLossFunctionType() == LossFunctionType::CrossEntropy)
		{
			uint* tempHostLabelsBuffer;
			size_t labelsBufferSize = m_inputDataCount * sizeof(uint);
			CudaAssert(cudaMallocHost<uint>(&tempHostLabelsBuffer, labelsBufferSize));
			CudaAssert(cudaMemcpy(tempHostLabelsBuffer, outputLayer->GetDataLabels(), labelsBufferSize, cudaMemcpyDeviceToHost));

			CalculateNegativeLogLikelihoods(tempHostLabelsBuffer);

			CudaAssert(cudaFreeHost(tempHostLabelsBuffer));
		}
		else
		{
			ShipAssert(false, "Currently not supported!");
		}
	}
}

void MockSoftMaxLayer::CrossEntropyBackwardProp(uint* dataLabels)
{
	for (uint activationIndex = 0; activationIndex < m_activationDataSize; ++activationIndex)
	{
		for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
		{
			uint activationsOffset = activationIndex * m_inputDataCount + dataIndex;
			m_inputGradientsBuffer[activationsOffset] = m_activationDataBuffer[activationsOffset] - (dataLabels[dataIndex] == activationIndex ? 1.f : 0.f);
		}
	}
}

void MockSoftMaxLayer::DoBackwardProp()
{
	if (m_nextLayers[0]->GetLayerType() == LayerType::Output)
	{
		OutputLayer* outputLayer = static_cast<OutputLayer*>(m_nextLayers[0]);
		if (outputLayer->GetLossFunctionType() == LossFunctionType::CrossEntropy)
		{
			uint* tempHostLabelsBuffer;
			size_t labelsBufferSize = m_inputDataCount * sizeof(uint);
			CudaAssert(cudaMallocHost<uint>(&tempHostLabelsBuffer, labelsBufferSize));
			CudaAssert(cudaMemcpy(tempHostLabelsBuffer, outputLayer->GetDataLabels(), labelsBufferSize, cudaMemcpyDeviceToHost));

			CrossEntropyBackwardProp(tempHostLabelsBuffer);

			CudaAssert(cudaFreeHost(tempHostLabelsBuffer));
		}
		else
		{
			ShipAssert(false, "Currently not supported!");
		}
	}
	else
	{
		ShipAssert(false, "Currently not supported!");
	}
}