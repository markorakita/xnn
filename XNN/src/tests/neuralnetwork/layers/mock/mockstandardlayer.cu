// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network standard layer, used in tests.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockstandardlayer.cuh"

#include <chrono>

#include <cuda_runtime.h>

#include "../../mock/include/mockactivationfunctions.cuh"
#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"

MockStandardLayer::MockStandardLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons,
	float weightsUpdateMomentum, float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate,
	float weightsUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
	float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType, float activationAlpha)
	:
	MockWeightsLayer(0, (size_t)numNeurons * inputNumChannels * inputDataWidth * inputDataHeight * sizeof(float), inputNumChannels* inputDataWidth* inputDataHeight,
		weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor,
		numNeurons * sizeof(float), biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate,
		biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha)
{
	m_layerType = LayerType::Standard;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

	m_numNeurons = numNeurons;

	m_activationNumChannels = 1;
	m_activationDataWidth = m_numNeurons;
	m_activationDataHeight = 1;
	m_activationDataSize = m_activationDataWidth * m_activationDataHeight;

	m_inputBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = (size_t)m_inputDataCount * m_activationDataSize * sizeof(float);

	m_holdsActivationGradients = true;

	m_preactivationGradientsBuffer = NULL;
	m_preactivationDataBuffer = NULL;
}

void MockStandardLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating preactivation and activation data buffers.
	CudaAssert(cudaMallocHost<float>(&m_preactivationDataBuffer, m_activationBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

		// Allocating preactivation gradients buffer.
		CudaAssert(cudaMallocHost<float>(&m_preactivationGradientsBuffer, m_activationBufferSize));

		// Allocating activation gradients buffer.
		if (m_holdsActivationGradients)
		{
			CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
		}
	}
}

MockStandardLayer::~MockStandardLayer()
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
	if (m_preactivationDataBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_preactivationDataBuffer));
		m_preactivationDataBuffer = NULL;
	}
	if (m_activationDataBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_activationDataBuffer));
		m_activationDataBuffer = NULL;
	}
	if (m_preactivationGradientsBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_preactivationGradientsBuffer));
		m_preactivationGradientsBuffer = NULL;
	}
	if (m_holdsActivationGradients && m_activationGradientsBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_activationGradientsBuffer));
		m_activationGradientsBuffer = NULL;
	}
}

void MockStandardLayer::LoadInputs()
{
	ShipAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockStandardLayer::LoadActivationGradients()
{
	ShipAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockStandardLayer::CalculatePreactivations()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
		{
			const uint preactivationDataBufferOffset = dataIndex + neuronIndex * m_inputDataCount;
			m_preactivationDataBuffer[preactivationDataBufferOffset] = 0.f;
			for (uint weightIndex = 0; weightIndex < m_numWeightsPerNeuron; ++weightIndex)
			{
				m_preactivationDataBuffer[preactivationDataBufferOffset] += m_inputDataBuffer[dataIndex + weightIndex * m_inputDataCount] *
					m_weightsBuffer[neuronIndex * m_numWeightsPerNeuron + weightIndex];
			}
		}
	}
}

void MockStandardLayer::AddBiases()
{
	for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
	{
		float biasValue = m_biasesBuffer[neuronIndex];
		for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
		{
			m_preactivationDataBuffer[neuronIndex * m_inputDataCount + dataIndex] += biasValue;
		}
	}
}

void MockStandardLayer::CalculateActivations()
{
	ApplyActivationBF(m_activationType, m_activationAlpha, m_preactivationDataBuffer, (uint)(m_activationBufferSize / sizeof(float)),
		m_activationDataBuffer);
}

void MockStandardLayer::DoForwardProp(PropagationMode propagationMode)
{
	CalculatePreactivations();
	AddBiases();
	CalculateActivations();
}

void MockStandardLayer::CalculateBiasesGradients()
{
	float batchSize = m_parallelismMode == ParallelismMode::Model ? (float)m_inputDataCount : (float)(m_tierSize * m_inputDataCount);
	for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
	{
		float biasGradient = 0.f;
		uint neuronPreactivationsOffset = neuronIndex * m_inputDataCount;
		for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
		{
			biasGradient += m_preactivationGradientsBuffer[neuronPreactivationsOffset + dataIndex];
		}

		m_biasesGradientsBuffer[neuronIndex] = biasGradient / batchSize;
	}
}

void MockStandardLayer::CalculateWeightsGradients()
{
	float batchSize = m_parallelismMode == ParallelismMode::Model ? (float)m_inputDataCount : (float)(m_tierSize * m_inputDataCount);
	for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
	{
		for (uint weightIndex = 0; weightIndex < m_numWeightsPerNeuron; ++weightIndex)
		{
			const uint weightsGradientsBufferOffset = neuronIndex * m_numWeightsPerNeuron + weightIndex;
			m_weightsGradientsBuffer[weightsGradientsBufferOffset] = 0.f;
			for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
			{
				m_weightsGradientsBuffer[weightsGradientsBufferOffset] += m_inputDataBuffer[weightIndex * m_inputDataCount + dataIndex] *
					m_preactivationGradientsBuffer[neuronIndex * m_inputDataCount + dataIndex];
			}
			m_weightsGradientsBuffer[weightsGradientsBufferOffset] /= batchSize;
		}
	}
}

void MockStandardLayer::CalculateInputGradients()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint weightIndex = 0; weightIndex < m_numWeightsPerNeuron; ++weightIndex)
		{
			const uint inputGradientsBufferOffset = dataIndex + weightIndex * m_inputDataCount;
			m_inputGradientsBuffer[inputGradientsBufferOffset] = 0.f;
			for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
			{
				m_inputGradientsBuffer[inputGradientsBufferOffset] += m_preactivationGradientsBuffer[neuronIndex * m_inputDataCount + dataIndex] *
					m_weightsBuffer[neuronIndex * m_numWeightsPerNeuron + weightIndex];
			}
		}
	}
}

void MockStandardLayer::CalculatePreactivationsGradients()
{
	CalculatePreactivationGradientsBF(m_activationType, m_activationAlpha, m_activationGradientsBuffer, m_activationDataBuffer,
		(uint)(m_activationBufferSize / sizeof(float)), m_preactivationGradientsBuffer);
}

void MockStandardLayer::DoBackwardProp()
{
	CalculatePreactivationsGradients();
	CalculateInputGradients();
	CalculateWeightsGradients();
	CalculateBiasesGradients();
}