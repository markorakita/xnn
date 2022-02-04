// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network output layer, used in tests.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockoutputlayer.cuh"

#include <chrono>

#include <cuda_runtime.h>

#include "../../../../neuralnetwork/layers/include/softmaxlayer.cuh"
#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"

MockOutputLayer::MockOutputLayer(uint inputDataSize, uint inputDataCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses,
	curandState* curandStatesBuffer)
	:
	OutputLayer(0, 0, inputDataSize, inputDataCount, inputDataCount, lossFunctionType, calculateMultipleGuessAccuracy, numGuesses, 0)
{
	m_holdsInputData = true;

	m_curandStatesBuffer = curandStatesBuffer;

	m_multipleGuessScores = NULL;

	m_generateRandomInputGradients = false;
	m_inputGradientsMean = 0.f;
	m_inputGradientsStDev = 0.f;
}

MockOutputLayer::MockOutputLayer(uint inputDataSize, uint inputDataCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses,
	curandState* curandStatesBuffer, float gradientsMean, float gradientsStDev) :
	MockOutputLayer(inputDataSize, inputDataCount, lossFunctionType, calculateMultipleGuessAccuracy, numGuesses, curandStatesBuffer)
{
	m_generateRandomInputGradients = true;
	m_inputGradientsMean = gradientsMean;
	m_inputGradientsStDev = gradientsStDev;
}

void MockOutputLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(0));

	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	CudaAssert(cudaMallocHost<uint>(&m_dataLabels, m_labelsBufferSize));

	CudaAssert(cudaMallocHost<float>(&m_scores, m_lossBuffersSize));

	if (m_lossFunctionType != LossFunctionType::CrossEntropy)
	{
		CudaAssert(cudaMallocHost<float>(&m_lossBuffer, m_lossBuffersSize));
	}

	if (m_calculateMultipleGuessAccuracy)
	{
		CudaAssert(cudaMallocHost<float>(&m_multipleGuessScores, m_lossBuffersSize));
	}

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));
	}
}

MockOutputLayer::~MockOutputLayer()
{
	if (m_holdsInputData && m_inputDataBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
		m_inputDataBuffer = NULL;
	}
	if (m_dataLabels != NULL)
	{
		CudaAssert(cudaFreeHost(m_dataLabels));
		m_dataLabels = NULL;
	}
	if (m_scores != NULL)
	{
		CudaAssert(cudaFreeHost(m_scores));
		m_scores = NULL;
	}
	if (m_lossFunctionType != LossFunctionType::CrossEntropy && m_lossBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_lossBuffer));
		m_lossBuffer = NULL;
	}
	if (m_calculateMultipleGuessAccuracy && m_multipleGuessScores != NULL)
	{
		CudaAssert(cudaFreeHost(m_multipleGuessScores));
		m_multipleGuessScores = NULL;
	}
	if (m_inputGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_inputGradientsBuffer));
		m_inputGradientsBuffer = NULL;
	}

	m_activationDataBuffer = NULL;
}

void MockOutputLayer::LoadDataLabels(vector<uint> dataLabels)
{
	for (size_t i = 0; i < dataLabels.size(); ++i)
	{
		m_dataLabels[i] = dataLabels[i];
	}
}

void MockOutputLayer::LoadInputs()
{
	ShipAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockOutputLayer::CalculateLogisticRegressionLossesAndScores()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		float inputData = m_inputDataBuffer[dataIndex];
		uint classificationLabel = m_dataLabels[dataIndex];

		m_lossBuffer[dataIndex] = inputData >= 0.f ?
			(logf(1.0f + expf(-inputData)) + (1.0f - (float)classificationLabel) * inputData) :
			(logf(1.0f + expf(inputData)) - (float)classificationLabel * inputData);

		float sigmoidActivation = inputData >= 0.f ?
			1.0f / (1.0f + exp(-inputData)) :
			(1.0f - 1.0f / (1.0f + exp(inputData)));

		m_scores[dataIndex] = sigmoidActivation < 0.5f ?
			(classificationLabel == 0 ? 1.0f : 0.f) :
			(classificationLabel == 1 ? 1.0f : 0.f);
	}
}

void MockOutputLayer::LogisticRegressionForwardProp()
{
	CalculateLogisticRegressionLossesAndScores();

	// Calculating loss.
	m_loss = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_loss += m_lossBuffer[i];
	}

	// Calculating accuracy.
	m_accuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_accuracy += m_scores[i];
	}
}

void MockOutputLayer::CalculateCrossEntropyScores()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		float predictedProbability = m_inputDataBuffer[m_dataLabels[dataIndex] * m_inputDataCount + dataIndex];

		// Counting for how many incorrect labels we predicted higher or equal probability.
		uint predictedHigherOrEqualCnt = 0;
		for (size_t inputActivationIndex = 0; inputActivationIndex < m_inputDataSize; ++inputActivationIndex)
		{
			if (m_inputDataBuffer[inputActivationIndex * m_inputDataCount + dataIndex] >= predictedProbability)
			{
				++predictedHigherOrEqualCnt;
			}
		}

		m_scores[dataIndex] = predictedHigherOrEqualCnt > 1 ? 0.f : 1.0f;
		if (m_calculateMultipleGuessAccuracy)
		{
			m_multipleGuessScores[dataIndex] = predictedHigherOrEqualCnt > m_numGuesses ? 0.f : 1.f;
		}
	}
}

void MockOutputLayer::CrossEntropyForwardProp()
{
	SoftMaxLayer* softMaxLayer = NULL;
	if (m_prevLayers[0]->GetLayerType() == LayerType::SoftMax)
	{
		softMaxLayer = static_cast<SoftMaxLayer*>(m_prevLayers[0]);
	}
	else
	{
		ShipAssert(false, "It is expected to have SoftMax layer before Output layer in case of CrossEntropy loss!");
	}

	// Calculating loss.
	float* tempLogisticRegressionLossBuffer;
	size_t logisticRegressionLossBufferSize = m_inputDataCount * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&tempLogisticRegressionLossBuffer, logisticRegressionLossBufferSize));
	CudaAssert(cudaMemcpy(tempLogisticRegressionLossBuffer, softMaxLayer->GetNegativeLogLikelihoodsBuffer(), logisticRegressionLossBufferSize, cudaMemcpyDeviceToHost));

	m_loss = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_loss += tempLogisticRegressionLossBuffer[i];
	}

	CudaAssert(cudaFreeHost(tempLogisticRegressionLossBuffer));

	CalculateCrossEntropyScores();

	// Calculating accuracy.
	m_accuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_accuracy += m_scores[i];
	}

	// Calculating multiple guess accuracy.
	m_multipleGuessAccuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_multipleGuessAccuracy += m_multipleGuessScores[i];
	}
}

void MockOutputLayer::DoForwardProp(PropagationMode propagationMode)
{
	m_activationDataBuffer = m_inputDataBuffer;

	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		LogisticRegressionForwardProp();
	}
	else if (m_lossFunctionType == LossFunctionType::CrossEntropy)
	{
		CrossEntropyForwardProp();
	}
	else
	{
		ShipAssert(false, "Currently not supported!");
	}
}

void MockOutputLayer::DoBackwardProp()
{
	if (m_generateRandomInputGradients)
	{
		const uint inputGradientsBufferLength = (uint)(m_inputBufferSize / sizeof(float));
		InitializeBufferFromNormalDistribution(m_inputGradientsBuffer, inputGradientsBufferLength, m_inputGradientsMean, m_inputGradientsStDev, m_curandStatesBuffer);
	}
	else
	{
		if (m_lossFunctionType == LossFunctionType::LogisticRegression)
		{
			if (m_prevLayers[0]->GetLayerType() == LayerType::SoftMax)
			{
				// If previous layer is SoftMax we are letting it handle the gradient computation, for numerical stability.
				m_inputGradientsBuffer = NULL;
			}
			else
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
	}
}