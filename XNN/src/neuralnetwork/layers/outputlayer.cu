// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network output layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/outputlayer.cuh"

#include "include/softmaxlayer.cuh"
#include "../include/matrixoperations.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/config.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

OutputLayer::OutputLayer(cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint inputDataSize, uint inputDataCount,
	uint labelsCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses, uint numTestPasses)
{
	m_layerType = LayerType::Output;
	m_parallelismMode = ParallelismMode::Model;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = 0;
	m_tierSize = 1;
	m_lossFunctionType = lossFunctionType;

	m_numTestPasses = numTestPasses;
	m_testPassCounter = 0;

	m_inputDataSize = m_activationDataSize = inputDataSize;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = false;
	m_inputBufferSize = m_activationBufferSize = (size_t)m_inputDataSize * m_inputDataCount * sizeof(float);
	m_testAverageInputsBuffer = NULL;

	m_labelsOffset = 0;
	m_labelsBufferSize = labelsCount * sizeof(uint);

	m_lossBuffersSize = m_inputDataCount * sizeof(float);
	m_loss = 0.f;
	m_accuracy = 0.f;
	m_multipleGuessAccuracy = 0.f;

	m_calculateMultipleGuessAccuracy = calculateMultipleGuessAccuracy;
	m_numGuesses = numGuesses;

	m_holdsActivationGradients = false;

	m_lossBuffer = NULL;
	m_hostLossBuffer = NULL;
	m_dataLabels = NULL;
	m_hostLabelsBuffer = NULL;
	m_scores = NULL;
	m_hostScores = NULL;
	m_multipleGuessScores = NULL;
	m_hostMultipleGuessScores = NULL;
}

void OutputLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(0));

	CudaAssert(cudaMalloc<uint>(&m_dataLabels, m_labelsBufferSize));
	m_memoryConsumptionSize += m_labelsBufferSize;
	CudaAssert(cudaMallocHost<uint>(&m_hostLabelsBuffer, m_labelsBufferSize));

	if (m_lossFunctionType != LossFunctionType::CrossEntropy)
	{
		CudaAssert(cudaMalloc<float>(&m_lossBuffer, m_lossBuffersSize));
	}
	CudaAssert(cudaMallocHost<float>(&m_hostLossBuffer, m_lossBuffersSize));

	CudaAssert(cudaMalloc<float>(&m_scores, m_lossBuffersSize));
	m_memoryConsumptionSize += m_lossBuffersSize;
	CudaAssert(cudaMallocHost<float>(&m_hostScores, m_lossBuffersSize));

	if (m_calculateMultipleGuessAccuracy)
	{
		CudaAssert(cudaMalloc<float>(&m_multipleGuessScores, m_lossBuffersSize));
		m_memoryConsumptionSize += m_lossBuffersSize;
		CudaAssert(cudaMallocHost<float>(&m_hostMultipleGuessScores, m_lossBuffersSize));
	}

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;
	}
}

OutputLayer::~OutputLayer()
{
	m_inputDataBuffer = NULL;
	m_activationDataBuffer = NULL;

	if (m_dataLabels != NULL)
	{
		CudaAssert(cudaFree(m_dataLabels));
	}
	if (m_hostLabelsBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_hostLabelsBuffer));
	}
	
	if (m_lossFunctionType != LossFunctionType::CrossEntropy && m_lossBuffer != NULL)
	{
		CudaAssert(cudaFree(m_lossBuffer));
	}
	if (m_hostLossBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_hostLossBuffer));
	}

	if (m_scores != NULL)
	{
		CudaAssert(cudaFree(m_scores));
	}
	if (m_hostScores != NULL)
	{
		CudaAssert(cudaFreeHost(m_hostScores));
	}

	if (m_calculateMultipleGuessAccuracy)
	{
		if (m_multipleGuessScores != NULL)
		{
			CudaAssert(cudaFree(m_multipleGuessScores));
		}
		if (m_hostMultipleGuessScores != NULL)
		{
			CudaAssert(cudaFreeHost(m_hostMultipleGuessScores));
		}
	}

	if (m_testAverageInputsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_testAverageInputsBuffer));
	}
}

void OutputLayer::Reinitialize(uint newInputDataCount)
{
	m_inputDataCount = newInputDataCount;
	m_inputBufferSize = m_activationBufferSize = (size_t)m_inputDataSize * m_inputDataCount * sizeof(float);
	m_lossBuffersSize = m_inputDataCount * sizeof(float);
}

void OutputLayer::LoadDataLabels(vector<uint> dataLabels)
{
	m_labelsOffset = 0;

	for (size_t i = 0; i < dataLabels.size(); ++i)
	{
		m_hostLabelsBuffer[i] = dataLabels[i];
	}

	CudaAssert(cudaMemcpyAsync(m_dataLabels, m_hostLabelsBuffer, dataLabels.size() * sizeof(uint), cudaMemcpyHostToDevice, m_deviceMemoryStream));

	SynchronizeMemoryOperations();
}

/*
	Calculates losses and accuracy scores for logistic regression loss function.
*/
__global__ void CalculateLogisticRegressionLossesAndScores(float* inputBuffer, uint* dataLabels, const uint numInputSamples, float* losses, float* scores)
{
	const uint c_dataIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_dataIndex < numInputSamples)
	{
		float inputData = inputBuffer[c_dataIndex];
		uint classificationLabel = dataLabels[c_dataIndex];

		losses[c_dataIndex] = inputData >= 0.f ?
			(__logf(1.0f + __expf(-inputData)) + (1.0f - (float)classificationLabel) * inputData) :
			(__logf(1.0f + __expf(inputData)) - (float)classificationLabel * inputData);

		float sigmoidActivation = inputData >= 0.f ?
			__fdividef(1.0f, 1.0f + __expf(-inputData)) :
			(1.0f - __fdividef(1.0f, 1.0f + __expf(inputData)));

		scores[c_dataIndex] = sigmoidActivation < 0.5f ?
			(classificationLabel == 0 ? 1.0f : 0.f) :
			(classificationLabel == 1 ? 1.0f : 0.f);
	}
}

void OutputLayer::LogisticRegressionForwardProp(float* inputBuffer)
{
	// Calculating losses and scores.
	const uint c_numThreadsPerBlock = 128;
	const uint c_numBlocks = DivideUp(m_inputDataCount, c_numThreadsPerBlock);
	LAUNCH_KERNEL_ASYNC(CalculateLogisticRegressionLossesAndScores, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(inputBuffer,
		m_dataLabels + m_labelsOffset, m_inputDataCount, m_lossBuffer, m_scores);
	CudaAssert(cudaGetLastError());
	SynchronizeCalculations();

	// Copying loss and score buffers to host memory. It has to be done after calculations are finished!
	CudaAssert(cudaMemcpyAsync(m_hostLossBuffer, m_lossBuffer, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	CudaAssert(cudaMemcpyAsync(m_hostScores, m_scores, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	SynchronizeMemoryOperations();

	// Calculating loss.
	m_loss = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_loss += m_hostLossBuffer[i];
	}

	// Calculating accuracy.
	m_accuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_accuracy += m_hostScores[i];
	}
}

/*
	Calculates accuracy scores for cross entropy loss function.
*/
__global__ void CalculateCrossEntropyScores(float* inputActivations, uint* dataLabels, const uint numInputSamples, const uint numInputActivations,
	float* scores, bool calculateMultipleGuessAccuracy, uint numGuesses, float* multipleGuessScores)
{
	const uint c_dataIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_dataIndex < numInputSamples)
	{
		float predictedProbability = inputActivations[dataLabels[c_dataIndex] * numInputSamples + c_dataIndex];

		// Counting for how many incorrect labels we predicted higher or equal probability.
		uint predictedHigherOrEqualCnt = 0;
		for (size_t inputActivationIndex = 0; inputActivationIndex < numInputActivations; ++inputActivationIndex)
		{
			if (inputActivations[inputActivationIndex * numInputSamples + c_dataIndex] >= predictedProbability)
			{
				++predictedHigherOrEqualCnt;
			}
		}

		scores[c_dataIndex] = predictedHigherOrEqualCnt > 1 ? 0.f : 1.0f;
		if (calculateMultipleGuessAccuracy)
		{
			multipleGuessScores[c_dataIndex] = predictedHigherOrEqualCnt > numGuesses ? 0.f : 1.f;
		}
	}
}

void OutputLayer::CrossEntropyForwardProp(float* inputBuffer)
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

	// Calculating log likelihoods and scores.
	const uint c_numThreadsPerBlock = 128;
	const uint c_numBlocks = DivideUp(m_inputDataCount, c_numThreadsPerBlock);
	LAUNCH_KERNEL_ASYNC(CalculateCrossEntropyScores, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(inputBuffer, m_dataLabels + m_labelsOffset,
		m_inputDataCount, m_activationDataSize, m_scores, m_calculateMultipleGuessAccuracy, m_numGuesses, m_multipleGuessScores);
	CudaAssert(cudaGetLastError());
	SynchronizeCalculations();

	// Copying loss and score buffers to host memory. It has to be done after calculations are finished!
	CudaAssert(cudaMemcpyAsync(m_hostLossBuffer, softMaxLayer->GetNegativeLogLikelihoodsBuffer(), m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	CudaAssert(cudaMemcpyAsync(m_hostScores, m_scores, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	if (m_calculateMultipleGuessAccuracy)
	{
		CudaAssert(cudaMemcpyAsync(m_hostMultipleGuessScores, m_multipleGuessScores, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	}
	SynchronizeMemoryOperations();

	// Calculating loss.
	m_loss = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_loss += m_hostLossBuffer[i];
	}

	// Calculating accuracy.
	m_accuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_accuracy += m_hostScores[i];
	}

	if (m_calculateMultipleGuessAccuracy)
	{
		// Calculating multiple guess accuracy.
		m_multipleGuessAccuracy = 0.f;
		for (uint i = 0; i < m_inputDataCount; ++i)
		{
			m_multipleGuessAccuracy += m_hostMultipleGuessScores[i];
		}
	}
}

void OutputLayer::DoForwardProp(PropagationMode propagationMode)
{
	m_activationDataBuffer = m_inputDataBuffer;

	float* inputBuffer = m_inputDataBuffer;
	bool lastTestPass = false;
	if (propagationMode == PropagationMode::Test && m_numTestPasses > 1)
	{
		++m_testPassCounter;

		if (m_testPassCounter == 1)
		{
			// Allocate test average inputs buffer first time we do a test pass.
			if (m_testAverageInputsBuffer == NULL)
			{
				CudaAssert(cudaMalloc<float>(&m_testAverageInputsBuffer, m_inputBufferSize));
			}

			// Using device calculation stream on purpose, to avoid sync between streams.
			CudaAssert(cudaMemcpyAsync(m_testAverageInputsBuffer, m_inputDataBuffer, m_inputBufferSize, cudaMemcpyDeviceToDevice, m_deviceCalculationStream));
		}
		else
		{
			// Adding input from this pass.
			CalculateElementWiseSum(m_testAverageInputsBuffer, m_inputDataBuffer, (uint)(m_inputBufferSize / sizeof(float)), m_testAverageInputsBuffer,
				m_deviceCalculationStream);
		}

		if (m_testPassCounter == m_numTestPasses)
		{
			lastTestPass = true;
			m_testPassCounter = 0;

			// Averaging summed inputs from each test pass.
			CalculateElementWiseScale(m_testAverageInputsBuffer, 1.0f / m_numTestPasses, (uint)(m_inputBufferSize / sizeof(float)), m_testAverageInputsBuffer,
				m_deviceCalculationStream);
		}

		inputBuffer = m_testAverageInputsBuffer;
	}

	if (propagationMode == PropagationMode::Train || lastTestPass || m_numTestPasses == 1)
	{
		if (m_lossFunctionType == LossFunctionType::LogisticRegression)
		{
			LogisticRegressionForwardProp(inputBuffer);
		}
		else if (m_lossFunctionType == LossFunctionType::CrossEntropy)
		{
			CrossEntropyForwardProp(inputBuffer);
		}
		else
		{
			ShipAssert(false, "Currently not supported!");
		}
	}
}

/*
	Calculates input gradients for logistic regression loss function.
*/
__global__ void CalculateLogisticRegressionInputGradients(float* inputBuffer, uint* dataLabels, const uint numInputSamples, float* inputGradients)
{
	const uint c_dataIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_dataIndex < numInputSamples)
	{
		float inputData = inputBuffer[c_dataIndex];
		uint classificationLabel = dataLabels[c_dataIndex];

		float sigmoidActivation = inputData >= 0.f ?
			__fdividef(1.0f, 1.0f + __expf(-inputData)) :
			(1.0f - __fdividef(1.0f, 1.0f + __expf(inputData)));

		inputGradients[c_dataIndex] = sigmoidActivation - classificationLabel;
	}
}

void OutputLayer::LogisticRegressionBackwardProp()
{
	// Calculating losses and scores.
	const uint c_numThreadsPerBlock = 128;
	const uint c_numBlocks = DivideUp(m_inputDataCount, c_numThreadsPerBlock);
	LAUNCH_KERNEL_ASYNC(CalculateLogisticRegressionInputGradients, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(m_inputDataBuffer,
		m_dataLabels + m_labelsOffset, m_inputDataCount, m_inputGradientsBuffer);
	CudaAssert(cudaGetLastError());
	SynchronizeCalculations();
}

void OutputLayer::DoBackwardProp()
{
	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		LogisticRegressionBackwardProp();
	}
	else if (m_lossFunctionType == LossFunctionType::CrossEntropy)
	{
		if (m_prevLayers[0]->GetLayerType() == LayerType::SoftMax)
		{
			// If previous layer is SoftMax we are letting it handle the gradient computation, for numerical stability.
		}
		else
		{
			ShipAssert(false, "It is expected to have SoftMax layer before Output layer in case of CrossEntropy loss!");
		}
	}
	else
	{
		ShipAssert(false, "Currently not supported!");
	}
}