// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network standard layer.
// Created: 01/17/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/standardlayer.cuh"

#include "../include/activationfunctions.cuh"
#include "../../utils/include/config.cuh"
#include "../../utils/include/cublasasserts.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

StandardLayer::StandardLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, cublasHandle_t cublasHandle,
	curandState* curandStatesBuffer, uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount,
	bool holdsInputData, uint numNeurons, float weightsUpdateMomentum, float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep,
	float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay,
	float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType,
	float activationAlpha, bool holdsActivationGradients)
	:
	WeightsLayer(indexInTier, (size_t)numNeurons * inputNumChannels * inputDataWidth * inputDataHeight * sizeof(float), inputNumChannels * inputDataWidth * inputDataHeight,
		weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor,
		numNeurons * sizeof(float), biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate,
		biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha, curandStatesBuffer)
{
	m_layerType = LayerType::Standard;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_cublasHandle = cublasHandle;
	m_indexInTier = indexInTier;
	m_tierSize = tierSize;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_numNeurons = numNeurons;

	m_activationNumChannels = 1;
	m_activationDataWidth = m_numNeurons;
	m_activationDataHeight = 1;
	m_activationDataSize = m_activationDataWidth * m_activationDataHeight;

	m_inputBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = (size_t)m_inputDataCount * m_activationDataSize * sizeof(float);

	m_holdsActivationGradients = holdsActivationGradients;

	m_preactivationGradientsBuffer = NULL;
	m_preactivationDataBuffer = NULL;
}

void StandardLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	WeightsLayer::AllocateBuffers(allocateTrainBuffers);

	CudaAssert(cudaSetDevice(m_indexInTier));

	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;
	}

	// Allocating preactivation and activation data buffers.
	CudaAssert(cudaMalloc<float>(&m_preactivationDataBuffer, m_activationBufferSize));
	m_memoryConsumptionSize += m_activationBufferSize;
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));
	m_memoryConsumptionSize += m_activationBufferSize;

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;

		// Allocating preactivation gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_preactivationGradientsBuffer, m_activationBufferSize));
		m_memoryConsumptionSize += m_activationBufferSize;

		// Allocating activation gradients buffer.
		if (m_holdsActivationGradients)
		{
			CudaAssert(cudaMalloc<float>(&m_activationGradientsBuffer, m_activationBufferSize));
			m_memoryConsumptionSize += m_activationBufferSize;
		}
	}

	CudaAssert(cudaSetDevice(0));
}

StandardLayer::~StandardLayer()
{
	if (m_preactivationDataBuffer != NULL)
	{
		CudaAssert(cudaFree(m_preactivationDataBuffer));
	}
	if (m_preactivationGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_preactivationGradientsBuffer));
	}
}

void StandardLayer::Reinitialize(uint newInputDataCount)
{
	Layer::Reinitialize(newInputDataCount);

	m_activationBufferSize = (size_t)m_inputDataCount * m_activationDataSize * sizeof(float);
}

void StandardLayer::CalculatePreactivations()
{
	CudaCublasAssert(cublasSetStream_v2(m_cublasHandle, m_deviceCalculationStream));
	float alpha = 1.0f;
	float beta = 0.f;
	CudaCublasAssert(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, (int)m_inputDataCount, (int)m_numNeurons, (int)m_numWeightsPerNeuron,
		&alpha, m_inputDataBuffer, (int)m_inputDataCount, m_weightsBuffer, (int)m_numWeightsPerNeuron, &beta, m_preactivationDataBuffer, (int)m_inputDataCount));
}

/*
	Does grid stride and adds biases to preactivations.
*/
__global__ void AddNeuronBiases(float* preactivations, float* biases, const uint width, const uint height)
{
	for (uint y = blockIdx.y; y < height; y += gridDim.y)
	{
		// TODO: run tests to measure speed without shfl, and if same or better then replace this with concurrent read of same bias.
		int laneId = threadIdx.x % warpSize;
		int biasValue;
		if (laneId == 0)
		{
			biasValue = biases[y];
		}
		biasValue = __shfl_sync(0xFFFFFFFF, biasValue, 0);

		for (uint x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
		{
			preactivations[y * width + x] += biasValue;
		}
	}
}

void StandardLayer::AddBiases()
{
	const uint c_numThreadsPerBlock = min((uint)Config::MAX_NUM_THREADS, RoundUp(m_inputDataCount, Config::WARP_SIZE));
	dim3 blockDimensions(c_numThreadsPerBlock);
	uint c_numBlocks = (uint)((Config::MAX_NUM_THREADS / c_numThreadsPerBlock) * 128);
	dim3 gridDimensions(1, c_numBlocks);
	LAUNCH_KERNEL_ASYNC(AddNeuronBiases, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationDataBuffer, m_biasesBuffer,
		m_inputDataCount, m_numNeurons);
	CudaAssert(cudaGetLastError());
}

void StandardLayer::CalculateActivations()
{
	ApplyActivation(m_activationType, m_activationAlpha, m_preactivationDataBuffer, (uint)(m_activationBufferSize / sizeof(float)), m_activationDataBuffer,
		m_deviceCalculationStream);
}

void StandardLayer::DoForwardProp(PropagationMode propagationMode)
{
	CalculatePreactivations();
	AddBiases();
	CalculateActivations();
}

/*
	Calculates biases gradients, each thread calculating gradient for one bias.
*/
__global__ void __CalculateStandardBiasesGradients(float* preactivationGradients, const uint numNeurons, const uint inputDataCount, const uint batchSize,
	float* biasesGradients)
{
	const uint c_neuronIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const uint c_neuronPreactivationsOffset = c_neuronIndex * inputDataCount;

	if (c_neuronIndex < numNeurons)
	{
		float biasGradient = 0.f;
		for (uint dataIndex = 0; dataIndex < inputDataCount; ++dataIndex)
		{
			biasGradient += preactivationGradients[c_neuronPreactivationsOffset + dataIndex];
		}

		biasesGradients[c_neuronIndex] = biasGradient / (float)batchSize;
	}
}

void StandardLayer::CalculateBiasesGradients()
{
	const uint c_numThreadsPerBlock = 128;
	const uint c_numBlocks = DivideUp(m_numNeurons, c_numThreadsPerBlock);
	const uint c_batchSize = m_parallelismMode == ParallelismMode::Model ? m_inputDataCount : m_tierSize * m_inputDataCount;
	LAUNCH_KERNEL_ASYNC(__CalculateStandardBiasesGradients, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(m_preactivationGradientsBuffer,
		m_numNeurons, m_inputDataCount, c_batchSize, m_biasesGradientsBuffer);
	CudaAssert(cudaGetLastError());
}

void StandardLayer::CalculateWeightsGradients()
{
	CudaCublasAssert(cublasSetStream_v2(m_cublasHandle, m_deviceCalculationStream));
	uint batchSize = m_parallelismMode == ParallelismMode::Model ? m_inputDataCount : m_tierSize * m_inputDataCount;
	float alpha = 1.0f / batchSize;
	float beta = 0.f;
	CudaCublasAssert(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (int)m_numWeightsPerNeuron, (int)m_numNeurons, (int)m_inputDataCount,
		&alpha, m_inputDataBuffer, (int)m_inputDataCount, m_preactivationGradientsBuffer, (int)m_inputDataCount, &beta, m_weightsGradientsBuffer, (int)m_numWeightsPerNeuron));
}

void StandardLayer::CalculateInputGradients()
{
	CudaCublasAssert(cublasSetStream_v2(m_cublasHandle, m_deviceCalculationStream));
	float alpha = 1.0f;
	float beta = 0.f;
	CudaCublasAssert(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, (int)m_inputDataCount, (int)m_numWeightsPerNeuron, (int)m_numNeurons,
		&alpha, m_preactivationGradientsBuffer, (int)m_inputDataCount, m_weightsBuffer, (int)m_numWeightsPerNeuron, &beta, m_inputGradientsBuffer, (int)m_inputDataCount));
}

void StandardLayer::CalculatePreactivationsGradients()
{
	CalculatePreactivationGradients(m_activationType, m_activationAlpha, m_activationGradientsBuffer, m_activationDataBuffer,
		(uint)(m_activationBufferSize / sizeof(float)), m_preactivationGradientsBuffer, m_deviceCalculationStream);
}

void StandardLayer::DoBackwardProp()
{
	CalculatePreactivationsGradients();
	CalculateInputGradients();
	CalculateWeightsGradients();
	CalculateBiasesGradients();
}