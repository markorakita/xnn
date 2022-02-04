// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/softmaxlayer.cuh"

#include "include/outputlayer.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/config.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

SoftMaxLayer::SoftMaxLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint inputDataSize,
	uint inputDataCount, bool holdsInputData)
{
	m_layerType = LayerType::SoftMax;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = m_activationNumChannels = 1;
	m_inputDataWidth = m_activationDataWidth = inputDataSize;
	m_inputDataHeight = m_activationDataHeight = 1;
	m_inputDataSize = m_activationDataSize = inputDataSize;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_inputBufferSize = (size_t)m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = m_inputBufferSize;

	m_holdsActivationGradients = false;

	m_NLLsBuffer = NULL;
	m_inputActivationsMaxBuffer = NULL;
	m_exponentialsSumBuffer = NULL;
}

void SoftMaxLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(0));

	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;
	}

	// Allocating input activations maximums buffer.
	size_t inputActivationsMaxBufferSize = m_inputDataCount * sizeof(float);
	CudaAssert(cudaMalloc<float>(&m_inputActivationsMaxBuffer, inputActivationsMaxBufferSize));
	m_memoryConsumptionSize += inputActivationsMaxBufferSize;

	// Allocating sum of exponentials buffer.
	size_t exponentialsSumBufferSize = m_inputDataCount * sizeof(float);
	CudaAssert(cudaMalloc<float>(&m_exponentialsSumBuffer, exponentialsSumBufferSize));
	m_memoryConsumptionSize += exponentialsSumBufferSize;

	// Allocating activation data buffers.
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));
	m_memoryConsumptionSize += m_activationBufferSize;

	// Allocating buffers necessary for training.
	if (allocateTrainBuffers)
	{
		// Allocating input gradients buffer.
		CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));
		m_memoryConsumptionSize += m_inputBufferSize;
	}
}

SoftMaxLayer::~SoftMaxLayer()
{
	if (m_inputActivationsMaxBuffer != NULL)
	{
		CudaAssert(cudaFree(m_inputActivationsMaxBuffer));
	}
	if (m_exponentialsSumBuffer != NULL)
	{
		CudaAssert(cudaFree(m_exponentialsSumBuffer));
	}
	if (m_NLLsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_NLLsBuffer));
	}
}

/*
	Finds maximum values of input activations for each input sample.
*/
__global__ void FindMaximums(float* inputActivations, const uint numInputSamples, const uint numInputActivations, float* inputActivationsMaximums)
{
	const uint c_sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (c_sampleIndex < numInputSamples)
	{
		float activationMaximum = inputActivations[c_sampleIndex];
		for (uint activationIndex = 1; activationIndex < numInputActivations; ++activationIndex)
		{
			activationMaximum = max(activationMaximum, inputActivations[activationIndex * numInputSamples + c_sampleIndex]);
		}
		inputActivationsMaximums[c_sampleIndex] = activationMaximum;
	}
}

/*
	Subtracts maximum values of input activations from all input activations for each input sample.
*/
template <uint c_blockWidth>
__global__ void SubtractMaximums(float* inputActivations, const uint numInputSamples, const uint numInputActivations, float* inputActivationsMaximums,
	float* outputActivations)
{
	__shared__ float maximums[c_blockWidth];

	for (uint y = blockIdx.y * blockDim.y + threadIdx.y; y < numInputActivations; y += gridDim.y * blockDim.y)
	{
		__syncthreads();
		if (threadIdx.y == 0)
		{
			maximums[threadIdx.x] = inputActivationsMaximums[blockIdx.x * blockDim.x + threadIdx.x];
		}
		__syncthreads();

		const uint c_offset = y * numInputSamples;
		for (uint x = blockIdx.x * blockDim.x + threadIdx.x; x < numInputSamples; x += gridDim.x * blockDim.x)
		{
			outputActivations[c_offset + x] = inputActivations[c_offset + x] - maximums[threadIdx.x];
		}
	}
}

void SoftMaxLayer::StabilizeInputs()
{
	// Finding maximums of input activations.
	const uint c_numThreadsPerBlock = min((uint)Config::MAX_NUM_THREADS, RoundUp(m_inputDataCount, Config::WARP_SIZE));
	const uint c_numBlocks = DivideUp(m_inputDataCount, c_numThreadsPerBlock);
	LAUNCH_KERNEL_ASYNC(FindMaximums, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataCount,
		m_activationDataSize, m_inputActivationsMaxBuffer);
	CudaAssert(cudaGetLastError());

	// Substracting maximums of input activations from all the input activations.
	const uint c_blockWidth = 64;
	const uint c_blockHeight = (uint)Config::MAX_NUM_THREADS / c_blockWidth;
	dim3 blockDimensions(c_blockWidth, c_blockHeight);
	const uint c_maxGridBlocks = 128;
	const uint c_gridWidth = min(c_maxGridBlocks, DivideUp(m_inputDataCount, c_blockWidth));
	const uint c_gridHeight = min(c_maxGridBlocks / c_gridWidth, DivideUp(m_activationDataSize, c_blockHeight));
	dim3 gridDimensions(c_gridWidth, c_gridHeight);
	LAUNCH_KERNEL_ASYNC((SubtractMaximums<c_blockWidth>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataCount,
		m_activationDataSize, m_inputActivationsMaxBuffer, m_activationDataBuffer);
	CudaAssert(cudaGetLastError());
}

/*
	Computes the exponentials of activations.
*/
__global__ void ComputeExponentials(float* activations, const uint activationsLength)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < activationsLength; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = __expf(activations[activationIndex]);
	}
}

/*
	Computes sum of the exponentials of activations.
*/
__global__ void ComputeSumOfExponentials(float* activations, const uint numInputSamples, const uint numActivations, float* exponentialsSumBuffer)
{
	const uint c_sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (c_sampleIndex < numInputSamples)
	{
		float exponentialsSum = 0.f;
		for (uint activationIndex = 0; activationIndex < numActivations; ++activationIndex)
		{
			exponentialsSum += activations[activationIndex * numInputSamples + c_sampleIndex];
		}
		exponentialsSumBuffer[c_sampleIndex] = exponentialsSum;
	}
}

/*
	Divides activation exponentials with their sum to get soft maximums.
*/
template <uint c_blockWidth>
__global__ void DivideExponentialsWithSum(float* activationExponentials, const uint numInputSamples, const uint numActivations, float* exponentialsSumBuffer)
{
	__shared__ float exponentialsSums[c_blockWidth];

	for (uint y = blockIdx.y * blockDim.y + threadIdx.y; y < numActivations; y += gridDim.y * blockDim.y)
	{
		__syncthreads();
		if (threadIdx.y == 0)
		{
			exponentialsSums[threadIdx.x] = exponentialsSumBuffer[blockIdx.x * blockDim.x + threadIdx.x];
		}
		__syncthreads();

		const uint c_offset = y * numInputSamples;
		for (uint x = blockIdx.x * blockDim.x + threadIdx.x; x < numInputSamples; x += gridDim.x * blockDim.x)
		{
			activationExponentials[c_offset + x] = __fdividef(activationExponentials[c_offset + x], exponentialsSums[threadIdx.x]);
		}
	}
}

void SoftMaxLayer::CalculateSoftMaximums()
{
	// Computing the exponentials.
	const uint c_activationBufferLength = (uint)(m_activationBufferSize / sizeof(float));
	uint numBlocks = 128;
	uint numThreadsPerBlock = 128;
	dim3 blockDimensions(numThreadsPerBlock);
	dim3 gridDimensions(min(numBlocks, DivideUp(c_activationBufferLength, numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(ComputeExponentials, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_activationDataBuffer, c_activationBufferLength);
	CudaAssert(cudaGetLastError());

	// Computing sum of the exponentials.
	numThreadsPerBlock = min((uint)Config::MAX_NUM_THREADS, RoundUp(m_inputDataCount, Config::WARP_SIZE));
	numBlocks = DivideUp(m_inputDataCount, numThreadsPerBlock);
	LAUNCH_KERNEL_ASYNC(ComputeSumOfExponentials, dim3(numBlocks), dim3(numThreadsPerBlock), m_deviceCalculationStream)(m_activationDataBuffer, m_inputDataCount,
		m_activationDataSize, m_exponentialsSumBuffer);
	CudaAssert(cudaGetLastError());

	// Dividing exponentials with their sum to get soft maximums.
	const uint c_blockWidth = 64;
	const uint c_blockHeight = (uint)Config::MAX_NUM_THREADS / c_blockWidth;
	blockDimensions = dim3(c_blockWidth, c_blockHeight);
	const uint c_maxGridBlocks = 128;
	const uint c_gridWidth = min(c_maxGridBlocks, DivideUp(m_inputDataCount, c_blockWidth));
	const uint c_gridHeight = min(c_maxGridBlocks / c_gridWidth, DivideUp(m_activationDataSize, c_blockHeight));
	gridDimensions = dim3(c_gridWidth, c_gridHeight);
	LAUNCH_KERNEL_ASYNC((DivideExponentialsWithSum<c_blockWidth>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_activationDataBuffer, m_inputDataCount,
		m_activationDataSize, m_exponentialsSumBuffer);
	CudaAssert(cudaGetLastError());
}

/*
	Calculates negative log likelihoods using LogSumExp formula.
*/
__global__ void __CalculateNegativeLogLikelihoods(float* inputActivations, uint* dataLabels, const uint numInputSamples, float* inputActivationsMaximums,
	float* exponentialsSumBuffer, float* nllsBuffer)
{
	const uint dataIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (dataIndex < numInputSamples)
	{
		nllsBuffer[dataIndex] = inputActivationsMaximums[dataIndex] + __logf(exponentialsSumBuffer[dataIndex]) -
			inputActivations[dataLabels[dataIndex] * numInputSamples + dataIndex];
	}
}

void SoftMaxLayer::CalculateNegativeLogLikelihoods(uint* dataLabels)
{
	if (m_NLLsBuffer == NULL)
	{
		size_t nllsBufferSize = m_inputDataCount * sizeof(float);
		CudaAssert(cudaMalloc<float>(&m_NLLsBuffer, nllsBufferSize));
		m_memoryConsumptionSize += nllsBufferSize;
	}

	const uint numThreadsPerBlock = 128;
	const uint numBlocks = DivideUp(m_inputDataCount, numThreadsPerBlock);
	LAUNCH_KERNEL_ASYNC(__CalculateNegativeLogLikelihoods, dim3(numBlocks), dim3(numThreadsPerBlock), m_deviceCalculationStream)(m_inputDataBuffer,
		dataLabels, m_inputDataCount, m_inputActivationsMaxBuffer, m_exponentialsSumBuffer, m_NLLsBuffer);
	CudaAssert(cudaGetLastError());
}

void SoftMaxLayer::DoForwardProp(PropagationMode propagationMode)
{
	StabilizeInputs();
	CalculateSoftMaximums();

	if (m_nextLayers[0]->GetLayerType() == LayerType::Output)
	{
		OutputLayer* outputLayer = static_cast<OutputLayer*>(m_nextLayers[0]);
		if (outputLayer->GetLossFunctionType() == LossFunctionType::CrossEntropy)
		{
			CalculateNegativeLogLikelihoods(outputLayer->GetDataLabels());
		}
		else
		{
			ShipAssert(false, "Currently not supported!");
		}
	}
}

/*
	Calculates input gradients in case of cross entropy loss in output layer.
*/
__global__ void CalculateCrossEntropyInputGradients(float* activations, uint* dataLabels, const uint dataCount, const uint numActivations,
	float* inputGradients)
{
	const uint c_dataIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const uint c_activationIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const uint c_activationsOffset = c_activationIndex * dataCount + c_dataIndex;

	if (c_dataIndex < dataCount && c_activationIndex < numActivations)
	{
		inputGradients[c_activationsOffset] = (dataLabels[c_dataIndex] == c_activationIndex ? 1.f : 0.f) - activations[c_activationsOffset];
	}
}

void SoftMaxLayer::CrossEntropyBackwardProp(uint* dataLabels)
{
	const uint c_blockWidth = 32;
	const uint c_blockHeight = 4;
	dim3 blockDimensions(c_blockWidth, c_blockHeight);
	const uint c_gridWidth = DivideUp(m_inputDataCount, c_blockWidth);
	const uint c_gridHeight = DivideUp(m_activationDataSize, c_blockHeight);
	dim3 gridDimensions(c_gridWidth, c_gridHeight);
	LAUNCH_KERNEL_ASYNC(CalculateCrossEntropyInputGradients, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_activationDataBuffer,
		dataLabels, m_inputDataCount, m_activationDataSize, m_inputGradientsBuffer);
	CudaAssert(cudaGetLastError());
}

void SoftMaxLayer::DoBackwardProp()
{
	if (m_nextLayers[0]->GetLayerType() == LayerType::Output)
	{
		OutputLayer* outputLayer = static_cast<OutputLayer*>(m_nextLayers[0]);
		if (outputLayer->GetLossFunctionType() == LossFunctionType::CrossEntropy)
		{
			CrossEntropyBackwardProp(outputLayer->GetDataLabels());
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