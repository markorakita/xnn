// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract neural network layer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/layer.cuh"

#include <chrono>

#include <curand_kernel.h>

#include "include/inputlayer.cuh"
#include "../include/matrixoperations.cuh"
#include "../include/neuralnet.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

Layer::Layer()
{
	m_inputLayerIndexInTier = -1;
	m_inputDataBuffer = NULL;
	m_inputGradientsBuffer = NULL;
	m_activationDataBuffer = NULL;
	m_activationGradientsBuffer = NULL;
	m_activationGradientsHelpBuffer = NULL;
	m_memoryConsumptionSize = 0;
}

Layer::~Layer()
{
	if (m_holdsInputData && m_inputDataBuffer != NULL)
	{
		CudaAssert(cudaFree(m_inputDataBuffer));
	}

	if (m_inputGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_inputGradientsBuffer));
	}

	if (m_activationDataBuffer != NULL)
	{
		CudaAssert(cudaFree(m_activationDataBuffer));
	}

	if (m_holdsActivationGradients && m_activationGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_activationGradientsBuffer));
	}

	if (m_holdsActivationGradients && m_activationGradientsHelpBuffer != NULL)
	{
		CudaAssert(cudaFree(m_activationGradientsHelpBuffer));
	}
}

void Layer::Reinitialize(uint newInputDataCount)
{
	m_inputDataCount = newInputDataCount;
	m_inputBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = m_inputBufferSize;
}

void Layer::LoadInputs()
{
	// If it holds the input data then it means it is connected with some layer trained on different GPU.
	if (m_holdsInputData)
	{
		if (m_prevLayers[0]->GetParallelismMode() == ParallelismMode::Model)
		{
			if (m_parallelismMode == ParallelismMode::Model)
			{
				if (m_prevLayers[0]->GetActivationDataCount() != m_inputDataCount)
				{
					Reinitialize(m_prevLayers[0]->GetActivationDataCount());
				}

				size_t prevBufferSize = m_prevLayers[0]->GetActivationBufferSize();
				size_t prevBufferLength = prevBufferSize / sizeof(float);
				for (size_t prevLayerIndex = 0; prevLayerIndex < m_prevLayers.size(); ++prevLayerIndex)
				{
					if (m_prevLayers[prevLayerIndex]->GetIndexInTier() == m_indexInTier)
					{
						CudaAssert(cudaMemcpyAsync(m_inputDataBuffer + prevLayerIndex * prevBufferLength, m_prevLayers[prevLayerIndex]->GetActivationDataBuffer(),
							prevBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
					}
					else
					{
						CudaAssert(cudaMemcpyPeerAsync(m_inputDataBuffer + prevLayerIndex * prevBufferLength, m_indexInTier, m_prevLayers[prevLayerIndex]->GetActivationDataBuffer(),
							m_prevLayers[prevLayerIndex]->GetIndexInTier(), prevBufferSize, m_deviceMemoryStream));
					}
				}
			}
			else if (m_parallelismMode == ParallelismMode::Data)
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
		else if (m_prevLayers[0]->GetParallelismMode() == ParallelismMode::Data)
		{
			if (m_parallelismMode == ParallelismMode::Model)
			{
				uint inputLayerIndexInTier = (uint)(m_inputLayerIndexInTier + 1);

				if (m_prevLayers[inputLayerIndexInTier]->GetActivationDataCount() != m_inputDataCount)
				{
					Reinitialize(m_prevLayers[inputLayerIndexInTier]->GetActivationDataCount());
				}

				if (inputLayerIndexInTier == m_indexInTier)
				{
					CudaAssert(cudaMemcpyAsync(m_inputDataBuffer, m_prevLayers[inputLayerIndexInTier]->GetActivationDataBuffer(), m_inputBufferSize,
						cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
				}
				else
				{
					CudaAssert(cudaMemcpyPeerAsync(m_inputDataBuffer, m_indexInTier, m_prevLayers[inputLayerIndexInTier]->GetActivationDataBuffer(),
						m_prevLayers[inputLayerIndexInTier]->GetIndexInTier(), m_inputBufferSize, m_deviceMemoryStream));
				}
			}
			else if (m_parallelismMode == ParallelismMode::Data)
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
	}
	else
	{
		if (m_prevLayers[0]->GetLayerType() == LayerType::Input)
		{
			InputLayer* inputLayer = static_cast<InputLayer*>(m_prevLayers[0]);

			if (inputLayer->GetActivationDataCount(m_indexInTier) != m_inputDataCount)
			{
				Reinitialize(inputLayer->GetActivationDataCount(m_indexInTier));
			}

			m_inputDataBuffer = inputLayer->GetActivationDataBuffer(m_indexInTier);
		}
		else
		{
			if (m_prevLayers[0]->GetActivationDataCount() != m_inputDataCount)
			{
				Reinitialize(m_prevLayers[0]->GetActivationDataCount());
			}

			m_inputDataBuffer = m_prevLayers[0]->GetActivationDataBuffer();
		}
	}
}

void Layer::LoadActivationGradients()
{
	if (m_layerType == LayerType::Input)
	{
		ShipAssert(false, "Shouldn't load gradients to input layer!");
	}
	else if (m_layerType == LayerType::Output)
	{
		// Nothing to load for output layer.
		return;
	}

	// If it holds the activation gradients then it means it is connected with some layer trained on different GPU.
	if (m_holdsActivationGradients)
	{
		if (m_nextLayers[0]->GetParallelismMode() == ParallelismMode::Model)
		{
			if (m_parallelismMode == ParallelismMode::Model || m_parallelismMode == ParallelismMode::Data)
			{
				size_t activationBufferLength = m_activationBufferSize / sizeof(float);

				// Copy over gradients from first of next layers.
				float* inputGradientsBuffer = m_parallelismMode == ParallelismMode::Data ? m_nextLayers[0]->GetInputGradientsBuffer() :
					m_nextLayers[0]->GetInputGradientsBuffer() + m_indexInTier * activationBufferLength;
				if (m_nextLayers[0]->GetIndexInTier() == m_indexInTier)
				{
					CudaAssert(cudaMemcpyAsync(m_activationGradientsBuffer, inputGradientsBuffer, m_activationBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
				}
				else
				{
					CudaAssert(cudaMemcpyPeerAsync(m_activationGradientsBuffer, m_indexInTier, inputGradientsBuffer, m_nextLayers[0]->GetIndexInTier(),
						m_activationBufferSize, m_deviceMemoryStream));
				}

				// Add up gradients from rest of next layers.
				if (m_nextLayers.size() > 1)
				{
					if (m_activationGradientsHelpBuffer == NULL)
					{
						CudaAssert(cudaMalloc<float>(&m_activationGradientsHelpBuffer, m_activationBufferSize));
						m_memoryConsumptionSize += m_activationBufferSize;
					}
					
					for (size_t nextLayerIndex = 1; nextLayerIndex < m_nextLayers.size(); ++nextLayerIndex)
					{
						// Copy gradients to temp buffer.
						inputGradientsBuffer = m_parallelismMode == ParallelismMode::Data ? m_nextLayers[nextLayerIndex]->GetInputGradientsBuffer() :
							m_nextLayers[nextLayerIndex]->GetInputGradientsBuffer() + m_indexInTier * activationBufferLength;
						if (m_nextLayers[nextLayerIndex]->GetIndexInTier() == m_indexInTier)
						{
							CudaAssert(cudaMemcpyAsync(m_activationGradientsHelpBuffer, inputGradientsBuffer, m_activationBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
						}
						else
						{
							CudaAssert(cudaMemcpyPeerAsync(m_activationGradientsHelpBuffer, m_indexInTier, inputGradientsBuffer, m_nextLayers[nextLayerIndex]->GetIndexInTier(),
								m_activationBufferSize, m_deviceMemoryStream));
						}

						// Add gradients from temp buffer to gradients buffer.
						CalculateElementWiseSum(m_activationGradientsBuffer, m_activationGradientsHelpBuffer, (uint)activationBufferLength, m_activationGradientsBuffer, m_deviceMemoryStream);
					}
				}
			}
			else
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
		else if (m_nextLayers[0]->GetParallelismMode() == ParallelismMode::Data)
		{
			if (m_parallelismMode == ParallelismMode::Model)
			{
				ShipAssert(false, "Currently not supported!");
			}
			else if (m_parallelismMode == ParallelismMode::Data)
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
	}
	else
	{
		m_activationGradientsBuffer = m_nextLayers[0]->GetInputGradientsBuffer();
	}
}

/*
	Initializes buffer with random values sampled from uniform distribution.
*/
__global__ void InitializeBufferUniform(float* buffer, const uint bufferLength, float rangeStart, float rangeEnd, curandState* curandStatesBuffer)
{
	const uint bufferOffset = blockIdx.x * blockDim.x + threadIdx.x;
	const float rangeSpan = rangeEnd - rangeStart;

	// Saving state to register for efficiency.
	curandState localCurandState = curandStatesBuffer[bufferOffset];

	for (uint bufferIndex = bufferOffset; bufferIndex < bufferLength; bufferIndex += gridDim.x * blockDim.x)
	{
		buffer[bufferIndex] = rangeStart + rangeSpan * curand_uniform(&localCurandState);
	}

	// Copying state back to global memory.
	// We need to do this since each generation of random number changes the state of the generator.
	curandStatesBuffer[bufferOffset] = localCurandState;
}

void Layer::InitializeBufferFromUniformDistribution(float* buffer, uint bufferLength, float rangeStart, float rangeEnd, curandState* curandStatesBuffer)
{
	dim3 blockDimensions(NeuralNet::c_numCurandThreadsPerBlock);
	dim3 gridDimensions(NeuralNet::c_numCurandBlocks);
	LAUNCH_KERNEL_ASYNC(InitializeBufferUniform, gridDimensions, blockDimensions, m_deviceCalculationStream)(buffer, bufferLength,
		rangeStart, rangeEnd, curandStatesBuffer);
	CudaAssert(cudaGetLastError());
}

/*
	Initializes buffer with random values sampled from normal distribution.
*/
__global__ void InitializeBufferGaussian(float* buffer, const uint bufferLength, float mean, float stDev, curandState* curandStatesBuffer)
{
	const uint bufferOffset = blockIdx.x * blockDim.x + threadIdx.x;

	// Saving state to register for efficiency.
	curandState localCurandState = curandStatesBuffer[bufferOffset];

	for (uint bufferIndex = bufferOffset; bufferIndex < bufferLength; bufferIndex += gridDim.x * blockDim.x)
	{
		buffer[bufferIndex] = stDev * curand_normal(&localCurandState) + mean;
	}

	// Copying state back to global memory.
	// We need to do this since each generation of random number changes the state of the generator.
	curandStatesBuffer[bufferOffset] = localCurandState;
}

void Layer::InitializeBufferFromNormalDistribution(float* buffer, uint bufferLength, float mean, float stDev, curandState* curandStatesBuffer)
{
	dim3 blockDimensions(NeuralNet::c_numCurandThreadsPerBlock);
	dim3 gridDimensions(NeuralNet::c_numCurandBlocks);
	LAUNCH_KERNEL_ASYNC(InitializeBufferGaussian, gridDimensions, blockDimensions, m_deviceCalculationStream)(buffer, bufferLength, mean, stDev, curandStatesBuffer);
	CudaAssert(cudaGetLastError());
}

/*
	Initializes buffer elements to constant value.
*/
__global__ void InitializeBufferToConstantKernel(float* buffer, const uint bufferLength, float initialValue)
{
	for (uint bufferIndex = blockIdx.x * blockDim.x + threadIdx.x; bufferIndex < bufferLength; bufferIndex += gridDim.x * blockDim.x)
	{
		buffer[bufferIndex] = initialValue;
	}
}

void Layer::InitializeBufferToConstant(float* buffer, uint bufferLength, float initialValue)
{
	const uint numBlocks = 128;
	const uint numThreadsPerBlock = 128;
	dim3 blockDimensions(numThreadsPerBlock);
	dim3 gridDimensions(min(numBlocks, DivideUp(bufferLength, numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(InitializeBufferToConstantKernel, gridDimensions, blockDimensions, m_deviceCalculationStream)(buffer, bufferLength, initialValue);
	CudaAssert(cudaGetLastError());
}

void Layer::SynchronizeCalculations()
{
	CudaAssert(cudaStreamSynchronize(m_deviceCalculationStream));
}

void Layer::SynchronizeMemoryOperations()
{
	CudaAssert(cudaStreamSynchronize(m_deviceMemoryStream));
}