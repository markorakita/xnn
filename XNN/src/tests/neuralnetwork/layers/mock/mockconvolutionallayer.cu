// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network convolutional layer, used in tests.
// Created: 01/27/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockconvolutionallayer.cuh"

#include <chrono>

#include <cuda_runtime.h>

#include "../../mock/include/mockactivationfunctions.cuh"
#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"

MockConvolutionalLayer::MockConvolutionalLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
	uint filterHeight, uint numFilterChannels, float filtersUpdateMomentum, float filtersUpdateDecay, float filtersUpdateLearningRateProgressStep,
	float filtersUpdateStartingLearningRate, float filtersUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay,
	float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, int paddingX, int paddingY,
	uint stride, ActivationType activationType, float activationAlpha)
	:
	MockWeightsLayer(0, (size_t)numFilters * filterWidth * filterHeight * numFilterChannels * sizeof(float), inputNumChannels* filterWidth* filterHeight,
		filtersUpdateMomentum, filtersUpdateDecay, filtersUpdateLearningRateProgressStep, filtersUpdateStartingLearningRate, filtersUpdateLearningRateUpdateFactor,
		numFilters * sizeof(float), biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate,
		biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha)
{
	m_layerType = LayerType::Convolutional;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

	m_numFilters = numFilters;
	m_filterWidth = filterWidth;
	m_filterHeight = filterHeight;
	m_filterSize = m_filterWidth * m_filterHeight;
	m_numFilterChannels = numFilterChannels;

	m_paddingX = paddingX;
	m_paddingY = paddingY;
	m_stride = stride;
	m_numPatchesX = 1 + (uint)ceil((2.0 * paddingX + m_inputDataWidth - m_filterWidth) / m_stride);
	m_numPatchesY = 1 + (uint)ceil((2.0 * paddingY + m_inputDataHeight - m_filterHeight) / m_stride);

	m_activationNumChannels = m_numFilters;
	m_activationDataWidth = m_numPatchesX;
	m_activationDataHeight = m_numPatchesY;
	m_activationDataSize = m_activationDataWidth * m_activationDataHeight;

	m_inputBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = (size_t)m_numFilters * m_activationDataSize * m_inputDataCount * sizeof(float);

	m_holdsActivationGradients = true;

	m_preactivationDataBuffer = NULL;
	m_preactivationGradientsBuffer = NULL;
}

void MockConvolutionalLayer::AllocateBuffers(bool allocateTrainBuffers)
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

MockConvolutionalLayer::~MockConvolutionalLayer()
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

void MockConvolutionalLayer::LoadInputs()
{
	ShipAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockConvolutionalLayer::LoadActivationGradients()
{
	ShipAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockConvolutionalLayer::CalculatePreactivations()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
		{
			const uint c_activationChannelOffset = m_activationDataSize * filterIndex * m_inputDataCount;
			for (uint channel = 0; channel < m_inputNumChannels; ++channel)
			{
				const uint c_filtersChannelOffset = channel * m_numFilters * m_filterSize;
				const uint c_dataChannelOffset = channel * m_inputDataCount * m_inputDataSize;
				int startY = -m_paddingY;
				for (uint patchY = 0; patchY < m_numPatchesY; ++patchY)
				{
					int startX = -m_paddingX;
					for (uint patchX = 0; patchX < m_numPatchesX; ++patchX)
					{
						const uint c_activationDataIndex = c_activationChannelOffset + (patchY * m_numPatchesX + patchX) * m_inputDataCount + dataIndex;
						if (channel == 0)
						{
							m_preactivationDataBuffer[c_activationDataIndex] = 0.0f;
						}
						for (int currY = startY; currY < startY + (int)m_filterHeight; ++currY)
						{
							for (int currX = startX; currX < startX + (int)m_filterWidth; ++currX)
							{
								if (currY >= 0 && currY < (int)m_inputDataHeight && currX >= 0 && currX < (int)m_inputDataWidth)
								{
									m_preactivationDataBuffer[c_activationDataIndex] +=
										m_weightsBuffer[c_filtersChannelOffset + ((currY - startY) * m_filterWidth + currX - startX) * m_numFilters + filterIndex] *
										m_inputDataBuffer[c_dataChannelOffset + (currY * m_inputDataWidth + currX) * m_inputDataCount + dataIndex];
								}
							}
						}
						startX += m_stride;
					}
					startY += m_stride;
				}
			}
		}
	}
}

void MockConvolutionalLayer::AddBiases()
{
	const uint c_width = m_inputDataCount * m_numPatchesY * m_numPatchesX;
	for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
	{
		for (uint i = 0; i < c_width; ++i)
		{
			m_preactivationDataBuffer[filterIndex * c_width + i] += m_biasesBuffer[filterIndex];
		}
	}
}

void MockConvolutionalLayer::CalculateActivations()
{
	ApplyActivationBF(m_activationType, m_activationAlpha, m_preactivationDataBuffer, (uint)(m_activationBufferSize / sizeof(float)),
		m_activationDataBuffer);
}

void MockConvolutionalLayer::DoForwardProp(PropagationMode propagationMode)
{
	CalculatePreactivations();
	AddBiases();
	CalculateActivations();
}

void MockConvolutionalLayer::CalculateBiasesGradients()
{
	uint batchSize = m_parallelismMode == ParallelismMode::Model ? m_inputDataCount : m_tierSize * m_inputDataCount;
	const uint c_width = m_inputDataCount * m_numPatchesY * m_numPatchesX;
	for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
	{
		float biasGradient = 0.f;
		for (uint i = 0; i < c_width; ++i)
		{
			biasGradient += m_preactivationGradientsBuffer[filterIndex * c_width + i];
		}

		m_biasesGradientsBuffer[filterIndex] = biasGradient / (float)batchSize;
	}
}

void MockConvolutionalLayer::CalculateWeightsGradients()
{
	// Initializing gradients to zero.
	size_t filtersBufferLength = m_weightsBufferSize / sizeof(float);
	for (size_t i = 0; i < filtersBufferLength; ++i)
	{
		m_weightsGradientsBuffer[i] = 0.f;
	}

	// Calculating gradients.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
		{
			const uint c_activationChannelOffset = m_activationDataSize * filterIndex * m_inputDataCount;
			for (uint channel = 0; channel < m_inputNumChannels; ++channel)
			{
				const uint c_filtersChannelOffset = channel * m_numFilters * m_filterSize;
				const uint c_dataChannelOffset = channel * m_inputDataCount * m_inputDataSize;
				int startY = -m_paddingY;
				for (uint patchY = 0; patchY < m_numPatchesY; ++patchY)
				{
					int startX = -m_paddingX;
					for (uint patchX = 0; patchX < m_numPatchesX; ++patchX)
					{
						const uint c_activationDataIndex = c_activationChannelOffset + (patchY * m_numPatchesX + patchX) * m_inputDataCount + dataIndex;
						if (channel == 0)
						{
							m_preactivationDataBuffer[c_activationDataIndex] = 0.0f;
						}
						for (int currY = startY; currY < startY + (int)m_filterHeight; ++currY)
						{
							for (int currX = startX; currX < startX + (int)m_filterWidth; ++currX)
							{
								if (currY >= 0 && currY < (int)m_inputDataHeight && currX >= 0 && currX < (int)m_inputDataWidth)
								{
									m_weightsGradientsBuffer[c_filtersChannelOffset + ((currY - startY) * m_filterWidth + currX - startX) * m_numFilters + filterIndex] +=
										m_preactivationGradientsBuffer[c_activationDataIndex] *
										m_inputDataBuffer[c_dataChannelOffset + (currY * m_inputDataWidth + currX) * m_inputDataCount + dataIndex];
								}
							}
						}
						startX += (int)m_stride;
					}
					startY += (int)m_stride;
				}
			}
		}
	}

	// Scaling gradients with batch size.
	float batchSize = m_parallelismMode == ParallelismMode::Model ? (float)m_inputDataCount : (float)(m_tierSize * m_inputDataCount);
	for (size_t i = 0; i < filtersBufferLength; ++i)
	{
		m_weightsGradientsBuffer[i] /= batchSize;
	}
}

void MockConvolutionalLayer::CalculateInputGradients()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint channel = 0; channel < m_inputNumChannels; ++channel)
		{
			for (uint pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)
			{
				for (uint pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
				{
					const uint c_currPixel = pixelY * m_inputDataWidth + pixelX;
					const uint c_firstPatchX = pixelX + m_paddingX < m_filterWidth ? 0 : (pixelX + m_paddingX - m_filterWidth) / m_stride + 1;
					const uint c_firstPatchY = pixelY + m_paddingY < m_filterHeight ? 0 : (pixelY + m_paddingY - m_filterHeight) / m_stride + 1;
					const uint c_lastPatchX = min(m_numPatchesX, (pixelX + m_paddingX) / m_stride + 1);
					const uint c_lastPatchY = min(m_numPatchesY, (pixelY + m_paddingY) / m_stride + 1);

					float gradient = 0.0f;

					for (uint currPatchY = c_firstPatchY; currPatchY < c_lastPatchY; ++currPatchY)
					{
						const uint c_filterPixelY = pixelY + m_paddingY - currPatchY * m_stride;
						for (uint currPatchX = c_firstPatchX; currPatchX < c_lastPatchX; ++currPatchX)
						{
							const uint c_filterPixelX = pixelX + m_paddingX - currPatchX * m_stride;
							const uint c_filterPixel = c_filterPixelY * m_filterWidth + c_filterPixelX;
							const uint c_currPatch = currPatchY * m_numPatchesX + currPatchX;

							for (uint currFilter = 0; currFilter < m_numFilters; ++currFilter)
							{
								gradient += m_weightsBuffer[(channel * m_filterSize + c_filterPixel) * m_numFilters + currFilter] *
									m_preactivationGradientsBuffer[(currFilter * m_numPatchesX * m_numPatchesY + c_currPatch) * m_inputDataCount + dataIndex];
							}
						}
					}

					m_inputGradientsBuffer[(channel * m_inputDataSize + c_currPixel) * m_inputDataCount + dataIndex] = gradient;
				}
			}
		}
	}
}

void MockConvolutionalLayer::CalculatePreactivationsGradients()
{
	CalculatePreactivationGradientsBF(m_activationType, m_activationAlpha, m_activationGradientsBuffer, m_activationDataBuffer,
		(uint)(m_activationBufferSize / sizeof(float)), m_preactivationGradientsBuffer);
}

void MockConvolutionalLayer::DoBackwardProp()
{
	CalculatePreactivationsGradients();
	CalculateInputGradients();
	CalculateWeightsGradients();
	CalculateBiasesGradients();
}