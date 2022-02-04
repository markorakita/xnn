// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network max pool layer, used in tests.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockmaxpoollayer.cuh"

#include <cuda_runtime.h>

#include "../../../../utils/include/asserts.cuh"
#include "../../../../utils/include/cudaasserts.cuh"

MockMaxPoolLayer::MockMaxPoolLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
	uint unitHeight, int paddingX, int paddingY, uint unitStride)
	:
	MaxPoolLayer(ParallelismMode::Model, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, true, unitWidth,
		unitHeight, paddingX, paddingY, unitStride, true)
{
}

void MockMaxPoolLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	// Allocating input data buffer.
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating activation data buffer.
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

MockMaxPoolLayer::~MockMaxPoolLayer()
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

void MockMaxPoolLayer::LoadInputs()
{
	ShipAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockMaxPoolLayer::LoadActivationGradients()
{
	ShipAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockMaxPoolLayer::DoForwardProp(PropagationMode propagationMode)
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint channel = 0; channel < m_inputNumChannels; ++channel)
		{
			const uint c_activationChannelOffset = channel * m_activationDataSize * m_inputDataCount;
			const uint c_dataChannelOffset = channel * m_inputDataSize * m_inputDataCount;
			int startY = -m_paddingY;
			for (uint unitY = 0; unitY < m_numUnitsY; ++unitY)
			{
				int startX = -m_paddingX;
				for (uint unitX = 0; unitX < m_numUnitsX; ++unitX)
				{
					const uint c_activationDataIndex = c_activationChannelOffset + (unitY * m_numUnitsX + unitX) * m_inputDataCount + dataIndex;
					m_activationDataBuffer[c_activationDataIndex] = -FLT_MAX;
					for (int currY = startY; currY < startY + (int)m_unitHeight; ++currY)
					{
						for (int currX = startX; currX < startX + (int)m_unitWidth; ++currX)
						{
							if (currY >= 0 && currY < (int)m_inputDataHeight && currX >= 0 && currX < (int)m_inputDataWidth)
							{
								m_activationDataBuffer[c_activationDataIndex] = fmaxf(m_activationDataBuffer[c_activationDataIndex],
									m_inputDataBuffer[c_dataChannelOffset + (currY * m_inputDataWidth + currX) * m_inputDataCount + dataIndex]);
							}
						}
					}
					startX += m_unitStride;
				}
				startY += m_unitStride;
			}
		}
	}
}

void MockMaxPoolLayer::DoBackwardProp()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint channel = 0; channel < m_inputNumChannels; ++channel)
		{
			const uint c_activationChannelOffset = channel * m_activationDataSize * m_inputDataCount + dataIndex;
			const uint c_dataChannelOffset = channel * m_inputDataSize * m_inputDataCount + dataIndex;
			for (uint pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)			
			{
				for (uint pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
				{
					const uint c_dataPixelOffset = c_dataChannelOffset + (pixelY * m_inputDataWidth + pixelX) * m_inputDataCount;
					float data = m_inputDataBuffer[c_dataPixelOffset];

					// Calculating indexes of units whose activation this pixel affected.
					const uint c_firstUnitX = (uint)m_paddingX + pixelX < m_unitWidth ? 0 : ((uint)m_paddingX + pixelX - m_unitWidth) / m_unitStride + 1;
					const uint c_firstUnitY = (uint)m_paddingY + pixelY < m_unitHeight ? 0 : ((uint)m_paddingY + pixelY - m_unitHeight) / m_unitStride + 1;
					const uint c_lastUnitX = min(m_numUnitsX, ((uint)m_paddingX + pixelX) / m_unitStride + 1);
					const uint c_lastUnitY = min(m_numUnitsY, ((uint)m_paddingY + pixelY) / m_unitStride + 1);
					
					// Calculating pixel gradient.
					float calculatedGradient = 0.f;
					for (uint unitY = c_firstUnitY; unitY < c_lastUnitY; ++unitY)
					{
						for (uint unitX = c_firstUnitX; unitX < c_lastUnitX; ++unitX)
						{
							const uint c_unitOffset = c_activationChannelOffset + (unitY * m_numUnitsX + unitX) * m_inputDataCount;
							float activation = m_activationDataBuffer[c_unitOffset];
							float activationGradient = m_activationGradientsBuffer[c_unitOffset];

							calculatedGradient += fabs((double)data - activation) < 0.000000001f ? activationGradient : 0.f;
						}
					}

					m_inputGradientsBuffer[c_dataPixelOffset] = calculatedGradient;
				}
			}
		}
	}
}