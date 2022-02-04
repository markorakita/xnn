// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for abstract layer.
// Created: 12/07/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/testlayer.cuh"

#include <cuda_runtime.h>

#include "mock/include/mockinputlayer.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

TestLayer::TestLayer()
{
	// Registering tests.
	m_tests["initializebufferfromuniformdistribution"] = bind(&TestLayer::TestInitializeBufferFromUniformDistribution, this);
	m_tests["initializebufferfromnormaldistribution"] = bind(&TestLayer::TestInitializeBufferFromNormalDistribution, this);
	m_tests["initializebuffertoconstant"] = bind(&TestLayer::TestInitializeBufferToConstant, this);
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestLayer::TestInitializeBufferFromUniformDistribution()
{
	NeuralNet neuralNet(1);
	MockInputLayer mockInputLayer(1, 1000, 1, 128, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(false);
	Layer* mockInputLayerPt = static_cast<Layer*>(&mockInputLayer);

	const float rangeStart = -0.03f;
	const float rangeEnd = 0.07f;
	const uint activationDataBufferLength = (uint)(mockInputLayer.GetActivationBufferSize() / sizeof(float));
	mockInputLayerPt->InitializeBufferFromUniformDistribution(mockInputLayer.GetActivationDataBuffer(), activationDataBufferLength, rangeStart, rangeEnd, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayerPt->SynchronizeCalculations();

	float* hostBuffer;
	CudaAssert(cudaMallocHost<float>(&hostBuffer, mockInputLayer.GetActivationBufferSize()));
	CudaAssert(cudaMemcpy(hostBuffer, mockInputLayer.GetActivationDataBuffer(), mockInputLayer.GetActivationBufferSize(), cudaMemcpyDeviceToHost));

	float minValue = hostBuffer[0];
	float maxValue = hostBuffer[0];
	bool foundDiffThanZero = false;
	for (size_t i = 1; i < activationDataBufferLength; ++i)
	{
		minValue = min(minValue, hostBuffer[i]);
		maxValue = max(maxValue, hostBuffer[i]);

		foundDiffThanZero = foundDiffThanZero || hostBuffer[i] != 0.f;
	}

	return minValue > rangeStart && maxValue <= rangeEnd && foundDiffThanZero;
}

bool TestLayer::TestInitializeBufferFromNormalDistribution()
{
	NeuralNet neuralNet(1);
	MockInputLayer mockInputLayer(1, 1000, 1, 128, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(false);
	Layer* mockInputLayerPt = static_cast<Layer*>(&mockInputLayer);

	const float mean = 0.05f;
	const float stDev = 0.02f;
	const uint activationDataBufferLength = (uint)(mockInputLayer.GetActivationBufferSize() / sizeof(float));
	mockInputLayerPt->InitializeBufferFromNormalDistribution(mockInputLayer.GetActivationDataBuffer(), activationDataBufferLength, mean, stDev, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayerPt->SynchronizeCalculations();

	float* hostBuffer;
	CudaAssert(cudaMallocHost<float>(&hostBuffer, mockInputLayer.GetActivationBufferSize()));
	CudaAssert(cudaMemcpy(hostBuffer, mockInputLayer.GetActivationDataBuffer(), mockInputLayer.GetActivationBufferSize(), cudaMemcpyDeviceToHost));

	float minValue = hostBuffer[0];
	float maxValue = hostBuffer[0];
	float avgValue = 0.f;
	float cumAvgValue = hostBuffer[0];
	bool foundDiffThanZero = false;
	for (size_t i = 1; i < activationDataBufferLength; ++i)
	{
		minValue = min(minValue, hostBuffer[i]);
		maxValue = max(maxValue, hostBuffer[i]);
		cumAvgValue += hostBuffer[i];

		if (i % 1000 == 0 || i == (size_t)activationDataBufferLength - 1)
		{
			avgValue += cumAvgValue / activationDataBufferLength;
			cumAvgValue = 0.f;
		}

		foundDiffThanZero = foundDiffThanZero || hostBuffer[i] != 0.f;
	}

	return minValue < (mean - stDev) && maxValue > (mean + stDev) && abs(mean - avgValue) < 0.001f && foundDiffThanZero;

	return true;
}

bool TestLayer::TestInitializeBufferToConstant()
{
	NeuralNet neuralNet(1);
	MockInputLayer mockInputLayer(1, 1000, 1, 128, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(false);
	Layer* mockInputLayerPt = static_cast<Layer*>(&mockInputLayer);

	float initialValue = -0.7f;
	const uint activationDataBufferLength = (uint)(mockInputLayer.GetActivationBufferSize() / sizeof(float));
	mockInputLayerPt->InitializeBufferToConstant(mockInputLayer.GetActivationDataBuffer(), activationDataBufferLength, initialValue);
	mockInputLayerPt->SynchronizeCalculations();

	float* hostBuffer;
	CudaAssert(cudaMallocHost<float>(&hostBuffer, mockInputLayer.GetActivationBufferSize()));
	CudaAssert(cudaMemcpy(hostBuffer, mockInputLayer.GetActivationDataBuffer(), mockInputLayer.GetActivationBufferSize(), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < activationDataBufferLength; ++i)
	{
		if (hostBuffer[i] != initialValue)
		{
			return false;
		}
	}

	return true;
}