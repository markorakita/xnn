// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testsoftmaxlayer.cuh"

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mocksoftmaxlayer.cuh"
#include "../../include/testingutils.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/outputlayer.cuh"
#include "../../../neuralnetwork/layers/include/softmaxlayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

const float TestSoftMaxLayer::c_inputActivationsMean = -6.f;
const float TestSoftMaxLayer::c_inputActivationsStDev = 3.f;

TestSoftMaxLayer::TestSoftMaxLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestSoftMaxLayer::TestDoForwardProp, this);
	m_tests["dobackwardprop"] = bind(&TestSoftMaxLayer::TestDoBackwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestSoftMaxLayer::TestSingleForwardProp(uint inputDataSize, uint inputDataCount)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;

	// Creating layers.
	MockInputLayer mockInputLayer(1, inputDataSize, 1, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockSoftMaxLayer mockSoftMaxLayer(inputDataSize, inputDataCount);
	mockSoftMaxLayer.AllocateBuffers(allocateTrainBuffers);
	mockSoftMaxLayer.AddPrevLayer(&mockInputLayer);
	SoftMaxLayer softMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, false);
	softMaxLayer.AllocateBuffers(allocateTrainBuffers);
	softMaxLayer.AddPrevLayer(&mockInputLayer);
	OutputLayer outputLayer(0, 0, inputDataSize, inputDataCount, inputDataCount, LossFunctionType::CrossEntropy, true, 5, 0);
	outputLayer.AllocateBuffers(allocateTrainBuffers);
	mockSoftMaxLayer.AddNextLayer(&outputLayer);
	softMaxLayer.AddNextLayer(&outputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromNormalDistribution(c_inputActivationsMean, c_inputActivationsStDev);

	// Creating pseudo random labels.
	vector<uint> labels;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		labels.push_back((57 * i * i) % inputDataSize);
	}
	outputLayer.LoadDataLabels(labels);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	softMaxLayer.LoadInputs();
	mockSoftMaxLayer.LoadInputs();
	softMaxLayer.DoForwardProp(propagationMode);
	mockSoftMaxLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaStreamSynchronize(0));

	// Transferring activations to host.
	size_t activationsBufferSize = mockSoftMaxLayer.GetActivationBufferSize();
	float* softMaxLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&softMaxLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(softMaxLayerActivationBuffer, softMaxLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockSoftMaxLayerActivationBuffer = mockSoftMaxLayer.GetActivationDataBuffer();
	const float maxActivationsDiff = 0.000001f;
	const float maxActivationsDiffPercentage = 0.0001f;
	const float maxActivationsDiffPercentageThreshold = 0.0000001f;
	CompareBuffers(softMaxLayerActivationBuffer, mockSoftMaxLayerActivationBuffer, activationsBufferLength, maxActivationsDiff,
		maxActivationsDiffPercentage, maxActivationsDiffPercentageThreshold, correctResult, numDifferences, firstDifference, firstDifferentMock,
		firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	bool foundIrregularMock = false;
	float irregularMock = 0.0f;
	bool foundIrregularReg = false;
	float irregularReg = 0.0f;
	for (size_t i = 0; i < activationsBufferLength; ++i)
	{
		if (mockSoftMaxLayerActivationBuffer[i] < 0.0f || mockSoftMaxLayerActivationBuffer[i] > 1.0f)
		{
			foundIrregularMock = true;
			irregularMock = mockSoftMaxLayerActivationBuffer[i];
		}
		if (softMaxLayerActivationBuffer[i] < 0.0f || softMaxLayerActivationBuffer[i] > 1.0f)
		{
			foundIrregularReg = true;
			irregularReg = softMaxLayerActivationBuffer[i];
		}
	}

	CudaAssert(cudaFreeHost(softMaxLayerActivationBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock softmax layer activations are zeros! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All softmax layer activations are zeros! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount));
		return false;
	}
	else if (foundIrregularMock)
	{
		EmitWarning("Found irregular mock softmax layer activation! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount) + "; Irregular value: " + to_string(irregularMock));
		return false;
	}
	else if (foundIrregularReg)
	{
		EmitWarning("Found irregular softmax layer activation! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount) + "; Irregular value: " + to_string(irregularReg));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
			"; Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount));
		return false;
	}

	// Transferring cross entropy loss buffer to host.
	size_t crossEntropyLossBufferSize = inputDataCount * sizeof(float);
	float* softMaxLayerCrossEntropyLossBuffer;
	CudaAssert(cudaMallocHost<float>(&softMaxLayerCrossEntropyLossBuffer, crossEntropyLossBufferSize));
	CudaAssert(cudaMemcpy(softMaxLayerCrossEntropyLossBuffer, softMaxLayer.GetNegativeLogLikelihoodsBuffer(), crossEntropyLossBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	const float maxLossDiff = 0.00001f;
	const float maxLossDiffPercentage = 0.0001f;
	const float maxLossDiffPercentageThreshold = 0.000001f;
	CompareBuffers(softMaxLayerCrossEntropyLossBuffer, mockSoftMaxLayer.GetNegativeLogLikelihoodsBuffer(), inputDataCount, maxLossDiff,
		maxLossDiffPercentage, maxLossDiffPercentageThreshold, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg,
		foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(softMaxLayerCrossEntropyLossBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock softmax layer cross entropy losses are zeros! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All softmax layer cross entropy losses are zeros! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect cross entropy losses calculation! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock cross entropy loss: " + to_string(firstDifferentMock) + "; First different regular cross entropy loss: " + to_string(firstDifferentReg) +
			"; Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount));
		return false;
	}

	cout << "Forward prop passed. Input data size: " << inputDataSize << "; Input data count: " << inputDataCount << endl;
	return true;
}

bool TestSoftMaxLayer::TestSingleBackwardProp(uint inputDataSize, uint inputDataCount)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = true;

	// Creating layers.
	MockInputLayer mockInputLayer(1, inputDataSize, 1, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockSoftMaxLayer mockSoftMaxLayer(inputDataSize, inputDataCount);
	mockSoftMaxLayer.AllocateBuffers(allocateTrainBuffers);
	mockSoftMaxLayer.AddPrevLayer(&mockInputLayer);
	SoftMaxLayer softMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, false);
	softMaxLayer.AllocateBuffers(allocateTrainBuffers);
	softMaxLayer.AddPrevLayer(&mockInputLayer);
	OutputLayer outputLayer(0, 0, inputDataSize, inputDataCount, inputDataCount, LossFunctionType::CrossEntropy, true, 5, 0);
	outputLayer.AllocateBuffers(allocateTrainBuffers);
	mockSoftMaxLayer.AddNextLayer(&outputLayer);
	softMaxLayer.AddNextLayer(&outputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromNormalDistribution(c_inputActivationsMean, c_inputActivationsStDev);

	// Creating pseudo random labels.
	vector<uint> labels;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		labels.push_back((57 * i * i) % inputDataSize);
	}
	outputLayer.LoadDataLabels(labels);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	softMaxLayer.LoadInputs();
	mockSoftMaxLayer.LoadInputs();
	softMaxLayer.DoForwardProp(propagationMode);
	mockSoftMaxLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaStreamSynchronize(0));

	// Doing backward prop.
	softMaxLayer.DoBackwardProp();
	mockSoftMaxLayer.DoBackwardProp();
	CudaAssert(cudaStreamSynchronize(0));

	// Transferring results to host.
	size_t gradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* softMaxLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&softMaxLayerInputGradientsBuffer, gradientsBufferSize));
	CudaAssert(cudaMemcpy(softMaxLayerInputGradientsBuffer, softMaxLayer.GetInputGradientsBuffer(), gradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t gradientsBufferLength = gradientsBufferSize / sizeof(float);
	const float* mockSoftMaxLayerInputGradientsBuffer = mockSoftMaxLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.0001f;
	const float maxDiffPercentageThreshold = 0.0000001f;
	CompareBuffers(softMaxLayerInputGradientsBuffer, mockSoftMaxLayerInputGradientsBuffer, gradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(softMaxLayerInputGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock softmax layer input gradients are zeros! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All softmax layer input gradients are zeros! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
			"; Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount));
		return false;
	}

	cout << "Backward prop passed. Input data size: " << inputDataSize << "; Input data count: " << inputDataCount << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestSoftMaxLayer::TestDoForwardProp()
{
	vector<uint> inputDataSizes { 2, 10, 100, 1000 };
	vector<uint> inputDataCounts { 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025 };

	for (size_t i = 0; i < inputDataSizes.size(); ++i)
	{
		for (size_t j = 0; j < inputDataCounts.size(); ++j)
		{
			if (!TestSingleForwardProp(inputDataSizes[i], inputDataCounts[j]))
			{
				return false;
			}
		}
	}

	return true;
}

bool TestSoftMaxLayer::TestDoBackwardProp()
{
	vector<uint> inputDataSizes{ 2, 3, 4, 5, 10, 100, 1000 };
	vector<uint> inputDataCounts{ 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025 };

	for (size_t i = 0; i < inputDataSizes.size(); ++i)
	{
		for (size_t j = 0; j < inputDataCounts.size(); ++j)
		{
			if (!TestSingleBackwardProp(inputDataSizes[i], inputDataCounts[j]))
			{
				return false;
			}
		}
	}

	return true;
}