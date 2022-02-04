// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for max pool layer.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testmaxpoollayer.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mockmaxpoollayer.cuh"
#include "mock/include/mockoutputlayer.cuh"
#include "../../include/testingutils.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/maxpoollayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

TestMaxPoolLayer::TestMaxPoolLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestMaxPoolLayer::TestDoForwardProp, this);
	m_tests["dobackwardprop"] = bind(&TestMaxPoolLayer::TestDoBackwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestMaxPoolLayer::TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
	uint unitHeight, int paddingLeft, int paddingTop, uint unitStride)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockMaxPoolLayer mockMaxPoolLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, unitWidth, unitHeight, paddingLeft, paddingTop,
		unitStride);
	mockMaxPoolLayer.AllocateBuffers(allocateTrainBuffers);
	mockMaxPoolLayer.AddPrevLayer(&mockInputLayer);
	MaxPoolLayer maxPoolLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false, unitWidth, unitHeight,
		paddingLeft, paddingTop, unitStride, false);
	maxPoolLayer.AllocateBuffers(allocateTrainBuffers);
	maxPoolLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockMaxPoolLayer.LoadInputs();
	maxPoolLayer.LoadInputs();
	maxPoolLayer.DoForwardProp(propagationMode);
	mockMaxPoolLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockMaxPoolLayer.GetActivationBufferSize();
	float* maxPoolLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&maxPoolLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(maxPoolLayerActivationBuffer, maxPoolLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockMaxPoolLayerActivationBuffer = mockMaxPoolLayer.GetActivationDataBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.000001f;
	CompareBuffers(maxPoolLayerActivationBuffer, mockMaxPoolLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(maxPoolLayerActivationBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock max pool activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All max pool activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) +
			"; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
		return false;
	}

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Unit width: " << unitWidth <<
		"; Padding left: " << paddingLeft << "; Unit stride: " << unitStride << endl;
	return true;
}

bool TestMaxPoolLayer::TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
	uint unitHeight, int paddingLeft, int paddingTop, uint unitStride)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = true;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockMaxPoolLayer mockMaxPoolLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, unitWidth, unitHeight, paddingLeft, paddingTop, unitStride);
	mockMaxPoolLayer.AllocateBuffers(allocateTrainBuffers);
	mockMaxPoolLayer.AddPrevLayer(&mockInputLayer);
	MaxPoolLayer maxPoolLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false, unitWidth, unitHeight,
		paddingLeft, paddingTop, unitStride, false);
	maxPoolLayer.AllocateBuffers(allocateTrainBuffers);
	maxPoolLayer.AddPrevLayer(&mockInputLayer);
	// TODO: use constructor with mean and std dev, when you experimentally decide values for those
	MockOutputLayer mockOutputLayer(maxPoolLayer.GetActivationDataSize() * maxPoolLayer.GetActivationNumChannels(), inputDataCount, LossFunctionType::CrossEntropy,
		false, 0, neuralNet.GetCurandStatesBuffers()[0]);
	mockOutputLayer.AllocateBuffers(allocateTrainBuffers);
	mockMaxPoolLayer.AddNextLayer(&mockOutputLayer);
	maxPoolLayer.AddNextLayer(&mockOutputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockMaxPoolLayer.LoadInputs();
	maxPoolLayer.LoadInputs();
	maxPoolLayer.DoForwardProp(propagationMode);
	mockMaxPoolLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	maxPoolLayer.LoadActivationGradients();
	maxPoolLayer.DoBackwardProp();
	mockMaxPoolLayer.LoadActivationGradients();
	mockMaxPoolLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* maxPoolLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&maxPoolLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(maxPoolLayerInputGradientsBuffer, maxPoolLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockMaxPoolLayerInputGradientsBuffer = mockMaxPoolLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.000001f;
	CompareBuffers(maxPoolLayerInputGradientsBuffer, mockMaxPoolLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	for (size_t i = 0; i < inputGradientsBufferLength; ++i)
	{
		if (abs(maxPoolLayerInputGradientsBuffer[i] - mockMaxPoolLayerInputGradientsBuffer[i]) > 0.00001f)
		{
			cout << "obican: " << maxPoolLayerInputGradientsBuffer[i] << "    ,     " << "mock: " << mockMaxPoolLayerInputGradientsBuffer[i] << endl;
			cout << i << endl;
		}
	}

	CudaAssert(cudaFreeHost(maxPoolLayerInputGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock max pool input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) +
			"; Unit stride: " + to_string(unitStride));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All max pool input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) +
			"; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
		return false;
	}

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Unit width: " << unitWidth <<
		"; Padding left: " << paddingLeft << "; Unit stride: " << unitStride << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestMaxPoolLayer::TestDoForwardProp()
{
	bool testPassed =

	// lastBatch == true

	// m_inputNumChannels % 16 == 0
	TestSingleForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 33 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 33 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 57 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 17 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&

	
	// lastBatch == false

	// m_inputDataCount % 128 == 0

	// m_inputNumChannels % 16 == 0
	TestSingleForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&

	// m_inputDataCount % 64 == 0

	// m_inputNumChannels % 16 == 0
	TestSingleForwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&

	// m_inputDataCount % 32 == 0

	// m_inputNumChannels % 16 == 0
	TestSingleForwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);

	return testPassed;
}

bool TestMaxPoolLayer::TestDoBackwardProp()
{
	bool testPassed =

	// lastBatch == true

	// m_inputNumChannels % 16 == 0
	TestSingleBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 33 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 33 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	// TODO: Currently unsupported, uncomment here and below if you support this one day.
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 57 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 17 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&


	// lastBatch == false

	// m_inputDataCount % 128 == 0

	// m_inputNumChannels % 16 == 0
	TestSingleBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&

	// m_inputDataCount % 64 == 0

	// m_inputNumChannels % 16 == 0
	TestSingleBackwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/) &&
	// m_inputNumChannels % 16 != 0
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&

	// m_inputDataCount % 32 == 0

	// m_inputNumChannels % 16 == 0
	TestSingleBackwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/) &&
	//TestSingleBackwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/) &&

	return testPassed;
}