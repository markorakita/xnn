// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testdropoutlayer.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "mock/include/mockdropoutlayer.cuh"
#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mockoutputlayer.cuh"
#include "../../include/testingutils.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/dropoutlayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

TestDropoutLayer::TestDropoutLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestDropoutLayer::TestDoForwardProp, this);
	m_tests["dobackwardprop"] = bind(&TestDropoutLayer::TestDoBackwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestDropoutLayer::TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability)
{
	NeuralNet neuralNet(1);
	// Dropout filters used in forward propagation are allocated only during training.
	const bool allocateTrainBuffers = true;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockDropoutLayer mockDropoutLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, dropProbability, neuralNet.GetCurandStatesBuffers()[0]);
	mockDropoutLayer.AllocateBuffers(allocateTrainBuffers);
	mockDropoutLayer.AddPrevLayer(&mockInputLayer);
	DropoutLayer dropoutLayer(ParallelismMode::Data, 0, 0, NULL, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		dropProbability, true, false);
	dropoutLayer.AllocateBuffers(allocateTrainBuffers);
	dropoutLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockDropoutLayer.LoadInputs();
	mockDropoutLayer.DoForwardProp(propagationMode);
	dropoutLayer.CopyDropoutFilterFromHost(mockDropoutLayer.GetDropoutFilter());
	dropoutLayer.LoadInputs();
	dropoutLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockDropoutLayer.GetActivationBufferSize();
	float* dropoutLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&dropoutLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(dropoutLayerActivationBuffer, dropoutLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockDropoutLayerActivationBuffer = mockDropoutLayer.GetActivationDataBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.00000001f;
	CompareBuffers(dropoutLayerActivationBuffer, mockDropoutLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(dropoutLayerActivationBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock dropout layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
			to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
			to_string(dropProbability));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All dropout layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
			to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
			to_string(dropProbability));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data width: " + to_string(inputDataWidth) + "; Input data height: " +
			to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " + to_string(dropProbability));
		return false;
	}

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input width: " << inputDataWidth << "; Input height: " << inputDataHeight <<
		"; Input data count: " << inputDataCount << "; Drop probability: " << dropProbability << endl;
	return true;
}

bool TestDropoutLayer::TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = true;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockDropoutLayer mockDropoutLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, dropProbability, neuralNet.GetCurandStatesBuffers()[0]);
	mockDropoutLayer.AllocateBuffers(allocateTrainBuffers);
	mockDropoutLayer.AddPrevLayer(&mockInputLayer);
	DropoutLayer dropoutLayer(ParallelismMode::Data, 0, 0, NULL, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		dropProbability, true, false);
	dropoutLayer.AllocateBuffers(allocateTrainBuffers);
	dropoutLayer.AddPrevLayer(&mockInputLayer);
	// TODO: use constructor with mean and std dev, when you experimentally decide values for those
	MockOutputLayer outputLayer(inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, LossFunctionType::CrossEntropy, false, 0,
		neuralNet.GetCurandStatesBuffers()[0]);
	outputLayer.AllocateBuffers(allocateTrainBuffers);
	mockDropoutLayer.AddNextLayer(&outputLayer);
	dropoutLayer.AddNextLayer(&outputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockDropoutLayer.LoadInputs();
	mockDropoutLayer.DoForwardProp(propagationMode);
	dropoutLayer.CopyDropoutFilterFromHost(mockDropoutLayer.GetDropoutFilter());
	dropoutLayer.LoadInputs();
	dropoutLayer.DoForwardProp(propagationMode);
	outputLayer.DoBackwardProp();
	dropoutLayer.LoadActivationGradients();
	dropoutLayer.DoBackwardProp();
	mockDropoutLayer.LoadActivationGradients();
	mockDropoutLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t gradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* dropoutLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&dropoutLayerInputGradientsBuffer, gradientsBufferSize));
	CudaAssert(cudaMemcpy(dropoutLayerInputGradientsBuffer, dropoutLayer.GetInputGradientsBuffer(), gradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t gradientsBufferLength = gradientsBufferSize / sizeof(float);
	const float* mockDropoutLayerInputGradientsBuffer = mockDropoutLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.00000001f;
	CompareBuffers(dropoutLayerInputGradientsBuffer, mockDropoutLayerInputGradientsBuffer, gradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(dropoutLayerInputGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock dropout layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
			to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
			to_string(dropProbability));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All dropout layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
			to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
			to_string(dropProbability));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data width: " + to_string(inputDataWidth) + "; Input data height: " +
			to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " + to_string(dropProbability));
		return false;
	}

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input width: " << inputDataWidth << "; Input height: " << inputDataHeight <<
		"; Input data count: " << inputDataCount << "; Drop probability: " << dropProbability << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestDropoutLayer::TestDoForwardProp()
{
	bool testPassed =

	// lastBatch == true

	// dropProbability == 0.2f
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.2f /*dropProbability*/) &&

	// dropProbability == 0.5f
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.5f /*dropProbability*/) &&

	// dropProbability == 0.7f
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.7f /*dropProbability*/) &&


	// lastBatch == false

	// dropProbability == 0.2f
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&

	// dropProbability == 0.5f
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&

	// dropProbability == 0.7f
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);

	return testPassed;
}

bool TestDropoutLayer::TestDoBackwardProp()
{
	bool testPassed =

	// lastBatch == true

	// dropProbability == 0.2f
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.2f /*dropProbability*/) &&

	// dropProbability == 0.5f
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.5f /*dropProbability*/) &&

	// dropProbability == 0.7f
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.7f /*dropProbability*/) &&


	// lastBatch == false

	// dropProbability == 0.2f
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/) &&

	// dropProbability == 0.5f
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/) &&

	// dropProbability == 0.7f
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);

	return testPassed;
}