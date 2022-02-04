// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for response normalization layer.
// Created: 02/11/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testresponsenormalizationlayer.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mockoutputlayer.cuh"
#include "mock/include/mockresponsenormalizationlayer.cuh"
#include "../../include/testingutils.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/responsenormalizationlayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

TestResponseNormalizationLayer::TestResponseNormalizationLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestResponseNormalizationLayer::TestDoForwardProp, this);
	m_tests["dobackwardprop"] = bind(&TestResponseNormalizationLayer::TestDoBackwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestResponseNormalizationLayer::TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth,
	float bias, float alphaCoeff, float betaCoeff)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockResponseNormalizationLayer mockReNormLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, depth, bias, alphaCoeff, betaCoeff);
	mockReNormLayer.AllocateBuffers(allocateTrainBuffers);
	mockReNormLayer.AddPrevLayer(&mockInputLayer);
	ResponseNormalizationLayer reNormLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		depth, bias, alphaCoeff, betaCoeff, false);
	reNormLayer.AllocateBuffers(allocateTrainBuffers);
	reNormLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockReNormLayer.LoadInputs();
	reNormLayer.LoadInputs();
	reNormLayer.DoForwardProp(propagationMode);
	mockReNormLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockReNormLayer.GetActivationBufferSize();
	float* reNormLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&reNormLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(reNormLayerActivationBuffer, reNormLayer.GetActivationDataBuffer(),
		activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockReNormLayerActivationBuffer = mockReNormLayer.GetActivationDataBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.00005f;
	CompareBuffers(reNormLayerActivationBuffer, mockReNormLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(reNormLayerActivationBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock response normalization activations are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
			"; Beta coeff: " + to_string(betaCoeff));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All response normalization activations are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
			"; Beta coeff: " + to_string(betaCoeff));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) +
			"; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) + "; Beta coeff: " + to_string(betaCoeff));
		return false;
	}

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Depth: " << depth <<
		"; Bias: " << bias << "; Alpha coeff: " << alphaCoeff << "; Beta coeff: " << betaCoeff << endl;
	return true;
}

bool TestResponseNormalizationLayer::TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth,
	float bias, float alphaCoeff, float betaCoeff)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = true;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockResponseNormalizationLayer mockReNormLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, depth, bias, alphaCoeff, betaCoeff);
	mockReNormLayer.AllocateBuffers(allocateTrainBuffers);
	mockReNormLayer.AddPrevLayer(&mockInputLayer);
	ResponseNormalizationLayer reNormLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		depth, bias, alphaCoeff, betaCoeff, false);
	reNormLayer.AllocateBuffers(allocateTrainBuffers);
	reNormLayer.AddPrevLayer(&mockInputLayer);
	// TODO: use constructor with mean and std dev, when you experimentally decide values for those
	MockOutputLayer mockOutputLayer(inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, LossFunctionType::CrossEntropy, false, 0,
		neuralNet.GetCurandStatesBuffers()[0]);
	mockOutputLayer.AllocateBuffers(allocateTrainBuffers);
	mockReNormLayer.AddNextLayer(&mockOutputLayer);
	reNormLayer.AddNextLayer(&mockOutputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockReNormLayer.LoadInputs();
	reNormLayer.LoadInputs();
	reNormLayer.DoForwardProp(propagationMode);
	mockReNormLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	reNormLayer.LoadActivationGradients();
	reNormLayer.DoBackwardProp();
	mockReNormLayer.LoadActivationGradients();
	mockReNormLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* reNormLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&reNormLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(reNormLayerInputGradientsBuffer, reNormLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockReNormLayerInputGradientsBuffer = mockReNormLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.00005f;
	CompareBuffers(reNormLayerInputGradientsBuffer, mockReNormLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(reNormLayerInputGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock response normalization input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
			"; Beta coeff: " + to_string(betaCoeff));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All response normalization input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
			"; Beta coeff: " + to_string(betaCoeff));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) +
			"; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) + "; Beta coeff: " + to_string(betaCoeff));
		return false;
	}

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Depth: " << depth <<
		"; Bias: " << bias << "; Alpha coeff: " << alphaCoeff << "; Beta coeff: " << betaCoeff << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestResponseNormalizationLayer::TestDoForwardProp()
{
	bool testPassed =

	// lastBatch == true
	
	TestSingleForwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 127 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 119 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 97 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 74 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 16 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&

	// lastBatch == false

	TestSingleForwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleForwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&

	// Various formula parameters

	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 6 /*depth*/, 1.2f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*depth*/, 6.7f /*bias*/,
		0.03f /*alphaCoeff*/, 0.2f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 2 /*depth*/, 0.5f /*bias*/,
		0.001f /*alphaCoeff*/, 3.0f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 10 /*depth*/, 3.1f /*bias*/,
		0.009f /*alphaCoeff*/, 1.0f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*depth*/, 4.9f /*bias*/,
		1.0f /*alphaCoeff*/, 1.0f /*betaCoeff*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 8 /*depth*/, 2.8f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/);

	return testPassed;
}

bool TestResponseNormalizationLayer::TestDoBackwardProp()
{
	bool testPassed =

	// lastBatch == true

	TestSingleBackwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 127 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 119 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 97 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 74 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 16 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&

	// lastBatch == false

	TestSingleBackwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&
	TestSingleBackwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/) &&

	// Various formula parameters

	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 6 /*depth*/, 1.2f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*depth*/, 6.7f /*bias*/,
		0.03f /*alphaCoeff*/, 0.2f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 2 /*depth*/, 0.5f /*bias*/,
		0.001f /*alphaCoeff*/, 3.0f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 10 /*depth*/, 3.1f /*bias*/,
		0.009f /*alphaCoeff*/, 1.0f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*depth*/, 4.9f /*bias*/,
		1.0f /*alphaCoeff*/, 1.0f /*betaCoeff*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 8 /*depth*/, 2.8f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/);

	return testPassed;
}