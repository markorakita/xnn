// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for standard layer.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/teststandardlayer.cuh"

#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

#include "mock/include/mockstandardlayer.cuh"
#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mockoutputlayer.cuh"
#include "../../include/testingutils.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/standardlayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

using namespace std::chrono;

TestStandardLayer::TestStandardLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestStandardLayer::TestDoForwardProp, this);
	m_tests["forwardpropspeed"] = bind(&TestStandardLayer::TestForwardPropSpeed, this);
	m_tests["dobackwardprop"] = bind(&TestStandardLayer::TestDoBackwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestStandardLayer::TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	float weightsDeviation = 0.01f;
	float biasesInitialValue = 1.0f;
	ActivationType activationType = ActivationType::Linear;
	float activationAlpha = 0.f;
	float weightsUpdateMomentum = 0.9f;
	float weightsUpdateDecay = 0.0005f;
	float weightsUpdateLearningRateProgressStep = 0.25f;
	float weightsUpdateStartingLearningRate = 0.01f;
	float weightsUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	StandardLayer standardLayer(ParallelismMode::Data, 0, 0, neuralNet.GetCublasHandles()[0], neuralNet.GetCurandStatesBuffers()[0], 0, 1, inputNumChannels,
		inputDataWidth, inputDataHeight, inputDataCount, false, numNeurons, weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep,
		weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha, false);
	standardLayer.AllocateBuffers(allocateTrainBuffers);
	standardLayer.InitializeWeightsFromNormalDistribution(0.f, weightsDeviation);
	standardLayer.InitializeBiasesToConstant(biasesInitialValue);
	standardLayer.AddPrevLayer(&mockInputLayer);
	MockStandardLayer mockStandardLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numNeurons, weightsUpdateMomentum, weightsUpdateDecay,
		weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay,
		biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha);
	mockStandardLayer.AllocateBuffers(allocateTrainBuffers);
	mockStandardLayer.CopyWeightsFromDevice(standardLayer.GetWeightsBuffer());
	mockStandardLayer.CopyBiasesFromDevice(standardLayer.GetBiasesBuffer());
	mockStandardLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockStandardLayer.LoadInputs();
	standardLayer.LoadInputs();
	standardLayer.DoForwardProp(propagationMode);
	mockStandardLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockStandardLayer.GetActivationBufferSize();
	float* standardLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerActivationBuffer, standardLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockStandardLayerActivationBuffer = mockStandardLayer.GetActivationDataBuffer();
	const float maxDiff = 0.01f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.0005f;
	CompareBuffers(standardLayerActivationBuffer, mockStandardLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerActivationBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock standard layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All standard layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
			"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
			"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Number of neurons: " << numNeurons << endl;
	return true;
}

bool TestStandardLayer::TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = true;

	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	float weightsDeviation = 0.01f;
	float biasesInitialValue = 1.0f;
	ActivationType activationType = ActivationType::Linear;
	float activationAlpha = 0.f;
	float weightsUpdateMomentum = 0.9f;
	float weightsUpdateDecay = 0.0005f;
	float weightsUpdateLearningRateProgressStep = 0.25f;
	float weightsUpdateStartingLearningRate = 0.01f;
	float weightsUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	StandardLayer standardLayer(ParallelismMode::Data, 0, 0, neuralNet.GetCublasHandles()[0], neuralNet.GetCurandStatesBuffers()[0], 0, 1, inputNumChannels,
		inputDataWidth, inputDataHeight, inputDataCount, false, numNeurons, weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep,
		weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha, false);
	standardLayer.AllocateBuffers(allocateTrainBuffers);
	standardLayer.InitializeWeightsFromNormalDistribution(0.f, weightsDeviation);
	standardLayer.InitializeBiasesToConstant(biasesInitialValue);
	standardLayer.AddPrevLayer(&mockInputLayer);
	MockStandardLayer mockStandardLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numNeurons, weightsUpdateMomentum, weightsUpdateDecay,
		weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay,
		biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha);
	mockStandardLayer.AllocateBuffers(allocateTrainBuffers);
	mockStandardLayer.CopyWeightsFromDevice(standardLayer.GetWeightsBuffer());
	mockStandardLayer.CopyBiasesFromDevice(standardLayer.GetBiasesBuffer());
	mockStandardLayer.AddPrevLayer(&mockInputLayer);
	// TODO: use constructor with mean and std dev, when you experimentally decide values for those
	MockOutputLayer mockOutputLayer(mockStandardLayer.GetActivationDataSize(), inputDataCount, LossFunctionType::CrossEntropy, false, 0,
		neuralNet.GetCurandStatesBuffers()[0]);
	mockOutputLayer.AllocateBuffers(allocateTrainBuffers);
	mockStandardLayer.AddNextLayer(&mockOutputLayer);
	standardLayer.AddNextLayer(&mockOutputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockStandardLayer.LoadInputs();
	standardLayer.LoadInputs();
	standardLayer.DoForwardProp(propagationMode);
	mockStandardLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	standardLayer.LoadActivationGradients();
	standardLayer.DoBackwardProp();
	mockStandardLayer.LoadActivationGradients();
	mockStandardLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring input gradients results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* standardLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerInputGradientsBuffer, standardLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking input gradients correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockStandardLayerInputGradientsBuffer = mockStandardLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.01f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.0005f;
	CompareBuffers(standardLayerInputGradientsBuffer, mockStandardLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerInputGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock standard layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All standard layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (input gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of neurons: " + to_string(numNeurons));
		return false;
	}

	// Transferring weights gradients results to host.
	size_t weightsGradientsBufferSize = mockStandardLayer.GetWeightsBufferSize();
	float* standardLayerWeightsGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerWeightsGradientsBuffer, weightsGradientsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerWeightsGradientsBuffer, standardLayer.GetWeightsGradientsBuffer(), weightsGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking weights gradients correctness.
	size_t weightsGradientsBufferLength = weightsGradientsBufferSize / sizeof(float);
	const float* mockStandardLayerWeightsGradientsBuffer = mockStandardLayer.GetWeightsGradientsBuffer();
	const float maxDiffWG = 0.01f;
	const float maxDiffPercentageWG = 0.1f;
	const float maxDiffPercentageThresholdWG = 0.005f;
	CompareBuffers(standardLayerWeightsGradientsBuffer, mockStandardLayerWeightsGradientsBuffer, weightsGradientsBufferLength, maxDiffWG, maxDiffPercentageWG,
		maxDiffPercentageThresholdWG, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerWeightsGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock standard layer weights gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All standard layer weights gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (weights gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock weights gradient: " + to_string(firstDifferentMock) + "; First different regular weights gradient: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of neurons: " + to_string(numNeurons));
		return false;
	}

	// Transferring biases gradients results to host.
	size_t biasesGradientsBufferSize = mockStandardLayer.GetBiasesBufferSize();
	float* standardLayerBiasesGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerBiasesGradientsBuffer, biasesGradientsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerBiasesGradientsBuffer, standardLayer.GetBiasesGradientsBuffer(), biasesGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking biases gradients correctness.
	size_t biasesGradientsBufferLength = biasesGradientsBufferSize / sizeof(float);
	const float* mockStandardLayerBiasesGradientsBuffer = mockStandardLayer.GetBiasesGradientsBuffer();
	const float maxDiffBG = 0.01f;
	const float maxDiffPercentageBG = 0.1f;
	const float maxDiffPercentageThresholdBG = 0.005f;
	CompareBuffers(standardLayerBiasesGradientsBuffer, mockStandardLayerBiasesGradientsBuffer, biasesGradientsBufferLength, maxDiffBG, maxDiffPercentageBG,
		maxDiffPercentageThresholdBG, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerBiasesGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock standard layer biases gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All standard layer biases gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (biases gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock biases gradient: " + to_string(firstDifferentMock) + "; First different regular biases gradient: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of neurons: " + to_string(numNeurons));
		return false;
	}

	// Updating parameters.
	float progress = 0.6f;
	standardLayer.UpdateLayerParameters(progress);
	mockStandardLayer.UpdateLayerParameters(progress);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring weights to host.
	size_t weightsBufferSize = mockStandardLayer.GetWeightsBufferSize();
	float* standardLayerWeightsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerWeightsBuffer, weightsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerWeightsBuffer, standardLayer.GetWeightsBuffer(), weightsBufferSize, cudaMemcpyDeviceToHost));

	// Checking weights correctness.
	size_t weightsBufferLength = weightsBufferSize / sizeof(float);
	const float* mockStandardLayerWeightsBuffer = mockStandardLayer.GetWeightsBuffer();
	const float maxDiffW = 0.01f;
	const float maxDiffPercentageW = 0.1f;
	const float maxDiffPercentageThresholdW = 0.005f;
	CompareBuffers(standardLayerWeightsBuffer, mockStandardLayerWeightsBuffer, weightsBufferLength, maxDiffW, maxDiffPercentageW, maxDiffPercentageThresholdW,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerWeightsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock standard layer weights are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All standard layer weights are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (updated weights)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock weights: " + to_string(firstDifferentMock) + "; First different regular weights: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of neurons: " + to_string(numNeurons));
		return false;
	}

	// Transferring biases to host.
	size_t biasesBufferSize = mockStandardLayer.GetBiasesBufferSize();
	float* standardLayerBiasesBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerBiasesBuffer, biasesBufferSize));
	CudaAssert(cudaMemcpy(standardLayerBiasesBuffer, standardLayer.GetBiasesBuffer(), biasesBufferSize, cudaMemcpyDeviceToHost));

	// Checking biases correctness.
	size_t biasesBufferLength = biasesBufferSize / sizeof(float);
	const float* mockStandardLayerBiasesBuffer = mockStandardLayer.GetBiasesBuffer();
	const float maxDiffB = 0.01f;
	const float maxDiffPercentageB = 0.1f;
	const float maxDiffPercentageThresholdB = 0.005f;
	CompareBuffers(standardLayerBiasesBuffer, mockStandardLayerBiasesBuffer, biasesBufferLength, maxDiffB, maxDiffPercentageB, maxDiffPercentageThresholdB,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerBiasesBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock standard layer biases are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All standard layer biases are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (updated biases)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock biases: " + to_string(firstDifferentMock) + "; First different regular biases: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of neurons: " + to_string(numNeurons));
		return false;
	}

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Number of neurons: " << numNeurons << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestStandardLayer::TestDoForwardProp()
{
	bool testPassed =

	// lastBatch == true
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 1000 /*numNeurons*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 2048 /*numNeurons*/) &&

	// lastBatch == false
	TestSingleForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 1000 /*numNeurons*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/);

	return testPassed;
}

/*
	Current speed record over 1000 launches: 7.585ms
*/
bool TestStandardLayer::TestForwardPropSpeed()
{
	// Creating layers.
	uint inputNumChannels = 256;
	uint inputDataWidth = 13;
	uint inputDataHeight = 13;
	uint inputDataCount = 128;
	uint numNeurons = 2048;
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	float weightsDeviation = 0.01f;
	float biasesInitialValue = 1.0f;
	ActivationType activationType = ActivationType::Linear;
	float weightsUpdateMomentum = 0.9f;
	float weightsUpdateDecay = 0.0005f;
	float weightsUpdateLearningRateProgressStep = 0.25f;
	float weightsUpdateStartingLearningRate = 0.01f;
	float weightsUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	StandardLayer standardLayer(ParallelismMode::Data, 0, 0, neuralNet.GetCublasHandles()[0], neuralNet.GetCurandStatesBuffers()[0], 0, 1, inputNumChannels,
		inputDataWidth, inputDataHeight, inputDataCount, false, numNeurons, weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep,
		weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, 0.f, false);
	standardLayer.AllocateBuffers(allocateTrainBuffers);
	standardLayer.InitializeWeightsFromNormalDistribution(0.f, weightsDeviation);
	standardLayer.InitializeBiasesToConstant(biasesInitialValue);
	standardLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);

	// Doing forward prop and measuring time.
	PropagationMode propagationMode = PropagationMode::Train;
	standardLayer.LoadInputs();
	const uint c_timesToLaunch = 1000;
	high_resolution_clock::time_point startTime = high_resolution_clock::now();
	for (uint i = 0; i < c_timesToLaunch; ++i)
	{
		standardLayer.DoForwardProp(propagationMode);
	}	
	CudaAssert(cudaDeviceSynchronize());
	high_resolution_clock::time_point endTime = high_resolution_clock::now();

	// Reporting time.
	long long durationInMilliseconds = duration_cast<milliseconds>(endTime - startTime).count();
	cout << "Forward prop took " << (float)durationInMilliseconds / (float)c_timesToLaunch << "ms in average to process." << endl;

	return true;
}

bool TestStandardLayer::TestDoBackwardProp()
{
	bool testPassed =

	// lastBatch == true
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 1000 /*numNeurons*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 2048 /*numNeurons*/) &&

	// lastBatch == false
	TestSingleBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 1000 /*numNeurons*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/) &&
	TestSingleBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 2048 /*numNeurons*/);

	return testPassed;
}