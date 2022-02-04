// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for convolutional layer.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testconvolutionallayer.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "mock/include/mockconvolutionallayer.cuh"
#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mockoutputlayer.cuh"
#include "../../include/testingutils.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/convolutionallayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

TestConvolutionalLayer::TestConvolutionalLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestConvolutionalLayer::TestDoForwardProp, this);
	m_tests["dobackwardprop"] = bind(&TestConvolutionalLayer::TestDoBackwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestConvolutionalLayer::TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
	uint filterHeight, int paddingX, int paddingY, uint stride)
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
	float filtersUpdateMomentum = 0.9f;
	float filtersUpdateDecay = 0.0005f;
	float filtersUpdateLearningRateProgressStep = 0.25f;
	float filtersUpdateStartingLearningRate = 0.01f;
	float filtersUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	ConvolutionalLayer convolutionalLayer(ParallelismMode::Data, 0, 0, neuralNet.GetCurandStatesBuffers()[0], 0, 1, inputNumChannels, inputDataWidth, inputDataHeight,
		inputDataCount, false, numFilters, filterWidth, filterHeight, inputNumChannels, filtersUpdateMomentum, filtersUpdateDecay, filtersUpdateLearningRateProgressStep,
		filtersUpdateStartingLearningRate, filtersUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, paddingX, paddingY, stride, activationType, activationAlpha, false);
	convolutionalLayer.AllocateBuffers(allocateTrainBuffers);
	convolutionalLayer.InitializeWeightsFromNormalDistribution(0.f, weightsDeviation);
	convolutionalLayer.InitializeBiasesToConstant(biasesInitialValue);
	convolutionalLayer.AddPrevLayer(&mockInputLayer);
	MockConvolutionalLayer mockConvolutionalLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numFilters, filterWidth, filterHeight,
		inputNumChannels, filtersUpdateMomentum, filtersUpdateDecay, filtersUpdateLearningRateProgressStep, filtersUpdateStartingLearningRate,
		filtersUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate,
		biasesUpdateLearningRateUpdateFactor, paddingX, paddingY, stride, activationType, activationAlpha);
	mockConvolutionalLayer.AllocateBuffers(allocateTrainBuffers);
	mockConvolutionalLayer.CopyWeightsFromDevice(convolutionalLayer.GetWeightsBuffer());
	mockConvolutionalLayer.CopyBiasesFromDevice(convolutionalLayer.GetBiasesBuffer());
	mockConvolutionalLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	if (inputNumChannels > 3)
	{
		mockInputLayer.GenerateActivationFromNormalDistribution(0.f, 0.1f);
	}
	else
	{
		mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);
	}

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockConvolutionalLayer.LoadInputs();
	convolutionalLayer.LoadInputs();
	convolutionalLayer.DoForwardProp(propagationMode);
	mockConvolutionalLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());
	
	// Transferring results to host.
	size_t activationsBufferSize = mockConvolutionalLayer.GetActivationBufferSize();
	float* convolutionalLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&convolutionalLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(convolutionalLayerActivationBuffer, convolutionalLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));
	
	// Checking correctness.
	bool correctResult = true;
	float firstDifference = 0.0f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockConvolutionalLayerActivationBuffer = mockConvolutionalLayer.GetActivationDataBuffer();
	for (size_t i = 0; i < activationsBufferLength; ++i)
	{
		float diff = fabs(mockConvolutionalLayerActivationBuffer[i] - convolutionalLayerActivationBuffer[i]);
		if (correctResult && (diff > 0.001f || (diff > 0.0001f && diff > 0.5f * max(abs(mockConvolutionalLayerActivationBuffer[i]), abs(convolutionalLayerActivationBuffer[i])))))
		{
			correctResult = false;
			firstDifference = mockConvolutionalLayerActivationBuffer[i] - convolutionalLayerActivationBuffer[i];
			PrintComputationInfo(i, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numFilters, filterWidth, filterHeight, paddingX, paddingY,
				stride, mockConvolutionalLayer.GetInputDataBuffer(), mockConvolutionalLayer.GetWeightsBuffer(), mockConvolutionalLayerActivationBuffer[i],
				convolutionalLayerActivationBuffer[i]);
		}
		if (mockConvolutionalLayerActivationBuffer[i] != 0.0f)
		{
			foundDifferentFromZeroMock = true;
		}
		if (convolutionalLayerActivationBuffer[i] != 0.0f)
		{
			foundDifferentFromZeroReg = true;
		}
	}

	CudaAssert(cudaFreeHost(convolutionalLayerActivationBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock convolutional activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All convolutional activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
			to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect forward prop! First difference: " + to_string(firstDifference) + "; Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Number of filters: " << numFilters << endl;
	return true;
}

void TestConvolutionalLayer::PrintComputationInfo(size_t activationDifferentPixelIndex, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount,
	uint numFilters, uint filterWidth, uint filterHeight, int paddingX, int paddingY, uint stride, float* inputDataBuffer, float* filtersBuffer,
	float differentActivationPixelMock, float differentActivationPixelRegular)
{
	size_t dataIndex = activationDifferentPixelIndex % inputDataCount;
	uint numPatchesX = 1 + (uint)ceil((2.0 * paddingX + inputDataWidth - filterWidth) / stride);
	uint numPatchesY = 1 + (uint)ceil((2.0 * paddingY + inputDataHeight - filterHeight) / stride);
	size_t filterIndex = activationDifferentPixelIndex / ((size_t)inputDataCount * numPatchesX * numPatchesY);
	size_t patchIndex = (activationDifferentPixelIndex - filterIndex * inputDataCount * numPatchesX * numPatchesY) / inputDataCount;
	size_t patchIndexY = patchIndex / numPatchesX;
	size_t patchIndexX = patchIndex % numPatchesX;

	cout << "Data pixels in patch that causes bad computation:" << endl << endl;
	int dataStartX = -paddingX + (int)(patchIndexX * stride);
	int dataStartY = -paddingY + (int)(patchIndexY * stride);
	for (size_t channel = 0; channel < inputNumChannels; ++channel)
	{
		cout << "Channel " << channel << ":" << endl;
		for (int i = dataStartY; i < dataStartY + (int)filterHeight; ++i)
		{
			for (int j = dataStartX; j < dataStartX + (int)filterWidth; ++j)
			{
				if (i < 0 || i >= (int)inputDataHeight || j < 0 || j >= (int)inputDataWidth)
				{
					cout << 0 << " ";
				}
				else
				{
					cout << inputDataBuffer[dataIndex + channel * inputDataCount * inputDataWidth * inputDataHeight + (i * (size_t)inputDataWidth + j) * inputDataCount] << " ";
				}
			}
			cout << endl;
		}
	}

	cout << endl << "Filter pixels:" << endl << endl;
	for (size_t channel = 0; channel < inputNumChannels; ++channel)
	{
		cout << "Channel " << channel << ":" << endl;
		for (int i = 0; i < (int)filterHeight; ++i)
		{
			for (int j = 0; j < (int)filterWidth; ++j)
			{
				cout << filtersBuffer[filterIndex + channel * numFilters * filterWidth * filterHeight + (i * (size_t)filterWidth + j) * numFilters] << " ";
			}
			cout << endl;
		}
	}

	cout << endl << "Computated pixel in mock layer: " << differentActivationPixelMock << endl;
	cout << "Computated pixel in regular layer: " << differentActivationPixelRegular << endl;
}

bool TestConvolutionalLayer::TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
	uint filterHeight, int paddingX, int paddingY, uint stride)
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
	float filtersUpdateMomentum = 0.9f;
	float filtersUpdateDecay = 0.0005f;
	float filtersUpdateLearningRateProgressStep = 0.25f;
	float filtersUpdateStartingLearningRate = 0.01f;
	float filtersUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	ConvolutionalLayer convolutionalLayer(ParallelismMode::Data, 0, 0, neuralNet.GetCurandStatesBuffers()[0], 0, 1, inputNumChannels, inputDataWidth, inputDataHeight,
		inputDataCount, false, numFilters, filterWidth, filterHeight, inputNumChannels, filtersUpdateMomentum, filtersUpdateDecay, filtersUpdateLearningRateProgressStep,
		filtersUpdateStartingLearningRate, filtersUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, paddingX, paddingY, stride, activationType, activationAlpha, false);
	convolutionalLayer.AllocateBuffers(allocateTrainBuffers);
	convolutionalLayer.InitializeWeightsFromNormalDistribution(0.f, weightsDeviation);
	convolutionalLayer.InitializeBiasesToConstant(biasesInitialValue);
	convolutionalLayer.AddPrevLayer(&mockInputLayer);
	MockConvolutionalLayer mockConvolutionalLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numFilters, filterWidth, filterHeight,
		inputNumChannels, filtersUpdateMomentum, filtersUpdateDecay, filtersUpdateLearningRateProgressStep, filtersUpdateStartingLearningRate,
		filtersUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate,
		biasesUpdateLearningRateUpdateFactor, paddingX, paddingY, stride, activationType, activationAlpha);
	mockConvolutionalLayer.AllocateBuffers(allocateTrainBuffers);
	mockConvolutionalLayer.CopyWeightsFromDevice(convolutionalLayer.GetWeightsBuffer());
	mockConvolutionalLayer.CopyBiasesFromDevice(convolutionalLayer.GetBiasesBuffer());
	mockConvolutionalLayer.AddPrevLayer(&mockInputLayer);
	// TODO: use constructor with mean and std dev, when you experimentally decide values for those
	MockOutputLayer mockOutputLayer(convolutionalLayer.GetActivationDataSize() * convolutionalLayer.GetActivationNumChannels(), inputDataCount,
		LossFunctionType::CrossEntropy, false, 0, neuralNet.GetCurandStatesBuffers()[0]);
	mockOutputLayer.AllocateBuffers(allocateTrainBuffers);
	mockConvolutionalLayer.AddNextLayer(&mockOutputLayer);
	convolutionalLayer.AddNextLayer(&mockOutputLayer);

	// Generating inputs.
	if (inputNumChannels > 3)
	{
		mockInputLayer.GenerateActivationFromNormalDistribution(0.f, 0.1f);
	}
	else
	{
		mockInputLayer.GenerateActivationFromUniformIntDistribution(-128, 127);
	}

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockConvolutionalLayer.LoadInputs();
	convolutionalLayer.LoadInputs();
	convolutionalLayer.DoForwardProp(propagationMode);
	mockConvolutionalLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	convolutionalLayer.LoadActivationGradients();
	convolutionalLayer.DoBackwardProp();
	mockConvolutionalLayer.LoadActivationGradients();
	mockConvolutionalLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring input gradients results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* convolutionalLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&convolutionalLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(convolutionalLayerInputGradientsBuffer, convolutionalLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking input gradients correctness.
	bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
	size_t numDifferences;
	float firstDifference, firstDifferentMock, firstDifferentReg;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockConvolutionalLayerInputGradientsBuffer = mockConvolutionalLayer.GetInputGradientsBuffer();
	const float maxDiff = inputNumChannels > 3 ? 0.005f : 0.2f;
	const float maxDiffPercentage = 2.0f;
	const float maxDiffPercentageThreshold = 0.001f;
	CompareBuffers(convolutionalLayerInputGradientsBuffer, mockConvolutionalLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage,
		maxDiffPercentageThreshold, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(convolutionalLayerInputGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock convolutional input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All convolutional input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (input gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of filters: " + to_string(numFilters));
		return false;
	}

	// Transferring filters gradients results to host.
	size_t filtersGradientsBufferSize = mockConvolutionalLayer.GetWeightsBufferSize();
	float* convolutionalLayerFiltersGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&convolutionalLayerFiltersGradientsBuffer, filtersGradientsBufferSize));
	CudaAssert(cudaMemcpy(convolutionalLayerFiltersGradientsBuffer, convolutionalLayer.GetWeightsGradientsBuffer(), filtersGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking filters gradients correctness.
	size_t filtersGradientsBufferLength = filtersGradientsBufferSize / sizeof(float);
	const float* mockConvolutionalLayerFiltersGradientsBuffer = mockConvolutionalLayer.GetWeightsGradientsBuffer();
	const float maxDiffFG = inputNumChannels > 3 ? 0.007f : 0.3f;
	const float maxDiffPercentageFG = inputNumChannels > 3 ? 2.0f : 0.5f;
	const float maxDiffPercentageThresholdFG = 0.001f;
	CompareBuffers(convolutionalLayerFiltersGradientsBuffer, mockConvolutionalLayerFiltersGradientsBuffer, filtersGradientsBufferLength, maxDiffFG, maxDiffPercentageFG,
		maxDiffPercentageThresholdFG, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(convolutionalLayerFiltersGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock convolutional filters gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All convolutional filters gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (filters gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock filters gradient: " + to_string(firstDifferentMock) + "; First different regular filters gradient: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of filters: " + to_string(numFilters));
		return false;
	}

	// Transferring biases gradients results to host.
	size_t biasesGradientsBufferSize = mockConvolutionalLayer.GetBiasesBufferSize();
	float* convolutionalLayerBiasesGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&convolutionalLayerBiasesGradientsBuffer, biasesGradientsBufferSize));
	CudaAssert(cudaMemcpy(convolutionalLayerBiasesGradientsBuffer, convolutionalLayer.GetBiasesGradientsBuffer(), biasesGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking biases gradients correctness.
	size_t biasesGradientsBufferLength = biasesGradientsBufferSize / sizeof(float);
	const float* mockConvolutionalLayerBiasesGradientsBuffer = mockConvolutionalLayer.GetBiasesGradientsBuffer();
	const float maxDiffBG = inputNumChannels > 3 ? 0.007f : 0.2f;
	const float maxDiffPercentageBG = inputNumChannels > 3 ? 2.0f : 0.5f;
	const float maxDiffPercentageThresholdBG = 0.001f;
	CompareBuffers(convolutionalLayerBiasesGradientsBuffer, mockConvolutionalLayerBiasesGradientsBuffer, biasesGradientsBufferLength, maxDiffBG, maxDiffPercentageBG,
		maxDiffPercentageThresholdBG, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(convolutionalLayerBiasesGradientsBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock convolutional biases gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All convolutional biases gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (biases gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock biases gradient: " + to_string(firstDifferentMock) + "; First different regular biases gradient: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of filters: " + to_string(numFilters));
		return false;
	}

	// Updating parameters.
	float progress = 0.6f;
	convolutionalLayer.UpdateLayerParameters(progress);
	mockConvolutionalLayer.UpdateLayerParameters(progress);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring filters to host.
	size_t filtersBufferSize = mockConvolutionalLayer.GetWeightsBufferSize();
	float* convolutionalLayerFiltersBuffer;
	CudaAssert(cudaMallocHost<float>(&convolutionalLayerFiltersBuffer, filtersBufferSize));
	CudaAssert(cudaMemcpy(convolutionalLayerFiltersBuffer, convolutionalLayer.GetWeightsBuffer(), filtersBufferSize, cudaMemcpyDeviceToHost));

	// Checking filters correctness.
	size_t filtersBufferLength = filtersBufferSize / sizeof(float);
	const float* mockConvolutionalLayerFiltersBuffer = mockConvolutionalLayer.GetWeightsBuffer();
	const float maxDiffF = 0.01f;
	const float maxDiffPercentageF = 0.1f;
	const float maxDiffPercentageThresholdF = 0.005f;
	CompareBuffers(convolutionalLayerFiltersBuffer, mockConvolutionalLayerFiltersBuffer, filtersBufferLength, maxDiffF, maxDiffPercentageF, maxDiffPercentageThresholdF,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(convolutionalLayerFiltersBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock convolutional filters are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All convolutional filters are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (updated filters)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock filters: " + to_string(firstDifferentMock) + "; First different regular filters: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of filters: " + to_string(numFilters));
		return false;
	}

	// Transferring biases to host.
	size_t biasesBufferSize = mockConvolutionalLayer.GetBiasesBufferSize();
	float* convolutionalLayerBiasesBuffer;
	CudaAssert(cudaMallocHost<float>(&convolutionalLayerBiasesBuffer, biasesBufferSize));
	CudaAssert(cudaMemcpy(convolutionalLayerBiasesBuffer, convolutionalLayer.GetBiasesBuffer(), biasesBufferSize, cudaMemcpyDeviceToHost));

	// Checking biases correctness.
	size_t biasesBufferLength = biasesBufferSize / sizeof(float);
	const float* mockConvolutionalLayerBiasesBuffer = mockConvolutionalLayer.GetBiasesBuffer();
	const float maxDiffB = 0.01f;
	const float maxDiffPercentageB = 0.1f;
	const float maxDiffPercentageThresholdB = 0.005f;
	CompareBuffers(convolutionalLayerBiasesBuffer, mockConvolutionalLayerBiasesBuffer, biasesBufferLength, maxDiffB, maxDiffPercentageB, maxDiffPercentageThresholdB,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(convolutionalLayerBiasesBuffer));

	if (!foundDifferentFromZeroMock)
	{
		EmitWarning("All mock convolutional biases are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!foundDifferentFromZeroReg)
	{
		EmitWarning("All convolutional biases are zeros! Input num channels: " + to_string(inputNumChannels) +
			"; Input data count: " + to_string(inputDataCount) + "; Number of filters: " + to_string(numFilters));
		return false;
	}
	else if (!correctResult)
	{
		EmitWarning("Incorrect backward prop (updated biases)! Num differences: " + to_string(numDifferences) + "; First difference: " +
			to_string(firstDifference) + "; First different mock biases: " + to_string(firstDifferentMock) + "; First different regular biases: " +
			to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
			"; Number of filters: " + to_string(numFilters));
		return false;
	}

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Number of filters: " << numFilters << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestConvolutionalLayer::TestDoForwardProp()
{
	bool testPassed =

	// lastBatch == true

	// inputNumChannels == 3
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 97 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 35 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 1 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 127 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&

	// inputNumChannels % 4 == 0
	TestSingleForwardProp(20 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 384 /*numFilters*/, 3 /*filterWidth*/,
		3 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 111 /*inputDataCount*/, 192 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 86 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 99 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputNumChannels % 8 == 0
	TestSingleForwardProp(32 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 22 /*inputDataCount*/, 256 /*numFilters*/, 3 /*filterWidth*/,
		3 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 79 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 88 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 125 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	
	// lastBatch == false

	// inputNumChannels == 3

	// inputDataCount % 128 == 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&

	// inputDataCount % 64 == 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 4 /*stride*/) &&

	// inputDataCount % 32 == 0
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&

	// inputNumChannels % 4 == 0

	// inputDataCount % 128 == 0
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 64 == 0
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 32 == 0
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputNumChannels % 8 == 0

	// inputDataCount % 128 == 0
	TestSingleForwardProp(32 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 64 == 0
	TestSingleForwardProp(32 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 32 == 0
	TestSingleForwardProp(32 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/);

	return testPassed;
}

bool TestConvolutionalLayer::TestDoBackwardProp()
{
	bool testPassed =

	// lastBatch == true

	// inputNumChannels == 3

	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 97 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 35 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 1 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 127 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&

	// inputNumChannels % 4 == 0
	// TODO: Currently unsupported, uncomment here and below if you support this one day.
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 384 /*numFilters*/, 3 /*filterWidth*/,
	//	3 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 111 /*inputDataCount*/, 192 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 86 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 99 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputNumChannels % 8 == 0
	TestSingleBackwardProp(32 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 22 /*inputDataCount*/, 256 /*numFilters*/, 3 /*filterWidth*/,
		3 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 79 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 88 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 125 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&


	// lastBatch == false

	// inputNumChannels == 3

	// inputDataCount % 128 == 0
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 128 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&

	// inputDataCount % 64 == 0
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 64 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&

	// inputDataCount % 32 == 0
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 64 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 48 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 32 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 4 /*stride*/) &&
	TestSingleBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 32 /*inputDataCount*/, 16 /*numFilters*/, 11 /*filterWidth*/,
		11 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 4 /*stride*/) &&

	// inputNumChannels % 4 == 0

	// inputDataCount % 128 == 0
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 64 == 0
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 32 == 0
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(20 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	//TestSingleBackwardProp(44 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
	//	5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputNumChannels % 8 == 0

	// inputDataCount % 128 == 0
	TestSingleBackwardProp(32 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 64 == 0
	TestSingleBackwardProp(32 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&

	// inputDataCount % 32 == 0
	TestSingleBackwardProp(32 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 128 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 2 /*paddingX*/, 2 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(48 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 64 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 0 /*paddingX*/, 0 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(64 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 32 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/) &&
	TestSingleBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 16 /*numFilters*/, 5 /*filterWidth*/,
		5 /*filterHeight*/, 1 /*paddingX*/, 1 /*paddingY*/, 1 /*stride*/);

	return testPassed;
}