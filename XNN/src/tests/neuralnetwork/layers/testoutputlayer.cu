// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for output layer.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testoutputlayer.cuh"

#include <chrono>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "include/testsoftmaxlayer.cuh"
#include "mock/include/mockinputlayer.cuh"
#include "mock/include/mockoutputlayer.cuh"
#include "mock/include/mockstandardlayer.cuh"
#include "../../../neuralnetwork/include/neuralnet.cuh"
#include "../../../neuralnetwork/layers/include/outputlayer.cuh"
#include "../../../neuralnetwork/layers/include/softmaxlayer.cuh"
#include "../../../utils/include/asserts.cuh"
#include "../../../utils/include/cudaasserts.cuh"

using namespace std::chrono;

TestOutputLayer::TestOutputLayer()
{
	// Registering tests.
	m_tests["doforwardprop"] = bind(&TestOutputLayer::TestDoForwardProp, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

bool TestOutputLayer::TestSingleCrossEntropyForwardProp(uint inputDataSize, uint inputDataCount)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;

	// Creating layers.
	MockInputLayer mockInputLayer(1, inputDataSize, 1, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	SoftMaxLayer softMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, false);
	softMaxLayer.AllocateBuffers(allocateTrainBuffers);
	softMaxLayer.AddPrevLayer(&mockInputLayer);
	MockOutputLayer mockOutputLayer(inputDataSize, inputDataCount, LossFunctionType::CrossEntropy, true, 5, NULL);
	mockOutputLayer.AllocateBuffers(allocateTrainBuffers);
	mockOutputLayer.AddPrevLayer(&softMaxLayer);
	OutputLayer outputLayer(0, 0, inputDataSize, inputDataCount, inputDataCount, LossFunctionType::CrossEntropy, true, 5, 0);
	outputLayer.AllocateBuffers(allocateTrainBuffers);
	outputLayer.AddPrevLayer(&softMaxLayer);
	softMaxLayer.AddNextLayer(&outputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromNormalDistribution(TestSoftMaxLayer::c_inputActivationsMean, TestSoftMaxLayer::c_inputActivationsStDev);

	// Creating pseudo random labels.
	vector<uint> labels;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		labels.push_back((57 * i * i) % inputDataSize);
	}

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	softMaxLayer.LoadInputs();
	softMaxLayer.DoForwardProp(propagationMode);
	outputLayer.LoadDataLabels(labels);
	mockOutputLayer.LoadDataLabels(labels);
	outputLayer.LoadInputs();
	mockOutputLayer.LoadInputs();
	outputLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaStreamSynchronize(0));

	// Transferring results to host.
	float* outputLayerScores = outputLayer.GetHostScores();
	float* mockOutputLayerScores = mockOutputLayer.GetScores();
	float* outputLayerMultipleGuessScores = outputLayer.GetHostMultipleGuessScores();
	float* mockOutputLayerMultipleGuessScores = mockOutputLayer.GetMultipleGuessScores();

	// Checking correctness.
	bool foundDifferentScores = false;
	bool foundDifferentMultipleGuessScores = false;
	float firstDifference = 0.f;
	const float c_allowedDiff = 0.000001f;
	const float c_allowedDiffCoeff = 0.0001f;
	const float c_allowedDiffCoeffThreshold = 0.0000001f;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		float diffScores = fabs(outputLayerScores[i] - mockOutputLayerScores[i]);
		if (diffScores > c_allowedDiff || (diffScores > c_allowedDiffCoeffThreshold &&
			diffScores > c_allowedDiffCoeff * max(outputLayerScores[i], mockOutputLayerScores[i])))
		{
			foundDifferentScores = true;
			firstDifference = outputLayerScores[i] - mockOutputLayerScores[i];
			break;
		}
		float diffMultipleGuessScores = fabs(outputLayerMultipleGuessScores[i] - mockOutputLayerMultipleGuessScores[i]);
		if (diffMultipleGuessScores > c_allowedDiff || (diffMultipleGuessScores > c_allowedDiffCoeffThreshold &&
			diffMultipleGuessScores > c_allowedDiffCoeff * max(outputLayerMultipleGuessScores[i], mockOutputLayerMultipleGuessScores[i])))
		{
			foundDifferentMultipleGuessScores = true;
			firstDifference = outputLayerMultipleGuessScores[i] - mockOutputLayerMultipleGuessScores[i];
			break;
		}
	}

	if (foundDifferentScores)
	{
		EmitWarning("Found different scores! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount) + "; First difference: " + to_string(firstDifference));
		return false;
	}
	else if (foundDifferentMultipleGuessScores)
	{
		EmitWarning("Found different multiple guess scores! Input data size: " + to_string(inputDataSize) +
			"; Input data count: " + to_string(inputDataCount) + "; First difference: " + to_string(firstDifference));
		return false;
	}

	float mockLoss = mockOutputLayer.GetLoss() / inputDataCount;
	float regLoss = outputLayer.GetLoss() / inputDataCount;
	float diffLoss = fabs(mockLoss - regLoss);
	if (diffLoss > c_allowedDiff || diffLoss > c_allowedDiffCoeff * max(mockLoss, regLoss))
	{
		EmitWarning("Calculated different losses! Input data size: " +
			to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount) + "; Mock loss: " + to_string(mockLoss) + "; Regular loss: " +
			to_string(regLoss));
		return false;
	}

	float mockAccuracy = mockOutputLayer.GetAccuracy() / inputDataCount;
	float regAccuracy = outputLayer.GetAccuracy() / inputDataCount;
	float diffAccuracy = fabs(mockAccuracy - regAccuracy);
	if (diffAccuracy > c_allowedDiff || diffAccuracy > c_allowedDiffCoeff * max(mockAccuracy, regAccuracy))
	{
		EmitWarning("Calculated different accuracies! Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount) +
			"; Mock accuracy: " + to_string(mockAccuracy) + "; Regular accuracy: " + to_string(regAccuracy));
		return false;
	}

	float mockMultipleGuessAccuracy = mockOutputLayer.GetMultipleGuessAccuracy() / inputDataCount;
	float regMultipleGuessAccuracy = outputLayer.GetMultipleGuessAccuracy() / inputDataCount;
	float diffMultipleGuessAccuracy = fabs(mockMultipleGuessAccuracy - regMultipleGuessAccuracy);
	if (diffMultipleGuessAccuracy > c_allowedDiff || diffMultipleGuessAccuracy > c_allowedDiffCoeff * max(mockMultipleGuessAccuracy, regMultipleGuessAccuracy))
	{
		EmitWarning("Calculated different multiple guess accuracies! Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount) +
			"; Mock multiple guess accuracy: " + to_string(mockMultipleGuessAccuracy) + "; Regular multiple guess accuracy: " + to_string(regMultipleGuessAccuracy));
		return false;
	}

	cout << "Forward prop for CrossEntropy loss passed. Input data size: " << inputDataSize << "; Input data count: " << inputDataCount << endl;
	return true;
}

bool TestOutputLayer::TestSingleLogisticRegressionForwardProp(uint inputDataCount)
{
	NeuralNet neuralNet(1);
	const bool allocateTrainBuffers = false;

	// Creating layers.
	MockInputLayer mockInputLayer(1, 1, 1, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
	mockInputLayer.AllocateBuffers(allocateTrainBuffers);
	MockOutputLayer mockOutputLayer(1, inputDataCount, LossFunctionType::LogisticRegression, true, 5, NULL);
	mockOutputLayer.AllocateBuffers(allocateTrainBuffers);
	mockOutputLayer.AddPrevLayer(&mockInputLayer);
	OutputLayer outputLayer(0, 0, 1, inputDataCount, inputDataCount, LossFunctionType::LogisticRegression, true, 5, 0);
	outputLayer.AllocateBuffers(allocateTrainBuffers);
	outputLayer.AddPrevLayer(&mockInputLayer);

	// Generating inputs.
	mockInputLayer.GenerateActivationFromUniformDistribution(0.f, 1.0f);

	// Creating pseudo random labels.
	vector<uint> labels;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		labels.push_back(i % 2);
	}

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	outputLayer.LoadDataLabels(labels);
	mockOutputLayer.LoadDataLabels(labels);
	outputLayer.LoadInputs();
	mockOutputLayer.LoadInputs();
	outputLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaStreamSynchronize(0));

	// Transferring results to host.
	float* outputLayerScores = outputLayer.GetHostScores();
	float* mockOutputLayerScores = mockOutputLayer.GetScores();

	// Checking correctness.
	bool foundDifferentScores = false;
	float firstDifference = 0.f;
	const float c_allowedDiff = 0.000001f;
	const float c_allowedDiffCoeff = 0.0001f;
	const float c_allowedDiffCoeffThreshold = 0.0000001f;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		float diffScores = fabs(outputLayerScores[i] - mockOutputLayerScores[i]);
		if (diffScores > c_allowedDiff || (diffScores > c_allowedDiffCoeffThreshold &&
			diffScores > c_allowedDiffCoeff * max(outputLayerScores[i], mockOutputLayerScores[i])))
		{
			foundDifferentScores = true;
			firstDifference = outputLayerScores[i] - mockOutputLayerScores[i];
			break;
		}
	}

	if (foundDifferentScores)
	{
		EmitWarning("Found different scores! Input data count: " + to_string(inputDataCount) + "; First difference: " + to_string(firstDifference));
		return false;
	}

	float mockLoss = mockOutputLayer.GetLoss() / inputDataCount;
	float regLoss = outputLayer.GetLoss() / inputDataCount;
	float diffLoss = fabs(mockLoss - regLoss);
	if (diffLoss > c_allowedDiff || diffLoss > c_allowedDiffCoeff * max(mockLoss, regLoss))
	{
		EmitWarning("Calculated different losses! Input data count: " + to_string(inputDataCount) + "; Mock loss: " + to_string(mockLoss) +
			"; Regular loss: " + to_string(regLoss));
		return false;
	}

	float mockAccuracy = mockOutputLayer.GetAccuracy() / inputDataCount;
	float regAccuracy = outputLayer.GetAccuracy() / inputDataCount;
	float diffAccuracy = fabs(mockAccuracy - regAccuracy);
	if (diffAccuracy > c_allowedDiff || diffAccuracy > c_allowedDiffCoeff * max(mockAccuracy, regAccuracy))
	{
		EmitWarning("Calculated different accuracies! Input data count: " + to_string(inputDataCount) +
			"; Mock accuracy: " + to_string(mockAccuracy) + "; Regular accuracy: " + to_string(regAccuracy));
		return false;
	}

	cout << "Forward prop for LogisticRegression loss passed. Input data count: " << inputDataCount << endl;
	return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestOutputLayer::TestDoForwardProp()
{
	vector<uint> inputDataSizes{ 2, 10, 100, 1000 };
	vector<uint> inputDataCounts{ 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025 };

	for (size_t i = 0; i < inputDataSizes.size(); ++i)
	{
		for (size_t j = 0; j < inputDataCounts.size(); ++j)
		{
			if (!TestSingleCrossEntropyForwardProp(inputDataSizes[i], inputDataCounts[j]))
			{
				return false;
			}
		}
	}

	for (size_t i = 0; i < inputDataCounts.size(); ++i)
	{
		if (!TestSingleLogisticRegressionForwardProp(inputDataCounts[i]))
		{
			return false;
		}
	}

	return true;
}