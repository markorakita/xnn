// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network output layer, used in tests.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <vector>

#include "../../../../../neuralnetwork/layers/include/outputlayer.cuh"

using namespace std;

typedef struct curandStateXORWOW curandState;

class MockOutputLayer : public OutputLayer
{
private:
	// Buffer for cuRAND states.
	curandState* m_curandStatesBuffer;

	// Should we generate random input gradients, for testing other layers backpropagation.
	bool m_generateRandomInputGradients;

	// Mean for randomly generated input gradients.
	float m_inputGradientsMean;

	// Standard deviation for randomly generated input gradients.
	float m_inputGradientsStDev;

	// Calculates accuracy scores for logistic regression loss function.
	void CalculateLogisticRegressionLossesAndScores();

	// Forward prop for logistic regression loss function.
	void LogisticRegressionForwardProp();

	// Calculates accuracy scores for cross entropy loss function.
	void CalculateCrossEntropyScores();

	// Forward prop for cross entropy loss function.
	void CrossEntropyForwardProp();

public:
	// Constructor.
	MockOutputLayer(uint inputDataSize, uint inputDataCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses,
		curandState* curandStatesBuffer);

	// Constructor.
	MockOutputLayer(uint inputDataSize, uint inputDataCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses,
		curandState* curandStatesBuffer, float gradientsMean, float gradientsStDev);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockOutputLayer();

	// Loads labels for input data samples.
	void LoadDataLabels(vector<uint> dataLabels);

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};