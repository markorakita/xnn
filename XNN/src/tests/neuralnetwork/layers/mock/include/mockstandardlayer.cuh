// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network standard layer, used in tests.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "mockweightslayer.cuh"
#include "../../../../../neuralnetwork/include/activationfunctions.cuh"

using namespace std;

class MockStandardLayer : public MockWeightsLayer
{
private:
	// Number of neurons in standard layer.
	uint m_numNeurons;

	// Preactivations buffer.
	float* m_preactivationDataBuffer;

	// Preactivations gradients buffer.
	float* m_preactivationGradientsBuffer;

	// Calculates preactivations.
	void CalculatePreactivations();

	// Adds biases to preactivations.
	void AddBiases();

	// Calculates activations.
	void CalculateActivations();

	// Calculates gradients of weights.
	virtual void CalculateWeightsGradients();

	// Calculates gradients of biases.
	virtual void CalculateBiasesGradients();

	// Calculates gradients of inputs.
	void CalculateInputGradients();

	// Calculates gradients of preactivations.
	void CalculatePreactivationsGradients();

public:
	// Constructor.
	MockStandardLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons, float weightsUpdateMomentum,
		float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor,
		float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,float biasesUpdateStartingLearningRate,
		float biasesUpdateLearningRateUpdateFactor, ActivationType activationType, float activationAlpha);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockStandardLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	virtual void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};