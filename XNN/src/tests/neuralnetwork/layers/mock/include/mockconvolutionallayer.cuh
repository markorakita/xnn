// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network convolutional layer, used in tests.
// Created: 01/27/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "mockweightslayer.cuh"
#include "../../../../../neuralnetwork/include/activationfunctions.cuh"

using namespace std;

class MockConvolutionalLayer : public MockWeightsLayer
{
private:
	// Number of convolutional filters.
	uint m_numFilters;

	// Width of a filter.
	uint m_filterWidth;

	// Height of a filter.
	uint m_filterHeight;

	// Size of a filter.
	uint m_filterSize;

	// Number of channels per filter.
	uint m_numFilterChannels;

	// Padding in dimension X.
	int m_paddingX;

	// Padding in dimension Y.
	int m_paddingY;

	// Stride for patching.
	uint m_stride;

	// Number of patches to apply filters on in dimension X.
	uint m_numPatchesX;

	// Number of patches to apply filters on in dimension Y.
	uint m_numPatchesY;

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

	// Calculates gradients of biases.
	virtual void CalculateBiasesGradients();

	// Calculates gradients of weights.
	virtual void CalculateWeightsGradients();

	// Calculates gradients of inputs.
	void CalculateInputGradients();

	// Calculates gradients of preactivations.
	void CalculatePreactivationsGradients();

public:
	// Constructor.
	MockConvolutionalLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
		uint filterHeight, uint numFilterChannels, float filtersUpdateMomentum, float filtersUpdateDecay, float filtersUpdateLearningRateProgressStep,
		float filtersUpdateStartingLearningRate, float filtersUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay,
		float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor,
		int paddingX, int paddingY, uint stride, ActivationType activationType, float activationAlpha);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockConvolutionalLayer();

	// Gets input data buffer.
	float* GetInputDataBuffer() const { return m_inputDataBuffer; }

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};