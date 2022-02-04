// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network softmax layer, used in tests.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/softmaxlayer.cuh"

using namespace std;

class MockSoftMaxLayer : public SoftMaxLayer
{
private:
	// Stabilizes inputs to prevent overflow. We acomplish this by substracting maximum value of input activations (for each input sample)
	// from all the input activations, before computing the exponentials.
	void StabilizeInputs();

	// Calculates soft maximums.
	void CalculateSoftMaximums();

	// Calculates negative log likelihoods. We are calculating it in this layer instead of output layer for better numerical stability.
	void CalculateNegativeLogLikelihoods(uint* dataLabels);

	// Does backward prop in case of cross entropy loss in output layer.
	void CrossEntropyBackwardProp(uint* dataLabels);

public:
	// Constructor.
	MockSoftMaxLayer(uint inputDataSize, uint inputDataCount);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockSoftMaxLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	virtual void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};