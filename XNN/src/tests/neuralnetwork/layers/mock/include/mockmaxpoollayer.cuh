// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network max pool layer, used in tests.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/maxpoollayer.cuh"

using namespace std;

class MockMaxPoolLayer : public MaxPoolLayer
{
public:
	// Constructor.
	MockMaxPoolLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
		uint unitHeight, int paddingX, int paddingY, uint unitStride);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockMaxPoolLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	virtual void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};