// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network response normalization layer, used in tests.
// Created: 02/09/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/responsenormalizationlayer.cuh"

using namespace std;

class MockResponseNormalizationLayer : public ResponseNormalizationLayer
{
public:
	// Constructor.
	MockResponseNormalizationLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth, float bias,
		float alphaCoeff, float betaCoeff);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockResponseNormalizationLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	virtual void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};