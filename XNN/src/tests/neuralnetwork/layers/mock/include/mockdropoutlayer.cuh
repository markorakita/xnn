// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network dropout layer, used in tests.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/dropoutlayer.cuh"

using namespace std;

typedef struct curandStateXORWOW curandState;

class MockDropoutLayer : public DropoutLayer
{
private:
	// Creates dropout filter.
	void CreateDropoutFilter();

	// Applies dropout filter.
	void ApplyDropoutFilter();

public:
	// Constructor.
	MockDropoutLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability,
		curandState* curandStatesBuffer);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~MockDropoutLayer();

	// Gets dropout filter.
	float* GetDropoutFilter() { return m_dropoutFilter; }

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	virtual void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};