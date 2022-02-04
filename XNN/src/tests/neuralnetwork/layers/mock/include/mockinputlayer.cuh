// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network input layer, used in tests.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

typedef struct curandStateXORWOW curandState;

class MockInputLayer : public Layer
{
private:
	// Buffer for cuRAND states.
	curandState* m_curandStatesBuffer;

public:
	// Constructor.
	MockInputLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, curandState* curandStatesBuffer);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Generates activation data with random values sampled from uniform distribution.
	void GenerateActivationFromUniformDistribution(float rangeStart, float rangeEnd);

	// Generates activation data with random values sampled from uniform whole integer distribution.
	void GenerateActivationFromUniformIntDistribution(int rangeStart, int rangeEnd);

	// Generates activation data with random values sampled from normal distribution.
	void GenerateActivationFromNormalDistribution(float mean, float stDev);

	// Loads input to layer.
	virtual void LoadInputs() {}

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode) {}

	// Does backward propagation through layer.
	virtual void DoBackwardProp() {}
};