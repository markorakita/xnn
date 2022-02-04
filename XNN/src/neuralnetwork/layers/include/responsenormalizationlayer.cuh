// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network response normalization layer.
// Created: 02/09/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;

/*
	Response normalization layer implements a form of lateral inhibition inspired by the type found in real neurons,
	creating competition for big activities amongst neuron outputs computed using different kernels.

	It is implemented by formula:
		activation[i] = input[i] / (bias + (alpha / depth) * sum[j](input[j] ^ 2)) ^ beta
*/
class ResponseNormalizationLayer : public Layer
{
protected:
	// Depth of normalization.
	uint m_depth;

	// Normalization bias.
	float m_bias;

	// Normalization alpha coefficient (see the formula above).
	float m_alphaCoeff;

	// Normalization beta coefficient (see the formula above).
	float m_betaCoeff;

public:
	// Constructor.
	ResponseNormalizationLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint indexInTier,
		uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint depth, float bias,
		float alphaCoeff, float betaCoeff, bool holdsActivationGradients);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~ResponseNormalizationLayer() {}

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};