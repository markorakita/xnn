// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;
typedef struct curandStateXORWOW curandState;

/*
	Dropout layer provides efficient way to simulate combining multiple trained models to reduce test error and prevent overfitting.
	It works by dropping each neuron activity with certain probability, preventing complex coadaptations between neurons.
*/
class DropoutLayer : public Layer
{
private:
	// Creates dropout filter.
	void CreateDropoutFilter();

	// Applies dropout filter.
	void ApplyDropoutFilter();

protected:
	// Dropout filter.
	float* m_dropoutFilter;

	// Biases buffer size.
	size_t m_dropoutFilterSize;

	// Buffer for cuRAND states.
	curandState* m_curandStatesBuffer;

	// Probability for dropping each activity.
	float m_dropProbability;

	// Should we use dropout filter from host.
	bool m_useHostDropoutFilter;

	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Constructor.
	DropoutLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, curandState* curandStatesBuffer,
		uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, float dropProbability,
		bool useHostDropoutFilter, bool holdsActivationGradients);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~DropoutLayer();

	// Copies dropout filter from host.
	void CopyDropoutFilterFromHost(float* hostDropoutFilter);

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};