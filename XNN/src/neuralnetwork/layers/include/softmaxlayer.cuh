// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;

/*
	Soft Max layer calculates soft maximums of input activations, so they sum to 1 and can be used as probabilities of prediction.
*/
class SoftMaxLayer : public Layer
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

protected:
	// Buffer to store maximums of input activations for each input sample.
	float* m_inputActivationsMaxBuffer;

	// Buffer to store sums of exponentials of input activations for each input sample.
	float* m_exponentialsSumBuffer;

	// Buffer to store negative log likelihoods for each input sample.
	float* m_NLLsBuffer;

public:
	// Constructor.
	SoftMaxLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint inputDataSize,
		uint inputDataCount, bool holdsInputData);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~SoftMaxLayer();

	// Gets negative log likelihoods buffer.
	float* GetNegativeLogLikelihoodsBuffer() const { return m_NLLsBuffer; }

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};