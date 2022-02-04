// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network output layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;

// Loss function types.
enum class LossFunctionType
{
	LogisticRegression,
	CrossEntropy
};

/*
	Output layer calculates loss function and accuracy.
*/
class OutputLayer : public Layer
{
private:
	// Forward prop for logistic regression loss function.
	void LogisticRegressionForwardProp(float* inputBuffer);

	// Forward prop for cross entropy loss function.
	void CrossEntropyForwardProp(float* inputBuffer);

	// Backward prop for logistic regression loss function.
	void LogisticRegressionBackwardProp();

protected:
	// Loss function type.
	LossFunctionType m_lossFunctionType;

	// Total value of loss function for current batch.
	float m_loss;

	// Total accuracy for current batch.
	float m_accuracy;

	// Labels buffer for input data samples.
	uint* m_dataLabels;

	// Host labels buffer.
	uint* m_hostLabelsBuffer;

	// Size of the labels buffer.
	size_t m_labelsBufferSize;

	// Labels offset for the case when we have data parallelism.
	uint m_labelsOffset;

	// Buffer for loss values.
	float* m_lossBuffer;

	// Host buffer for loss values.
	float* m_hostLossBuffer;

	// Score values for each data sample.
	float* m_scores;

	// Host buffer for score values.
	float* m_hostScores;

	// Size for loss and score buffers.
	size_t m_lossBuffersSize;

	// Should we calculate multiple guess accuracy.
	bool m_calculateMultipleGuessAccuracy;

	// Number of guesses for predicting correct output value, if we are calculating multiple guess accuracy.
	uint m_numGuesses;

	// Multiple guess score values for each data sample.
	float* m_multipleGuessScores;

	// Host buffer for multiple guess score values.
	float* m_hostMultipleGuessScores;

	// Multiple guess total accuracy for current batch.
	float m_multipleGuessAccuracy;

	// Number of test passes for one image.
	uint m_numTestPasses;

	// Counter of test passes.
	uint m_testPassCounter;

	// Test buffer for storing average value of inputs from each pass.
	float* m_testAverageInputsBuffer;

	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Constructor.
	OutputLayer(cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint inputDataSize, uint inputDataCount, uint labelsCount,
		LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses, uint numTestPasses);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~OutputLayer();

	// Loads labels for input data samples.
	void LoadDataLabels(vector<uint> dataLabels);

	// Moves labels offset for the next batch of data in case of data parallelism.
	void MoveLabelsOffset() { m_labelsOffset += m_inputDataCount; }

	// Gets labels buffer for input data samples.
	uint* GetDataLabels() const { return m_dataLabels + m_labelsOffset; }

	// Gets loss function type of output layer.
	LossFunctionType GetLossFunctionType() const { return m_lossFunctionType; }

	// Gets total value of loss function for current batch.
	float GetLoss() const { return m_loss; }

	// Gets current batch total accuracy.
	float GetAccuracy() const { return m_accuracy; }

	// Should we calculate multiple guess accuracy.
	bool ShouldCalculateMultipleGuessAccuracy() const { return m_calculateMultipleGuessAccuracy; }

	// Gets current batch multiple guess total accuracy.
	float GetMultipleGuessAccuracy() const { return m_multipleGuessAccuracy; }

	// Gets scores.
	float* GetScores() const { return m_scores; }

	// Gets host scores.
	float* GetHostScores() const { return m_hostScores; }

	// Gets multiple guess scores.
	float* GetMultipleGuessScores() const { return m_multipleGuessScores; }

	// Gets host multiple guess scores.
	float* GetHostMultipleGuessScores() const { return m_hostMultipleGuessScores; }

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};