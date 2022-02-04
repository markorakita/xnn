// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network standard layer.
// Created: 01/17/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "weightslayer.cuh"

using namespace std;

enum class ActivationType;
typedef struct cublasContext* cublasHandle_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct curandStateXORWOW curandState;

/*
    Standard neural network layer, with neurons and weights.

    Weight buffer is matrix with these specifications:
        num_columns = num_weights_per_neuron
        num_rows    = num_neurons

    Preactivations and activations buffers are matrices with these specifications:
        num_columns = data_count
        num_rows    = num_neurons (i.e. preactivations/activations count)
*/
class StandardLayer : public WeightsLayer
{
private:
    // Handle for cuBLAS operations.
    cublasHandle_t m_cublasHandle;

    // Number of neurons in standard layer.
    uint m_numNeurons;

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
    
    // Calculates gradients of weights.
    virtual void CalculateWeightsGradients();

    // Calculates gradients of biases.
    virtual void CalculateBiasesGradients();

    // Calculates gradients of inputs.
    void CalculateInputGradients();

    // Calculates gradients of preactivations.
    void CalculatePreactivationsGradients();

    // Reinitializes layer when input data count changes.
    virtual void Reinitialize(uint newInputDataCount);

public:
    // Constructor.
    StandardLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, cublasHandle_t cublasHandle,
        curandState* curandStatesBuffer, uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount,
        bool holdsInputData, uint numNeurons, float weightsUpdateMomentum, float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep,
        float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay,
        float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor,
        ActivationType activationType, float activationAlpha, bool holdsActivationGradients);

    // Allocates internal data buffers used in this layer.
    virtual void AllocateBuffers(bool allocateTrainBuffers);

    // Destructor.
    virtual ~StandardLayer();

    // Gets number of neurons.
    uint GetNumberOfNeurons() const { return m_numNeurons; }

    // Does forward propagation through layer.
    virtual void DoForwardProp(PropagationMode propagationMode);

    // Does backward propagation through layer.
    virtual void DoBackwardProp();
};