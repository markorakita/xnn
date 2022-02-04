// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network convolutional layer.
// Created: 01/03/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "weightslayer.cuh"
#include "../../include/activationfunctions.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;
typedef struct curandStateXORWOW curandState;

/*
    From Wikipedia:

    Convolutional layer is the core building block of a CNN. The layers parameters consist of a set of learnable filters (or kernels), which have
    a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across
    the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional
    activation map of that filter. As a result, the network learns filters that activate when they see some specific type of feature at some spatial
    position in the input.
*/
class ConvolutionalLayer : public WeightsLayer
{
private:
    // Number of convolutional filters.
    uint m_numFilters;

    // Width of a filter.
    uint m_filterWidth;

    // Height of a filter.
    uint m_filterHeight;

    // Size of a filter.
    uint m_filterSize;

    // Number of channels per filter.
    uint m_numFilterChannels;

    // Padding in dimension X.
    int m_paddingX;

    // Padding in dimension Y.
    int m_paddingY;

    // Stride for patching.
    uint m_stride;

    // Number of patches to apply filters on in dimension X.
    uint m_numPatchesX;

    // Number of patches to apply filters on in dimension Y.
    uint m_numPatchesY;

    // Preactivations buffer.
    float* m_preactivationDataBuffer;

    // Preactivations gradients buffer.
    float* m_preactivationGradientsBuffer;

    // Helper buffer for calculating filters' weights gradients per chunk of preactivation gradients.
    float* m_weightsGradientsPerChunkBuffer;

    // How many preactivation gradients are per chunk width for calculation of filters' weights gradients per chunk.
    uint m_preactivationGradientsPerChunkWidth;

    // Buffer for holding partial sums for calculating filters' biases gradients.
    float* m_biasesGradientsPartialSumsBuffer;

    // How many summations will be done per thread for partial sums for calculating filters' biases gradients.
    static const uint c_biasesGradientsSumsPerThread;

    // How many threads will be used pre block to calculate partial sums for one filter bias gradient.
    static const uint c_biasesGradientsPartialSumThreadsPerBlock;

    // How many blocks will be used to calculate partial sums for one filter bias gradient.
    uint m_biasesGradientsPartialSumBlocks;

    // Calculates preactivations.
    void CalculatePreactivations();

    // Adds biases to preactivations.
    void AddBiases();

    // Calculates activations.
    void CalculateActivations();

    // Calculates gradients of filters' weights.
    virtual void CalculateWeightsGradients();

    // Calculates gradients of filters' biases.
    virtual void CalculateBiasesGradients();

    // Calculates gradients of inputs.
    void CalculateInputGradients();

    // Calculates gradients of preactivations.
    void CalculatePreactivationsGradients();

    // Reinitializes layer when input data count changes.
    virtual void Reinitialize(uint newInputDataCount);

public:
    // Constructor.
    ConvolutionalLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, curandState* curandStatesBuffer,
        uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData,
        uint numFilters, uint filterWidth, uint filterHeight, uint numFilterChannels, float filtersUpdateMomentum, float filtersUpdateDecay,
        float filtersUpdateLearningRateProgressStep, float filtersUpdateStartingLearningRate, float filtersUpdateLearningRateUpdateFactor, float biasesUpdateMomentum,
        float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor,
        int paddingX, int paddingY, uint stride, ActivationType activationType, float activationAlpha, bool holdsActivationGradients);

    // Allocates internal data buffers used in this layer.
    virtual void AllocateBuffers(bool allocateTrainBuffers);

    // Destructor.
    virtual ~ConvolutionalLayer();

    // Gets number of filters.
    uint GetNumberOfFilters() const { return m_numFilters; }

    // Gets filter width.
    uint GetFilterWidth() const { return m_filterWidth; }

    // Gets filter height.
    uint GetFilterHeight() const { return m_filterHeight; }

    // Gets number of filter channels.
    uint GetNumberOfFilterChannels() const { return m_numFilterChannels; }

    // Does forward propagation through layer.
    virtual void DoForwardProp(PropagationMode propagationMode);

    // Does backward propagation through layer.
    virtual void DoBackwardProp();
};