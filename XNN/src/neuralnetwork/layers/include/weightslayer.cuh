// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract neural network layer with weights.
// Created: 01/22/2021.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"

using namespace std;

enum class ActivationType;
typedef struct curandStateXORWOW curandState;

class WeightsLayer : public Layer
{
protected:
    // Weights buffer.
    float* m_weightsBuffer;

    // Weights buffer size.
    size_t m_weightsBufferSize;

    // Number of weights per neuron.
    uint m_numWeightsPerNeuron;

    // Weights gradients buffer.
    float* m_weightsGradientsBuffer;

    // Weights update buffer.
    float* m_weightsUpdateBuffer;

    // Weights update momentum.
    float m_weightsUpdateMomentum;

    // Weights update decay.
    float m_weightsUpdateDecay;

    // Weights update learning rate progress step.
    float m_weightsUpdateLearningRateProgressStep;

    // Weights update starting learning rate.
    float m_weightsUpdateStartingLearningRate;

    // Weights update learning rate update factor.
    float m_weightsUpdateLearningRateUpdateFactor;

    // Biases buffer.
    float* m_biasesBuffer;

    // Biases buffer size.
    size_t m_biasesBufferSize;

    // Biases gradients buffer.
    float* m_biasesGradientsBuffer;

    // Biases update buffer.
    float* m_biasesUpdateBuffer;

    // Biases update momentum.
    float m_biasesUpdateMomentum;

    // Biases update decay.
    float m_biasesUpdateDecay;

    // Biases update learning rate progress step.
    float m_biasesUpdateLearningRateProgressStep;

    // Biases update starting learning rate.
    float m_biasesUpdateStartingLearningRate;

    // Biases update learning rate update factor.
    float m_biasesUpdateLearningRateUpdateFactor;

    // Buffer for cuRAND states.
    curandState* m_curandStatesBuffer;

    // Activation type of this layer.
    ActivationType m_activationType;

    // Activation alpha parameter.
    float m_activationAlpha;

    // Calculates gradients of weights.
    virtual void CalculateWeightsGradients() = 0;

    // Calculates gradients of biases.
    virtual void CalculateBiasesGradients() = 0;

    // Base constructor.
    WeightsLayer(uint indexInTier, size_t weightsBufferSize, uint numWeightsPerNeuron, float weightsUpdateMomentum, float weightsUpdateDecay,
        float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor,
        size_t biasesBufferSize, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
        float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType,
        float activationAlpha, curandState* curandStatesBuffer);

public:
    // Allocates internal data buffers used in this layer.
    virtual void AllocateBuffers(bool allocateTrainBuffers);

    // Base destructor.
    virtual ~WeightsLayer();

    // Initializes weights to constant value.
    void InitializeWeightsToConstant(float initialValue);

    // Initializes weights with random values sampled from uniform distribution.
    void InitializeWeightsFromUniformDistribution(float rangeStart, float rangeEnd);

    // Initializes weights with random values sampled from normal distribution.
    void InitializeWeightsFromNormalDistribution(float mean, float stDev);

    // Initializes weights using Xavier method.
    void InitializeWeightsXavier();

    // Initializes weights using He method.
    void InitializeWeightsHe();

    // Initializes biases to constant value.
    void InitializeBiasesToConstant(float initialValue);

    // Initializes biases with random values sampled from uniform distribution.
    void InitializeBiasesFromUniformDistribution(float rangeStart, float rangeEnd);

    // Initializes biases with random values sampled from normal distribution.
    void InitializeBiasesFromNormalDistribution(float mean, float stDev);

    // Copies weights from host buffer.
    void CopyWeightsFromHost(float* hostWeightsBuffer);

    // Copies weights update buffer from host buffer.
    void CopyWeightsUpdateFromHost(float* hostWeightsUpdateBuffer);

    // Copies biases from host buffer.
    void CopyBiasesFromHost(float* hostBiasesBuffer);

    // Copies biases update buffer from host buffer.
    void CopyBiasesUpdateFromHost(float* hostBiasesUpdateBuffer);

    // Copies weights from other layer.
    void CopyWeightsFromLayer(WeightsLayer* weightsLayer);

    // Copies biases from other layer.
    void CopyBiasesFromLayer(WeightsLayer* weightsLayer);

    // Gets weights buffer.
    float* GetWeightsBuffer() const { return m_weightsBuffer; }

    // Gets weights update buffer.
    float* GetWeightsUpdateBuffer() const { return m_weightsUpdateBuffer; }

    // Gets weights buffer size.
    size_t GetWeightsBufferSize() const { return m_weightsBufferSize; }

    // Gets weights gradients buffer.
    float* GetWeightsGradientsBuffer() const { return m_weightsGradientsBuffer; }

    // Gets biases buffer.
    float* GetBiasesBuffer() const { return m_biasesBuffer; }

    // Gets biases update buffer.
    float* GetBiasesUpdateBuffer() const { return m_biasesUpdateBuffer; }

    // Gets biases buffer size.
    size_t GetBiasesBufferSize() const { return m_biasesBufferSize; }

    // Gets biases gradients buffer.
    float* GetBiasesGradientsBuffer() const { return m_biasesGradientsBuffer; }

    // Updates layer's parameters (weights, biases, etc.)
    virtual void UpdateLayerParameters(float learningProgress);
};