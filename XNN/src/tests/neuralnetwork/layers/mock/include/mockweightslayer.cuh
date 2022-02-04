// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network weights layer, used in tests.
// Created: 01/24/2021.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/weightslayer.cuh"

using namespace std;

enum class ActivationType;

class MockWeightsLayer : public WeightsLayer
{
private:
    // Updates layer parameters by applying momentum to last update, learning rate to gradients, and decay to parameters.
    void static ApplyParamatersUpdate(float* paramsBuffer, float* gradientsBuffer, float* updatesBuffer, uint numElements, float updateMomentum,
        float updateDecay, float startingLearningRate, float learningRateProgressStep, float learningRateUpdateFactor, float learningProgress);

public:
    // Constructor.
    MockWeightsLayer(uint indexInTier, size_t weightsBufferSize, uint numWeightsPerNeuron, float weightsUpdateMomentum, float weightsUpdateDecay,
        float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor,
        size_t biasesBufferSize, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
        float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType, float activationAlpha);

    // Allocates internal data buffers used in this layer.
    virtual void AllocateBuffers(bool allocateTrainBuffers);

    // Base destructor.
    virtual ~MockWeightsLayer();

    // Initializes host buffer to initial value.
    void InitializeHostBuffer(float* hostBuffer, size_t bufferSize, float initialValue);

    // Copies weights from device buffer.
    void CopyWeightsFromDevice(float* deviceWeightsBuffer);

    // Copies biases from device buffer.
    void CopyBiasesFromDevice(float* deviceBiasesBuffer);

    // Updates layer's parameters (weights, biases, etc.)
    virtual void UpdateLayerParameters(float learningProgress);
};