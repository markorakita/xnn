// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network weights layer, used in tests.
// Created: 01/24/2021.
// ----------------------------------------------------------------------------------------------------

#include "include/mockweightslayer.cuh"

#include "../../../../utils/include/cudaasserts.cuh"
#include "../../../../utils/include/cudahelper.cuh"

MockWeightsLayer::MockWeightsLayer(uint indexInTier, size_t weightsBufferSize, uint numWeightsPerNeuron, float weightsUpdateMomentum,
    float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate,
    float weightsUpdateLearningRateUpdateFactor, size_t biasesBufferSize, float biasesUpdateMomentum, float biasesUpdateDecay,
    float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor,
    ActivationType activationType, float activationAlpha)
    :
    WeightsLayer(indexInTier, weightsBufferSize, numWeightsPerNeuron, weightsUpdateMomentum, weightsUpdateDecay,
        weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesBufferSize,
        biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate,
        biasesUpdateLearningRateUpdateFactor, activationType, activationAlpha, NULL)
{
}

void MockWeightsLayer::AllocateBuffers(bool allocateTrainBuffers)
{
    // Allocating weights buffers.
    CudaAssert(cudaMallocHost<float>(&m_weightsBuffer, m_weightsBufferSize));

    // Allocating biases buffer.
    CudaAssert(cudaMallocHost<float>(&m_biasesBuffer, m_biasesBufferSize));

    // Allocating buffers necessary for training.
    if (allocateTrainBuffers)
    {
        // Allocating weights gradients buffer.
        CudaAssert(cudaMallocHost<float>(&m_weightsGradientsBuffer, m_weightsBufferSize));

        // Allocating weights update buffer.
        CudaAssert(cudaMallocHost<float>(&m_weightsUpdateBuffer, m_weightsBufferSize));
        InitializeHostBuffer(m_weightsUpdateBuffer, (uint)(m_weightsBufferSize / sizeof(float)), 0.f);

        // Allocating biases gradients buffer.
        CudaAssert(cudaMallocHost<float>(&m_biasesGradientsBuffer, m_biasesBufferSize));

        // Allocating biases update buffer.
        CudaAssert(cudaMallocHost<float>(&m_biasesUpdateBuffer, m_biasesBufferSize));
        InitializeHostBuffer(m_biasesUpdateBuffer, (uint)(m_biasesBufferSize / sizeof(float)), 0.f);
    }
}

MockWeightsLayer::~MockWeightsLayer()
{
    // Setting buffers to null after freeing, to avoid freeing them again in parent weightslayer.
    if (m_weightsBuffer != NULL)
    {
        CudaAssert(cudaFreeHost(m_weightsBuffer));
        m_weightsBuffer = NULL;
    }
    if (m_weightsGradientsBuffer != NULL)
    {
        CudaAssert(cudaFreeHost(m_weightsGradientsBuffer));
        m_weightsGradientsBuffer = NULL;
    }
    if (m_weightsUpdateBuffer != NULL)
    {
        CudaAssert(cudaFreeHost(m_weightsUpdateBuffer));
        m_weightsUpdateBuffer = NULL;
    }
    if (m_biasesBuffer != NULL)
    {
        CudaAssert(cudaFreeHost(m_biasesBuffer));
        m_biasesBuffer = NULL;
    }
    if (m_biasesGradientsBuffer != NULL)
    {
        CudaAssert(cudaFreeHost(m_biasesGradientsBuffer));
        m_biasesGradientsBuffer = NULL;
    }
    if (m_biasesUpdateBuffer != NULL)
    {
        CudaAssert(cudaFreeHost(m_biasesUpdateBuffer));
        m_biasesUpdateBuffer = NULL;
    }
}

void MockWeightsLayer::InitializeHostBuffer(float* hostBuffer, size_t bufferSize, float initialValue)
{
    float* deviceBuffer;
    CudaAssert(cudaMalloc<float>(&deviceBuffer, bufferSize));

    InitializeBufferToConstant(deviceBuffer, (uint)(bufferSize / sizeof(float)), initialValue);
    SynchronizeCalculations();

    CudaAssert(cudaMemcpy(hostBuffer, deviceBuffer, bufferSize, cudaMemcpyDeviceToHost));

    CudaAssert(cudaFree(deviceBuffer));
}

void MockWeightsLayer::CopyWeightsFromDevice(float* deviceWeightsBuffer)
{
    CudaAssert(cudaMemcpy(m_weightsBuffer, deviceWeightsBuffer, m_weightsBufferSize, cudaMemcpyDeviceToHost));
}

void MockWeightsLayer::CopyBiasesFromDevice(float* deviceBiasesBuffer)
{
    CudaAssert(cudaMemcpy(m_biasesBuffer, deviceBiasesBuffer, m_biasesBufferSize, cudaMemcpyDeviceToHost));
}

void MockWeightsLayer::ApplyParamatersUpdate(float* paramsBuffer, float* gradientsBuffer, float* updatesBuffer, uint numElements, float updateMomentum,
    float updateDecay, float startingLearningRate, float learningRateProgressStep, float learningRateUpdateFactor, float learningProgress)
{
    float updateProgressSteps = floorf(learningProgress / learningRateProgressStep);
    const float learningRate = startingLearningRate * powf(learningRateUpdateFactor, updateProgressSteps);
    for (uint i = 0; i < numElements; ++i)
    {
        updatesBuffer[i] = updateMomentum * updatesBuffer[i] + learningRate * (gradientsBuffer[i] + updateDecay * paramsBuffer[i]);
        paramsBuffer[i] -= updatesBuffer[i];
    }
}

void MockWeightsLayer::UpdateLayerParameters(float learningProgress)
{
    // Updating weights.
    ApplyParamatersUpdate(m_weightsBuffer, m_weightsGradientsBuffer, m_weightsUpdateBuffer, (uint)(m_weightsBufferSize / sizeof(float)), m_weightsUpdateMomentum,
        m_weightsUpdateDecay, m_weightsUpdateStartingLearningRate, m_weightsUpdateLearningRateProgressStep, m_weightsUpdateLearningRateUpdateFactor,
        learningProgress);

    // Updating biases.
    ApplyParamatersUpdate(m_biasesBuffer, m_biasesGradientsBuffer, m_biasesUpdateBuffer, (uint)(m_biasesBufferSize / sizeof(float)), m_biasesUpdateMomentum,
        m_biasesUpdateDecay, m_biasesUpdateStartingLearningRate, m_biasesUpdateLearningRateProgressStep, m_biasesUpdateLearningRateUpdateFactor,
        learningProgress);
}