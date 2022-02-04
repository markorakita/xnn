// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for neural network activation functions.
// Created: 03/22/2021.
// ----------------------------------------------------------------------------------------------------

#include "include/testactivationfunctions.cuh"

#include <cuda_runtime.h>

#include "layers/mock/include/mockinputlayer.cuh"
#include "layers/mock/include/mockoutputlayer.cuh"
#include "mock/include/mockactivationfunctions.cuh"
#include "../include/testingutils.cuh"
#include "../../neuralnetwork/include/activationfunctions.cuh"
#include "../../neuralnetwork/include/neuralnet.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/cudaasserts.cuh"

const float TestActivationFunctions::c_activationAlpha = 0.01f;
const uint TestActivationFunctions::c_numActivations = 1024;

TestActivationFunctions::TestActivationFunctions()
{
    // Registering tests.
    m_tests["applyreluactivation"] = bind(&TestActivationFunctions::TestApplyReLUActivation, this);
    m_tests["applyeluactivation"] = bind(&TestActivationFunctions::TestApplyELUActivation, this);
    m_tests["applyleakyreluactivation"] = bind(&TestActivationFunctions::TestApplyLeakyReLUActivation, this);
    m_tests["applysigmoidactivation"] = bind(&TestActivationFunctions::TestApplySigmoidActivation, this);
    m_tests["applytanhactivation"] = bind(&TestActivationFunctions::TestApplyTanhActivation, this);
    m_tests["calculatereluactivationgradient"] = bind(&TestActivationFunctions::TestCalculateReLUActivationGradient, this);
    m_tests["calculateeluactivationgradient"] = bind(&TestActivationFunctions::TestCalculateELUActivationGradient, this);
    m_tests["calculateleakyreluactivationgradient"] = bind(&TestActivationFunctions::TestCalculateLeakyReLUActivationGradient, this);
    m_tests["calculatesigmoidactivationgradient"] = bind(&TestActivationFunctions::TestCalculateSigmoidActivationGradient, this);
    m_tests["calculatetanhactivationgradient"] = bind(&TestActivationFunctions::TestCalculateTanhActivationGradient, this);
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

void TestActivationFunctions::ApplyTestActivations(ActivationType activationType, float* hostActivationsBufferBF, float* deviceActivationsBuffer,
    NeuralNet* neuralNet)
{
    MockInputLayer mockInputLayer(1, c_numActivations, 1, 1, neuralNet->GetCurandStatesBuffers()[0]);
    mockInputLayer.AllocateBuffers(false);
    mockInputLayer.GenerateActivationFromNormalDistribution(0.f, 3.0f);

    float* hostPreactivationsBuffer;
    CudaAssert(cudaMallocHost<float>(&hostPreactivationsBuffer, c_numActivations * sizeof(float)));
    CudaAssert(cudaMemcpy(hostPreactivationsBuffer, mockInputLayer.GetActivationDataBuffer(), c_numActivations * sizeof(float), cudaMemcpyDeviceToHost));

    ApplyActivation(activationType, c_activationAlpha, mockInputLayer.GetActivationDataBuffer(), c_numActivations, deviceActivationsBuffer, 0);
    ApplyActivationBF(activationType, c_activationAlpha, hostPreactivationsBuffer, c_numActivations, hostActivationsBufferBF);
    CudaAssert(cudaDeviceSynchronize());

    CudaAssert(cudaFreeHost(hostPreactivationsBuffer));
}

bool TestActivationFunctions::TestApplyActivation(ActivationType activationType, float maxComparisonDiff)
{
    NeuralNet neuralNet(1);
    size_t activationsBufferSize = c_numActivations * sizeof(float);

    float* hostActivationsBufferBF;
    CudaAssert(cudaMallocHost<float>(&hostActivationsBufferBF, activationsBufferSize));

    float* deviceActivationsBuffer;
    CudaAssert(cudaMalloc<float>(&deviceActivationsBuffer, activationsBufferSize));

    ApplyTestActivations(activationType, hostActivationsBufferBF, deviceActivationsBuffer, &neuralNet);

    float* hostActivationsBuffer;
    CudaAssert(cudaMallocHost<float>(&hostActivationsBuffer, c_numActivations * sizeof(float)));
    CudaAssert(cudaMemcpy(hostActivationsBuffer, deviceActivationsBuffer, c_numActivations * sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaFree(deviceActivationsBuffer));

    bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
    size_t numDifferences;
    float firstDifference, firstDifferentMock, firstDifferentReg;
    CompareBuffers(hostActivationsBuffer, hostActivationsBufferBF, c_numActivations, maxComparisonDiff, 0.1f, 0.1f * maxComparisonDiff,
        correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
        foundDifferentFromZeroReg);

    CudaAssert(cudaFreeHost(hostActivationsBufferBF));
    CudaAssert(cudaFreeHost(hostActivationsBuffer));

    if (!foundDifferentFromZeroMock)
    {
        EmitWarning("All BF calculated activations are zeros!");
        return false;
    }
    else if (!foundDifferentFromZeroReg)
    {
        EmitWarning("All calculated activations are zeros!");
        return false;
    }
    else if (!correctResult)
    {
        EmitWarning("Incorrectly calculated activations! Num differences: " + to_string(numDifferences) + "; First difference: " +
            to_string(firstDifference) + "; First different BF activation: " + to_string(firstDifferentMock) + "; First different activation: " +
            to_string(firstDifferentReg));
        return false;
    }

    return true;
}

bool TestActivationFunctions::TestCalculateActivationGradient(ActivationType activationType, float maxComparisonDiff)
{
    NeuralNet neuralNet(1);
    size_t activationsBufferSize = c_numActivations * sizeof(float);

    float* hostActivationsBufferBF;
    CudaAssert(cudaMallocHost<float>(&hostActivationsBufferBF, activationsBufferSize));

    float* deviceActivationsBuffer;
    CudaAssert(cudaMalloc<float>(&deviceActivationsBuffer, activationsBufferSize));

    ApplyTestActivations(activationType, hostActivationsBufferBF, deviceActivationsBuffer, &neuralNet);

    MockOutputLayer mockOutputLayer(c_numActivations, 1, LossFunctionType::CrossEntropy, false, 0, neuralNet.GetCurandStatesBuffers()[0],
        0.f, 0.5f);
    mockOutputLayer.AllocateBuffers(true);
    mockOutputLayer.DoBackwardProp();
    mockOutputLayer.SynchronizeCalculations();

    float* hostActivationGradientsBuffer;
    CudaAssert(cudaMallocHost<float>(&hostActivationGradientsBuffer, activationsBufferSize));
    CudaAssert(cudaMemcpy(hostActivationGradientsBuffer, mockOutputLayer.GetInputGradientsBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

    float* devicePreactivationGradientsBuffer;
    CudaAssert(cudaMalloc<float>(&devicePreactivationGradientsBuffer, activationsBufferSize));

    float* hostPreactivationGradientsBuffer;
    CudaAssert(cudaMallocHost<float>(&hostPreactivationGradientsBuffer, activationsBufferSize));

    float* hostPreactivationGradientsBufferBF;
    CudaAssert(cudaMallocHost<float>(&hostPreactivationGradientsBufferBF, activationsBufferSize));

    CalculatePreactivationGradients(activationType, c_activationAlpha, mockOutputLayer.GetInputGradientsBuffer(), deviceActivationsBuffer,
        c_numActivations, devicePreactivationGradientsBuffer, 0);
    CalculatePreactivationGradientsBF(activationType, c_activationAlpha, hostActivationGradientsBuffer, hostActivationsBufferBF,
        c_numActivations, hostPreactivationGradientsBufferBF);
    CudaAssert(cudaDeviceSynchronize());

    CudaAssert(cudaMemcpy(hostPreactivationGradientsBuffer, devicePreactivationGradientsBuffer, activationsBufferSize, cudaMemcpyDeviceToHost));

    CudaAssert(cudaFree(devicePreactivationGradientsBuffer));
    CudaAssert(cudaFreeHost(hostActivationGradientsBuffer));
    CudaAssert(cudaFree(deviceActivationsBuffer));
    CudaAssert(cudaFreeHost(hostActivationsBufferBF));

    bool correctResult, foundDifferentFromZeroMock, foundDifferentFromZeroReg;
    size_t numDifferences;
    float firstDifference, firstDifferentMock, firstDifferentReg;
    CompareBuffers(hostPreactivationGradientsBufferBF, hostPreactivationGradientsBuffer, c_numActivations, maxComparisonDiff, 0.1f, 0.1f * maxComparisonDiff,
        correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
        foundDifferentFromZeroReg);

    CudaAssert(cudaFreeHost(hostPreactivationGradientsBuffer));
    CudaAssert(cudaFreeHost(hostPreactivationGradientsBufferBF));

    if (!foundDifferentFromZeroMock)
    {
        EmitWarning("All BF calculated activation gradients are zeros!");
        return false;
    }
    else if (!foundDifferentFromZeroReg)
    {
        EmitWarning("All calculated activation gradients are zeros!");
        return false;
    }
    else if (!correctResult)
    {
        EmitWarning("Incorrectly calculated activation gradients! Num differences: " + to_string(numDifferences) + "; First difference: " +
            to_string(firstDifference) + "; First different BF activation gradient: " + to_string(firstDifferentMock) +
            "; First different activation gradient: " + to_string(firstDifferentReg));
        return false;
    }

    return true;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestActivationFunctions::TestApplyReLUActivation()
{
    return TestApplyActivation(ActivationType::ReLU, 0.000001f);
}

bool TestActivationFunctions::TestApplyELUActivation()
{
    return TestApplyActivation(ActivationType::ELU, 0.00001f);
}

bool TestActivationFunctions::TestApplyLeakyReLUActivation()
{
    return TestApplyActivation(ActivationType::LeakyReLU, 0.000001f);
}

bool TestActivationFunctions::TestApplySigmoidActivation()
{
    return TestApplyActivation(ActivationType::Sigmoid, 0.0001f);
}

bool TestActivationFunctions::TestApplyTanhActivation()
{
    return TestApplyActivation(ActivationType::Tanh, 0.0001f);
}

bool TestActivationFunctions::TestCalculateReLUActivationGradient()
{
    return TestCalculateActivationGradient(ActivationType::ReLU, 0.000001f);
}

bool TestActivationFunctions::TestCalculateELUActivationGradient()
{
    return TestCalculateActivationGradient(ActivationType::ELU, 0.00001f);
}

bool TestActivationFunctions::TestCalculateLeakyReLUActivationGradient()
{
    return TestCalculateActivationGradient(ActivationType::LeakyReLU, 0.000001f);
}

bool TestActivationFunctions::TestCalculateSigmoidActivationGradient()
{
    return TestCalculateActivationGradient(ActivationType::Sigmoid, 0.0001f);
}

bool TestActivationFunctions::TestCalculateTanhActivationGradient()
{
    return TestCalculateActivationGradient(ActivationType::Tanh, 0.0001f);
}