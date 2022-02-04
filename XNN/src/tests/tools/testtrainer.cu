// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for the networks trainer.
// Created: 11/27/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/testtrainer.cuh"

#include "../../neuralnetwork/include/activationfunctions.cuh"
#include "../../neuralnetwork/include/neuralnet.cuh"
#include "../../neuralnetwork/layers/include/outputlayer.cuh"
#include "../../neuralnetwork/layers/include/softmaxlayer.cuh"
#include "../../neuralnetwork/layers/include/standardlayer.cuh"
#include "../../tools/include/trainer.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/deftypes.cuh"
#include "../include/testingutils.cuh"
#include "../neuralnetwork/layers/mock/include/mockinputlayer.cuh"

TestTrainer::TestTrainer()
{
    // Registering tests.
    m_tests["standardsinglevsmultigputraining"] = bind(&TestTrainer::TestStandardSingleVsMultiGpuTraining, this);
}

bool TestTrainer::CheckStandardSingleVsMultiGpuPropagationForward(Trainer* singleGpuTrainer, Trainer* multiGpuTrainer, MockInputLayer* mockInputLayer,
    OutputLayer* singleGpuOutputLayer, OutputLayer* multiGpuOutputLayer)
{
    PropagationMode propagationMode = PropagationMode::Train;
    mockInputLayer->GenerateActivationFromUniformIntDistribution(-128, 127);
    CudaAssert(cudaDeviceSynchronize());

    size_t nextTier;
    singleGpuTrainer->PropagateBatchForward(1, nextTier, propagationMode);
    multiGpuTrainer->PropagateBatchForward(1, nextTier, propagationMode);
    multiGpuTrainer->PropagateBatchForward(2, nextTier, propagationMode);
    CudaAssert(cudaDeviceSynchronize());

    // Check that both networks output activations are same.
    float* singleGpuOutputLayerActivationsBuffer;
    CudaAssert(cudaMallocHost<float>(&singleGpuOutputLayerActivationsBuffer, singleGpuOutputLayer->GetActivationBufferSize()));
    CudaAssert(cudaMemcpy(singleGpuOutputLayerActivationsBuffer, singleGpuOutputLayer->GetActivationDataBuffer(),
        singleGpuOutputLayer->GetActivationBufferSize(), cudaMemcpyDeviceToHost));
    float* multiGpuOutputLayerActivationsBuffer;
    CudaAssert(cudaMallocHost<float>(&multiGpuOutputLayerActivationsBuffer, multiGpuOutputLayer->GetActivationBufferSize()));
    CudaAssert(cudaMemcpy(multiGpuOutputLayerActivationsBuffer, multiGpuOutputLayer->GetActivationDataBuffer(),
        multiGpuOutputLayer->GetActivationBufferSize(), cudaMemcpyDeviceToHost));
    CudaAssert(cudaDeviceSynchronize());

    bool correctResult, foundDifferentFromZeroFirstBuffer, foundDifferentFromZeroSecondBuffer;
    size_t numDifferences;
    float firstDifference, firstDifferentFirstBuffer, firstDifferentSecondBuffer;
    CompareBuffers(singleGpuOutputLayerActivationsBuffer, multiGpuOutputLayerActivationsBuffer, multiGpuOutputLayer->GetActivationBufferSize() / sizeof(float),
        0.f, 0.f, 0.f, correctResult, numDifferences, firstDifference, firstDifferentSecondBuffer, firstDifferentFirstBuffer, foundDifferentFromZeroSecondBuffer,
        foundDifferentFromZeroFirstBuffer);

    CudaAssert(cudaFreeHost(singleGpuOutputLayerActivationsBuffer));
    CudaAssert(cudaFreeHost(multiGpuOutputLayerActivationsBuffer));

    if (!foundDifferentFromZeroFirstBuffer)
    {
        EmitWarning("All values in the single GPU trainer output layer activations buffer are zeros!");
        return false;
    }
    else if (!foundDifferentFromZeroSecondBuffer)
    {
        EmitWarning("All values in the multi GPU trainer output layer activations buffer are zeros!");
        return false;
    }
    else if (!correctResult)
    {
        EmitWarning("Values in output layer activations between single and multi GPU trainers differ! Num differences: " + to_string(numDifferences) +
            "; First difference: " + to_string(firstDifference) + "; First different from single GPU trainer: " + to_string(firstDifferentFirstBuffer) +
            "; First different from multi GPU trainer: " + to_string(firstDifferentSecondBuffer) + ".");
        return false;
    }

    return true;
}

bool TestTrainer::TestStandardSingleVsMultiGpuTraining()
{
    CudaAssert(cudaSetDevice(0));

    NeuralNet neuralNet(1);
    const bool allocateTrainBuffers = true;

    uint inputNumChannels = 1;
    uint inputDataWidth = 28;
    uint inputDataHeight = 28;
    uint inputDataCount = 128;
    MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, neuralNet.GetCurandStatesBuffers()[0]);
    mockInputLayer.AllocateBuffers(allocateTrainBuffers);

    // Creating single GPU trainer.
    Trainer singleGpuTrainer;
    singleGpuTrainer.m_neuralNet = new NeuralNet(1);

    // Dummy tier instead of input layer.
    singleGpuTrainer.m_neuralNet->AddLayersTier(vector<Layer*>());

    uint firstStandardLayerNumNeurons = 4096;
    float firstStandardLayerWeightsDeviation = 0.01f;
    float firstStandardLayerWeightsMomentum = 0.9f;
    float firstStandardLayerWeightsDecay = 0.0005f;
    float firstStandardLayerWeightsStartingLR = 0.01f;
    float firstStandardLayerWeightsLRStep = 0.25f;
    float firstStandardLayerWeightsLRFactor = 0.15874f;
    float firstStandardLayerBiasesInitialValue = 1.0f;
    float firstStandardLayerBiasesMomentum = 0.9f;
    float firstStandardLayerBiasesDecay = 0.f;
    float firstStandardLayerBiasesStartingLR = 0.02f;
    float firstStandardLayerBiasesLRStep = 0.5f;
    float firstStandardLayerBiasesLRFactor = 0.1f;
    ActivationType firstStandardLayerActivationType = ActivationType::ReLU;
    vector<Layer*> singleGpuFirstStandardLayerTier;
    StandardLayer* singleGpuFirstStandardLayer = new StandardLayer(ParallelismMode::Model, singleGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        singleGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], singleGpuTrainer.m_neuralNet->GetCublasHandles()[0],
        singleGpuTrainer.m_neuralNet->GetCurandStatesBuffers()[0], 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
        firstStandardLayerNumNeurons, firstStandardLayerWeightsMomentum, firstStandardLayerWeightsDecay, firstStandardLayerWeightsLRStep,
        firstStandardLayerWeightsStartingLR, firstStandardLayerWeightsLRFactor, firstStandardLayerBiasesMomentum, firstStandardLayerBiasesDecay,
        firstStandardLayerBiasesLRStep, firstStandardLayerBiasesStartingLR, firstStandardLayerBiasesLRFactor, firstStandardLayerActivationType, 0.f, false);
    singleGpuFirstStandardLayer->AllocateBuffers(allocateTrainBuffers);
    singleGpuFirstStandardLayer->InitializeWeightsFromNormalDistribution(0.f, firstStandardLayerWeightsDeviation);
    singleGpuFirstStandardLayer->InitializeBiasesToConstant(firstStandardLayerBiasesInitialValue);
    singleGpuFirstStandardLayer->AddPrevLayer(&mockInputLayer);
    singleGpuFirstStandardLayerTier.push_back(singleGpuFirstStandardLayer);
    singleGpuTrainer.m_neuralNet->AddLayersTier(singleGpuFirstStandardLayerTier);

    uint secondStandardLayerNumNeurons = 10;
    float secondStandardLayerBiasesInitialValue = 0.f;
    ActivationType secondStandardLayerActivationType = ActivationType::Linear;
    vector<Layer*> singleGpuSecondStandardLayerTier;
    StandardLayer* singleGpuSecondStandardLayer = new StandardLayer(ParallelismMode::Model, singleGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        singleGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], singleGpuTrainer.m_neuralNet->GetCublasHandles()[0],
        singleGpuTrainer.m_neuralNet->GetCurandStatesBuffers()[0], 0, 1, singleGpuFirstStandardLayer->GetActivationNumChannels(),
        singleGpuFirstStandardLayer->GetActivationDataWidth(), singleGpuFirstStandardLayer->GetActivationDataHeight(),
        singleGpuFirstStandardLayer->GetActivationDataCount(), false, secondStandardLayerNumNeurons, firstStandardLayerWeightsMomentum,
        firstStandardLayerWeightsDecay, firstStandardLayerWeightsLRStep, firstStandardLayerWeightsStartingLR, firstStandardLayerWeightsLRFactor,
        firstStandardLayerBiasesMomentum, firstStandardLayerBiasesDecay, firstStandardLayerBiasesLRStep, firstStandardLayerBiasesStartingLR,
        firstStandardLayerBiasesLRFactor, secondStandardLayerActivationType, 0.f, false);
    singleGpuSecondStandardLayer->AllocateBuffers(allocateTrainBuffers);
    singleGpuSecondStandardLayer->InitializeWeightsFromNormalDistribution(0.f, firstStandardLayerWeightsDeviation);
    singleGpuSecondStandardLayer->InitializeBiasesToConstant(secondStandardLayerBiasesInitialValue);
    singleGpuSecondStandardLayer->AddPrevLayer(singleGpuFirstStandardLayer);
    singleGpuFirstStandardLayer->AddNextLayer(singleGpuSecondStandardLayer);
    singleGpuSecondStandardLayerTier.push_back(singleGpuSecondStandardLayer);
    singleGpuTrainer.m_neuralNet->AddLayersTier(singleGpuSecondStandardLayerTier);

    vector<Layer*> singleGpuSoftMaxLayerTier;
    SoftMaxLayer* singleGpuSoftMaxLayer = new SoftMaxLayer(ParallelismMode::Model, singleGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        singleGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], singleGpuSecondStandardLayer->GetActivationDataSize(),
        singleGpuSecondStandardLayer->GetActivationDataCount(), false);
    singleGpuSoftMaxLayer->AllocateBuffers(allocateTrainBuffers);
    singleGpuSoftMaxLayer->AddPrevLayer(singleGpuSecondStandardLayer);
    singleGpuSecondStandardLayer->AddNextLayer(singleGpuSoftMaxLayer);
    singleGpuSoftMaxLayerTier.push_back(singleGpuSoftMaxLayer);
    singleGpuTrainer.m_neuralNet->AddLayersTier(singleGpuSoftMaxLayerTier);

    vector<Layer*> singleGpuOutputLayerTier;
    OutputLayer* singleGpuOutputLayer = new OutputLayer(singleGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        singleGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], singleGpuSoftMaxLayer->GetActivationDataSize(), inputDataCount,
        singleGpuSoftMaxLayer->GetActivationDataCount(), LossFunctionType::CrossEntropy, false, 0, 1);
    singleGpuOutputLayer->AllocateBuffers(allocateTrainBuffers);
    singleGpuOutputLayer->AddPrevLayer(singleGpuSoftMaxLayer);
    singleGpuSoftMaxLayer->AddNextLayer(singleGpuOutputLayer);
    singleGpuOutputLayerTier.push_back(singleGpuOutputLayer);
    singleGpuTrainer.m_neuralNet->AddLayersTier(singleGpuOutputLayerTier);

    // Creating random labels.
    vector<uint> labels;
    for (uint i = 0; i < inputDataCount; ++i)
    {
        labels.push_back((57 * i * i) % secondStandardLayerNumNeurons);
    }
    singleGpuOutputLayer->LoadDataLabels(labels);

    // Creating multi GPU trainer.
    Trainer multiGpuTrainer;
    multiGpuTrainer.m_neuralNet = new NeuralNet(2);

    // Dummy tier instead of input layer.
    multiGpuTrainer.m_neuralNet->AddLayersTier(vector<Layer*>());

    vector<Layer*> multiGpuFirstStandardLayerTier;
    for (int i = 0; i < 2; ++i)
    {
        StandardLayer* multiGpuFirstStandardLayer = new StandardLayer(ParallelismMode::Model, multiGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[i],
            multiGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[i], multiGpuTrainer.m_neuralNet->GetCublasHandles()[i],
            multiGpuTrainer.m_neuralNet->GetCurandStatesBuffers()[0], i, 2, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, i > 0,
            firstStandardLayerNumNeurons / 2, firstStandardLayerWeightsMomentum, firstStandardLayerWeightsDecay, firstStandardLayerWeightsLRStep,
            firstStandardLayerWeightsStartingLR, firstStandardLayerWeightsLRFactor, firstStandardLayerBiasesMomentum, firstStandardLayerBiasesDecay,
            firstStandardLayerBiasesLRStep, firstStandardLayerBiasesStartingLR, firstStandardLayerBiasesLRFactor, firstStandardLayerActivationType, 0.f, i > 0);

        multiGpuFirstStandardLayer->AllocateBuffers(allocateTrainBuffers);

        if (i == 0)
        {
            multiGpuFirstStandardLayer->CopyWeightsFromLayer(singleGpuFirstStandardLayer);
            multiGpuFirstStandardLayer->CopyBiasesFromLayer(singleGpuFirstStandardLayer);
        }
        else
        {
            CudaAssert(cudaMemcpyPeer(multiGpuFirstStandardLayer->GetWeightsBuffer(), i,
                singleGpuFirstStandardLayer->GetWeightsBuffer() + multiGpuFirstStandardLayer->GetWeightsBufferSize() / sizeof(float),
                singleGpuFirstStandardLayer->GetIndexInTier(), multiGpuFirstStandardLayer->GetWeightsBufferSize()));
            CudaAssert(cudaMemcpyPeer(multiGpuFirstStandardLayer->GetBiasesBuffer(), i,
                singleGpuFirstStandardLayer->GetBiasesBuffer() + multiGpuFirstStandardLayer->GetBiasesBufferSize() / sizeof(float),
                singleGpuFirstStandardLayer->GetIndexInTier(), multiGpuFirstStandardLayer->GetBiasesBufferSize()));
            CudaAssert(cudaDeviceSynchronize());
        }

        multiGpuFirstStandardLayer->AddPrevLayer(&mockInputLayer);
        multiGpuFirstStandardLayerTier.push_back(multiGpuFirstStandardLayer);
    }
    multiGpuTrainer.m_neuralNet->AddLayersTier(multiGpuFirstStandardLayerTier);

    vector<Layer*> multiGpuSecondStandardLayerTier;
    StandardLayer* multiGpuSecondStandardLayer = new StandardLayer(ParallelismMode::Model, multiGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        multiGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], multiGpuTrainer.m_neuralNet->GetCublasHandles()[0],
        multiGpuTrainer.m_neuralNet->GetCurandStatesBuffers()[0], 0, 1, singleGpuFirstStandardLayer->GetActivationNumChannels(),
        singleGpuFirstStandardLayer->GetActivationDataWidth(), singleGpuFirstStandardLayer->GetActivationDataHeight(),
        singleGpuFirstStandardLayer->GetActivationDataCount(), true, secondStandardLayerNumNeurons, firstStandardLayerWeightsMomentum,
        firstStandardLayerWeightsDecay, firstStandardLayerWeightsLRStep, firstStandardLayerWeightsStartingLR, firstStandardLayerWeightsLRFactor,
        firstStandardLayerBiasesMomentum, firstStandardLayerBiasesDecay, firstStandardLayerBiasesLRStep, firstStandardLayerBiasesStartingLR,
        firstStandardLayerBiasesLRFactor, secondStandardLayerActivationType, 0.f, false);
    multiGpuSecondStandardLayer->AllocateBuffers(allocateTrainBuffers);
    multiGpuSecondStandardLayer->CopyWeightsFromLayer(singleGpuSecondStandardLayer);
    multiGpuSecondStandardLayer->CopyBiasesFromLayer(singleGpuSecondStandardLayer);
    for (size_t i = 0; i < 2; ++i)
    {
        multiGpuSecondStandardLayer->AddPrevLayer(multiGpuFirstStandardLayerTier[i]);
        multiGpuFirstStandardLayerTier[i]->AddNextLayer(multiGpuSecondStandardLayer);
    }
    multiGpuSecondStandardLayerTier.push_back(multiGpuSecondStandardLayer);
    multiGpuTrainer.m_neuralNet->AddLayersTier(multiGpuSecondStandardLayerTier);

    vector<Layer*> multiGpuSoftMaxLayerTier;
    SoftMaxLayer* multiGpuSoftMaxLayer = new SoftMaxLayer(ParallelismMode::Model, multiGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        multiGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], multiGpuSecondStandardLayer->GetActivationDataSize(),
        multiGpuSecondStandardLayer->GetActivationDataCount(), false);
    multiGpuSoftMaxLayer->AllocateBuffers(allocateTrainBuffers);
    multiGpuSoftMaxLayer->AddPrevLayer(multiGpuSecondStandardLayer);
    multiGpuSecondStandardLayer->AddNextLayer(multiGpuSoftMaxLayer);
    multiGpuSoftMaxLayerTier.push_back(multiGpuSoftMaxLayer);
    multiGpuTrainer.m_neuralNet->AddLayersTier(multiGpuSoftMaxLayerTier);

    vector<Layer*> multiGpuOutputLayerTier;
    OutputLayer* multiGpuOutputLayer = new OutputLayer(multiGpuTrainer.m_neuralNet->GetDeviceCalculationStreams()[0],
        multiGpuTrainer.m_neuralNet->GetDeviceMemoryStreams()[0], multiGpuSoftMaxLayer->GetActivationDataSize(), inputDataCount,
        multiGpuSoftMaxLayer->GetActivationDataCount(), LossFunctionType::CrossEntropy, false, 0, 1);
    multiGpuOutputLayer->AllocateBuffers(allocateTrainBuffers);
    multiGpuOutputLayer->AddPrevLayer(multiGpuSoftMaxLayer);
    multiGpuSoftMaxLayer->AddNextLayer(multiGpuOutputLayer);
    multiGpuOutputLayerTier.push_back(multiGpuOutputLayer);
    multiGpuTrainer.m_neuralNet->AddLayersTier(multiGpuOutputLayerTier);

    multiGpuOutputLayer->LoadDataLabels(labels);

    // Propagate both trainers forward to check activations.
    if (!CheckStandardSingleVsMultiGpuPropagationForward(&singleGpuTrainer, &multiGpuTrainer, &mockInputLayer, singleGpuOutputLayer, multiGpuOutputLayer))
    {
        return false;
    }

    // Propagate both trainers backward.
    size_t nextTier;
    Trainer::Direction direction;
    singleGpuTrainer.PropagateBatchBackward(1, 4, nextTier, direction);
    multiGpuTrainer.PropagateBatchBackward(1, 4, nextTier, direction);
    multiGpuTrainer.PropagateBatchBackward(1, 1, nextTier, direction);
    CudaAssert(cudaDeviceSynchronize());

    // Update parameters in both networks.
    singleGpuTrainer.UpdateTiersParameters(1, 1, 4);
    multiGpuTrainer.UpdateTiersParameters(1, 1, 4);

    // Propagate both trainers forward again to check activations.
    if (!CheckStandardSingleVsMultiGpuPropagationForward(&singleGpuTrainer, &multiGpuTrainer, &mockInputLayer, singleGpuOutputLayer, multiGpuOutputLayer))
    {
        return false;
    }

    return true;
}