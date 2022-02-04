// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Network model explorer.
// Created: 11/16/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/modelexplorer.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "../neuralnetwork/include/configurationparser.cuh"
#include "../neuralnetwork/include/neuralnet.cuh"
#include "../neuralnetwork/layers/include/convolutionallayer.cuh"
#include "../neuralnetwork/layers/include/standardlayer.cuh"
#include "../utils/include/consolehelper.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/utils.cuh"

const string ModelExplorer::c_configurationSignature = "-configfile";
const string ModelExplorer::c_trainedModelSignature = "-modelfile";
const string ModelExplorer::c_isModelFromCheckpointSignature = "-checkpoint";

ModelExplorer::ModelExplorer()
{
    m_networkConfigurationFile = "";
    m_trainedModelFile = "";
    m_isModelFromCheckpoint = false;

    m_neuralNet = NULL;
    m_configurationParser = new ConfigurationParser();
}

ModelExplorer::~ModelExplorer()
{
    if (m_neuralNet != NULL)
    {
        delete m_neuralNet;
    }

    delete m_configurationParser;
}

bool ModelExplorer::ParseArguments(int argc, char* argv[])
{
    if (!ParseArgument(argc, argv, c_configurationSignature, m_networkConfigurationFile) ||
        !ParseArgument(argc, argv, c_trainedModelSignature, m_trainedModelFile))
    {
        return false;
    }

    ParseArgument(argc, argv, c_isModelFromCheckpointSignature, m_isModelFromCheckpoint);

    return true;
}

void ModelExplorer::InitializeNetwork()
{
    if (m_isModelFromCheckpoint)
    {
        m_neuralNet = m_configurationParser->ParseNetworkFromConfiguration(ParsingMode::Training, m_networkConfigurationFile, "", 1, false);
        m_neuralNet->LoadModelCheckpoint(m_trainedModelFile);
    }
    else
    {
        m_neuralNet = m_configurationParser->ParseNetworkFromConfiguration(ParsingMode::Prediction, m_networkConfigurationFile, "", 1, false);
        m_neuralNet->LoadModelForPrediction(m_trainedModelFile);
    }
}

void ModelExplorer::AnalyzeBuffer(float* devBuffer, size_t bufferSize, float& minValue, float& avgValue, float& maxValue)
{
    // Copying buffer from device to host.
    float* hostBuffer;
    CudaAssert(cudaMallocHost<float>(&hostBuffer, bufferSize));
    CudaAssert(cudaMemcpy(hostBuffer, devBuffer, bufferSize, cudaMemcpyDeviceToHost));

    minValue = maxValue = hostBuffer[0];
    avgValue = 0.f;
    float cumAvgValue = hostBuffer[0];
    size_t bufferLength = bufferSize / sizeof(float);

    for (size_t i = 1; i < bufferLength; ++i)
    {
        minValue = min(minValue, hostBuffer[i]);
        maxValue = max(maxValue, hostBuffer[i]);
        cumAvgValue += hostBuffer[i];

        if (i % 1000 == 0 || i == bufferLength - 1)
        {
            avgValue += cumAvgValue / bufferLength;
            cumAvgValue = 0.f;
        }
    }

    CudaAssert(cudaFreeHost(hostBuffer));
}

void ModelExplorer::ExploreWeightsLayer(WeightsLayer* weightsLayer)
{
    ConsoleHelper::SetConsoleForeground(ConsoleColor::GRAY);

    float minWeight, avgWeight, maxWeight;
    AnalyzeBuffer(weightsLayer->GetWeightsBuffer(), weightsLayer->GetWeightsBufferSize(), minWeight, avgWeight, maxWeight);
    cout << "Min weight: " << minWeight << endl;
    cout << "Avg weight: " << avgWeight << endl;
    cout << "Max weight: " << maxWeight << endl;

    float minBias, avgBias, maxBias;
    AnalyzeBuffer(weightsLayer->GetBiasesBuffer(), weightsLayer->GetBiasesBufferSize(), minBias, avgBias, maxBias);
    cout << "Min bias: " << minBias << endl;
    cout << "Avg bias: " << avgBias << endl;
    cout << "Max bias: " << maxBias << endl;
}

void ModelExplorer::ExploreLayers()
{
    ConsoleHelper::SetConsoleForeground(ConsoleColor::CYAN);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << "    Exploring network layers:" << endl;
    cout << "**********************************************************************************************************************************" << endl << endl;

    for (const vector<Layer*>& layerTier : m_neuralNet->GetLayerTiers())
    {
        ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);

        if (layerTier[0]->GetLayerType() == LayerType::Convolutional)
        {
            cout << "Convolutional layer:" << endl;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::Standard)
        {
            cout << "Standard layer:" << endl;
        }
        else
        {
            continue;
        }

        ExploreWeightsLayer(static_cast<WeightsLayer*>(layerTier[0]));
        cout << endl;
    }

    ConsoleHelper::SetConsoleForeground(ConsoleColor::CYAN);
    cout << "**********************************************************************************************************************************" << endl << endl;
}

void ModelExplorer::ExploreModel()
{
    InitializeNetwork();
    ExploreLayers();
}