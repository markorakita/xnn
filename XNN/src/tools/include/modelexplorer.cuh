// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Network model explorer.
// Created: 11/16/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>

using namespace std;

class ConfigurationParser;
class NeuralNet;
class WeightsLayer;

class ModelExplorer
{
private:
    // Network configuration file path.
    string m_networkConfigurationFile;

    // Trained model file path.
    string m_trainedModelFile;

    // Is trained model file from a checkpoint.
    bool m_isModelFromCheckpoint;

    // Neural network which is explored.
    NeuralNet* m_neuralNet;

    // Neural networks configuration parser.
    ConfigurationParser* m_configurationParser;

    // Initializes neural net from configuration and trained model file.
    void InitializeNetwork();

    // Does standard analysis of buffer values.
    void AnalyzeBuffer(float* devBuffer, size_t bufferSize, float& minValue, float& avgValue, float& maxValue);

    // Explores weights layer.
    void ExploreWeightsLayer(WeightsLayer* weightsLayer);

    // Explores model, layer per layer.
    void ExploreLayers();

public:
    // Default constructor.
    ModelExplorer();

    // Destructor.
    ~ModelExplorer();

    // Parameters signatures.
    static const string c_configurationSignature;
    static const string c_trainedModelSignature;
    static const string c_isModelFromCheckpointSignature;

    // Parses arguments for explorer.
    bool ParseArguments(int argc, char* argv[]);

    // Runs model explore.
    void ExploreModel();
};