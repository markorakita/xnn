// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Network trainer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../utils/include/deftypes.cuh"

using namespace std;

class ConfigurationParser;
class Layer;
class NeuralNet;
enum class ParsingMode;
enum class PropagationMode;

class Trainer
{
private:
    friend class TestTrainer;

    enum class Direction
    {
        FORWARD,
        BACKWARD
    };

    // Neural network which is trained.
    NeuralNet* m_neuralNet;

    // Neural networks configuration parser.
    ConfigurationParser* m_configurationParser;

    // Should we load model from checkpoint and resume training from there.
    bool m_loadFromCheckpoint;

    // Default GPU device to train on.
    int m_defaultGPU;

    // Starting epoch.
    uint m_startEpoch;

    // Number of epochs to train.
    uint m_numEpochs;

    // Training data batch size.
    uint m_batchSize;

    // Network configuration file path.
    string m_networkConfigurationFile;

    // Folder with data for training.
    string m_trainDataFolder;

    // Folder with data for testing.
    string m_testDataFolder;

    // Folder to save checkpoints and progress in.
    string m_workFolder;

    // Trained model file path.
    string m_trainedModelFile;

    // Calculated loss.
    float m_loss;

    // Calculated accuracy.
    float m_accuracy;

    // Calculated multiple guess accuracy.
    float m_multipleGuessAccuracy;

    // Train data.
    vector<pair<string, uint> > m_trainData;

    // Test data.
    vector<pair<string, uint> > m_testData;

    // Synchronization for when all layers in first tier fprop.
    condition_variable m_firstTierLayersFpropSync;

    // Whether all layers in first tier fpropped.
    bool m_allFirstTierLayersFpropped;

    // Number of layers in first tier that fpropped.
    size_t m_numFirstTierLayersFpropped;

    // Mutex for changing first tier layers fpropped condition.
    mutex m_firstTierLayersFropMutex;

    // Thread for loading gradients to data parallelized layer at the transition from data to model parallelized layers;
    thread m_gradientsToDataLayerLoadThread;

    // Check if gradients are loaded to data parallelized layer at the transition from data to model parallelized layers,
    // before we write over them in backward propagation through model layers.
    bool m_gradientsToDataLayerLoaded;

    // Check if we joined the thread that loads gradients to data parallelized layer.
    bool m_joinedGradientsToDataLayerLoadThread;

    // Mutex for joining the thread that loads gradients to data parallelized layer.
    mutex m_gradientsToDataLayerLoadThreadJoinMutex;

    // Helper buffer for synchronizing weight gradients between layers in data parallel tiers.
    float* m_dataParallelTiersWeightGradientsHelperBuffer;

    // Size of the helper buffer for synchronizing weight gradients between layers in data parallel tiers.
    size_t m_dataParallelTiersWeightGradientsHelperBufferSize;

    // Helper buffer for synchronizing bias gradients between layers in data parallel tiers.
    float* m_dataParallelTiersBiasGradientsHelperBuffer;

    // Size of the helper buffer for synchronizing bias gradients between layers in data parallel tiers.
    size_t m_dataParallelTiersBiasGradientsHelperBufferSize;

    // Finds best choice for default device.
    void SetDefaultDevice();

    // Initializes network from given network configuration file.
    void InitializeNetwork(ParsingMode parsingMode);

    // Checks if configuration is valid.
    void ValidateConfiguration();

    // Calculates training memory consumption.
    void CalculateMemoryConsumption();

    // Initializes trainer parameters from given training configuration file.
    void InitializeTrainer();

    // Saves training checkpoint.
    void SaveCheckpoint(uint currEpoch, size_t dataTrainedCount, bool finalCheckpoint);

    // Loads saved checkpoint.
    void LoadCheckpoint();
    
    // Loads image data for training from certain folder;
    void LoadImageData(string folder, vector<pair<string, uint> >& data);

    // Loads text data for training from certain instances file.
    void LoadTextData(string instancesFile, vector<pair<string, uint> >& data);

    // Loads data for training.
    void LoadTrainData();

    // Loads data for testing.
    void LoadTestData();

    // Loads data batch to input layer.
    void LoadBatch(const vector<string>& dataFiles, PropagationMode propagationMode);

    // Loads gradients to layer with data parallelism, when his next layers have model parallelism.
    void LoadGradientsToLayer(Layer* layer);

    // Checks if two layers are compatible for same split.
    bool LayersCompatibleForSplit(Layer* firstLayer, Layer* secondLayer);

    // Creates layer splits for parallel propagation.
    vector<vector<Layer*> > CreateLayerSplits(size_t currTier, size_t& nextTier, int increment, function<bool(size_t)> stopCondition);

    // Does forward propagation on layers.
    void ForwardPropagateLayers(const vector<Layer*>& layers, PropagationMode propagationMode);

    // Does backward propagation on layers.
    void BackwardPropagateLayers(const vector<Layer*>& layers);

    // Propagates loaded data batch forward through the network.
    void PropagateBatchForward(size_t currTier, size_t& nextTier, PropagationMode propagationMode);

    // Propagates loaded data batch backward through the network.
    void PropagateBatchBackward(uint currEpoch, size_t currTier, size_t& nextTier, Direction& direction);

    // Prints result metrics.
    void PrintResults(uint percentDone, size_t dataCount, PropagationMode propagationMode);

    // Updates parameters of layers in specified tiers.
    void UpdateTiersParameters(uint currEpoch, size_t beginTier, size_t endTier);

    // Trains network on loaded data batch.
    void TrainBatch(uint currEpoch);

    // Trains network.
    void TrainNetwork();

    // Tests network on loaded data batch.
    void TestBatch();

    void TestBatchInPasses(vector<string>& dataFiles, vector<uint>& dataLabels, const vector<string>& nextDataFiles, const vector<uint>& nextDataLabels,
        uint numTestPasses, size_t& dataTestedCount);

    // Tests network.
    void TestNetwork();

    // Resets GPU devices used for training.
    void ResetDevices();

public:
    // Default constructor.
    Trainer();

    // Destructor.
    ~Trainer();

    // Parameters signatures.
    static const string c_configurationSignature;
    static const string c_trainDataFolderSignature;
    static const string c_testDataFolderSignature;
    static const string c_workFolderSignature;
    static const string c_numEpochsSignature;
    static const string c_batchSizeSignature;
    static const string c_loadFromCheckpointSignature;
    static const string c_defaultGpuSignature;
    static const string c_trainedModelSignature;

    // Other constants.
    static const string c_configurationFileName;
    static const string c_textDataTrainSetFileName;
    static const string c_textDataTestSetFileName;
    static const string c_resultsFileName;
    static const string c_oldCheckpointModelName;
    static const string c_checkpointModelName;
    static const string c_predictionModelName;

    // Parses arguments for training.
    bool ParseArguments(int argc, char *argv[]);

    // Runs training of network.
    void RunTraining();

    // Runs testing of network.
    void RunTesting();

    // Runs training and testing of network.
    void RunTrainingWithTesting();
};