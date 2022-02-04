// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Network trainer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/trainer.cuh"

#include <direct.h>
#include <fstream>
#include <iostream>

#include "include/datamaker.cuh"
#include "include/modelconverter.cuh"
#include "../neuralnetwork/include/configurationparser.cuh"
#include "../neuralnetwork/include/matrixoperations.cuh"
#include "../neuralnetwork/include/neuralnet.cuh"
#include "../neuralnetwork/layers/include/inputlayer.cuh"
#include "../neuralnetwork/layers/include/outputlayer.cuh"
#include "../neuralnetwork/layers/include/weightslayer.cuh"
#include "../utils/include/asserts.cuh"
#include "../utils/include/config.cuh"
#include "../utils/include/consolehelper.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/cudahelper.cuh"
#include "../utils/include/utils.cuh"

const string Trainer::c_configurationSignature = "-configfile";
const string Trainer::c_trainDataFolderSignature = "-traindata";
const string Trainer::c_testDataFolderSignature = "-testdata";
const string Trainer::c_workFolderSignature = "-workfolder";
const string Trainer::c_numEpochsSignature = "-numepochs";
const string Trainer::c_batchSizeSignature = "-batchsize";
const string Trainer::c_loadFromCheckpointSignature = "-continue";
const string Trainer::c_defaultGpuSignature = "-gpu";
const string Trainer::c_trainedModelSignature = "-modelfile";

const string Trainer::c_configurationFileName = "configuration.xnn";
const string Trainer::c_textDataTrainSetFileName = "trainSet.txt";
const string Trainer::c_textDataTestSetFileName = "testSet.txt";
const string Trainer::c_resultsFileName = "results.txt";
const string Trainer::c_oldCheckpointModelName = "old_model_checkpoint.xnnm";
const string Trainer::c_checkpointModelName = "model_checkpoint.xnnm";
const string Trainer::c_predictionModelName = "model.xnnm";

Trainer::Trainer()
{
    m_neuralNet = NULL;
    m_defaultGPU = -1;
    m_numFirstTierLayersFpropped = 0;
    m_gradientsToDataLayerLoaded = true;
    m_joinedGradientsToDataLayerLoadThread = true;

    m_dataParallelTiersWeightGradientsHelperBuffer = NULL;
    m_dataParallelTiersWeightGradientsHelperBufferSize = 0;
    m_dataParallelTiersBiasGradientsHelperBuffer = NULL;
    m_dataParallelTiersBiasGradientsHelperBufferSize = 0;

    m_trainedModelFile = "";
    m_numEpochs = 0;

    m_configurationParser = new ConfigurationParser();
}

bool Trainer::ParseArguments(int argc, char *argv[])
{
    ParseArgument(argc, argv, c_loadFromCheckpointSignature, m_loadFromCheckpoint);
    ParseArgument(argc, argv, c_trainDataFolderSignature, m_trainDataFolder);
    ParseArgument(argc, argv, c_testDataFolderSignature, m_testDataFolder);
    ParseArgument(argc, argv, c_trainedModelSignature, m_trainedModelFile);
    ParseArgument(argc, argv, c_numEpochsSignature, m_numEpochs);

    if ((m_trainDataFolder == "" && m_testDataFolder == "") ||
        !ParseArgument(argc, argv, c_workFolderSignature, m_workFolder) ||
        !ParseArgument(argc, argv, c_batchSizeSignature, m_batchSize))
    {
        return false;
    }

    if (m_loadFromCheckpoint)
    {
        m_networkConfigurationFile = m_workFolder + "\\" + c_configurationFileName;
    }
    else if (!ParseArgument(argc, argv, c_configurationSignature, m_networkConfigurationFile))
    {
        return false;
    }

    int defaultGpu;
    if (ParseArgument(argc, argv, c_defaultGpuSignature, defaultGpu))
    {
        if (defaultGpu < 0)
        {
            return false;
        }
        m_defaultGPU = defaultGpu;
    }

    return true;
}

Trainer::~Trainer()
{
    if (m_neuralNet != NULL)
    {
        delete m_neuralNet;
    }

    delete m_configurationParser;

    if (m_dataParallelTiersWeightGradientsHelperBuffer != NULL)
    {
        CudaAssert(cudaFree(m_dataParallelTiersWeightGradientsHelperBuffer));
        CudaAssert(cudaFree(m_dataParallelTiersBiasGradientsHelperBuffer));
    }
}

void Trainer::SetDefaultDevice()
{
    int numDevices;
    CudaAssert(cudaGetDeviceCount(&numDevices));

    if (m_defaultGPU < 0)
    {
        // TODO: This would require to change whole logic of tiers, currently we assume everywhere that index in tier is
        // equal to index of GPU used. Make this work one day when you find time...

        //// It hasn't been set explicitely with run parameters, so we will find one with most amount of free memory.
        //size_t maxFreeMemorySize = 0;
        //// By default it is zero.
        //int maxFreeMemoryDevice = 0;
        //for (int currDevice = 0; currDevice < numDevices; ++currDevice)
        //{
        //	CudaAssert(cudaSetDevice(currDevice));
        //	size_t freeMemorySize = GetSizeOfAvailableGpuMemory();
        //	if (freeMemorySize > maxFreeMemorySize)
        //	{
        //		maxFreeMemorySize = freeMemorySize;
        //		maxFreeMemoryDevice = currDevice;
        //	}
        //}
        //m_defaultGPU = maxFreeMemoryDevice;

        m_defaultGPU = 0;
    }
    else
    {
        ShipAssert(m_defaultGPU < numDevices, "Default GPU can't be set since there are not that many GPUs in the system!");
    }

    CudaAssert(cudaSetDevice(m_defaultGPU));
}

void Trainer::InitializeNetwork(ParsingMode parsingMode)
{
    if (m_neuralNet != NULL)
    {
        delete m_neuralNet;
    }

    if (parsingMode == ParsingMode::Training)
    {
        m_neuralNet = m_configurationParser->ParseNetworkFromConfiguration(parsingMode, m_networkConfigurationFile, m_trainDataFolder, m_batchSize, !m_loadFromCheckpoint);
        if (m_loadFromCheckpoint)
        {
            LoadCheckpoint();
        }
    }
    else
    {
        m_neuralNet = m_configurationParser->ParseNetworkFromConfiguration(parsingMode, m_networkConfigurationFile, m_testDataFolder, m_batchSize, false);
        m_neuralNet->LoadModelForPrediction(m_trainedModelFile == "" ? m_workFolder + "\\" + c_predictionModelName : m_trainedModelFile);
    }

    if (!m_loadFromCheckpoint)
    {
        // Copy configuration over to work folder.
        ifstream inputConfigFile(m_networkConfigurationFile);
        ofstream outputConfigFile(m_workFolder + "\\" + c_configurationFileName);
        string line;
        while (getline(inputConfigFile, line))
        {
            outputConfigFile << line << endl;
        }
        outputConfigFile.close();
        inputConfigFile.close();

        m_startEpoch = 1;
    }
}

void Trainer::ValidateConfiguration()
{
    ShipAssert(m_neuralNet->GetLayerTiers().size() > 2 && m_neuralNet->GetLayerTiers()[0][0]->GetLayerType() == LayerType::Input &&
        m_neuralNet->GetLayerTiers().back()[0]->GetLayerType() == LayerType::Output,
        "Invalid network configuration: each network needs to have at least input and output layers, plus one layer in between!");

    ShipAssert(m_neuralNet->GetMaxNetworkTierSize() <= Config::NUM_GPUS, "There are not enough GPUs in the system to train this network!");

    // TODO: Finish.
}

void Trainer::InitializeTrainer()
{
    // Setup helper buffers for synchronizing weight and bias gradients between layers in data parallel tiers.
    const vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
    for (size_t currTier = 1; currTier < layerTiers.size() - 1; ++currTier)
    {
        if (layerTiers[currTier][0]->GetParallelismMode() == ParallelismMode::Data)
        {
            if (layerTiers[currTier][0]->GetLayerType() == LayerType::Convolutional || layerTiers[currTier][0]->GetLayerType() == LayerType::Standard)
            {
                WeightsLayer* weightsLayer = static_cast<WeightsLayer*>(layerTiers[currTier][0]);
                m_dataParallelTiersWeightGradientsHelperBufferSize = max(m_dataParallelTiersWeightGradientsHelperBufferSize, weightsLayer->GetWeightsBufferSize());
                m_dataParallelTiersBiasGradientsHelperBufferSize = max(m_dataParallelTiersBiasGradientsHelperBufferSize, weightsLayer->GetBiasesBufferSize());
            }
        }
    }

    if (m_dataParallelTiersWeightGradientsHelperBufferSize > 0)
    {
        CudaAssert(cudaMalloc<float>(&m_dataParallelTiersWeightGradientsHelperBuffer, m_dataParallelTiersWeightGradientsHelperBufferSize));
        CudaAssert(cudaMalloc<float>(&m_dataParallelTiersBiasGradientsHelperBuffer, m_dataParallelTiersBiasGradientsHelperBufferSize));
    }
}

void Trainer::CalculateMemoryConsumption()
{
    ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << "    Training memory consumption:" << endl;
    cout << "**********************************************************************************************************************************" << endl << endl;

    int convLayerIndex = 0;
    int standardLayerIndex = 0;
    size_t trainedModelSize = 0;
    size_t totalMemoryConsumption = 0;

    for (const vector<Layer*>& layerTier : m_neuralNet->GetLayerTiers())
    {
        WeightsLayer* weightsLayer = nullptr;
        ConsoleHelper::SetConsoleForeground(ConsoleColor::CYAN);
        if (layerTier[0]->GetLayerType() == LayerType::Convolutional)
        {
            cout << "Convolutional layer " << ++convLayerIndex << ":" << endl;
            weightsLayer = static_cast<WeightsLayer*>(layerTier[0]);
        }
        else if (layerTier[0]->GetLayerType() == LayerType::Standard)
        {
            cout << "Standard layer " << ++standardLayerIndex << ":" << endl;
            weightsLayer = static_cast<WeightsLayer*>(layerTier[0]);
        }
        else if (layerTier[0]->GetLayerType() == LayerType::Dropout)
        {
            cout << "Dropout layer:" << endl;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::MaxPool)
        {
            cout << "MaxPool layer:" << endl;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::ResponseNormalization)
        {
            cout << "Response Normalization layer:" << endl;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::SoftMax)
        {
            cout << "SoftMax layer:" << endl;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::Input)
        {
            cout << "Input layer:" << endl;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::Output)
        {
            cout << "Output layer:" << endl;
        }
        else
        {
            continue;
        }

        ConsoleHelper::SetConsoleForeground(ConsoleColor::GRAY);
        if (weightsLayer == nullptr)
        {
            cout << "Memory consumption: " << (float)layerTier[0]->GetMemoryConsumptionSize() / (1024 * 1024) << "MB" << endl << endl;
        }
        else
        {
            cout << "Weights buffer size: " << (float)weightsLayer->GetWeightsBufferSize() / (1024 * 1024) << "MB" << endl;
            cout << "Biases buffer size: " << (float)weightsLayer->GetBiasesBufferSize() / (1024 * 1024) << "MB" << endl;
            cout << "Other training buffers size: " << (float)(weightsLayer->GetMemoryConsumptionSize() -
                weightsLayer->GetWeightsBufferSize() - weightsLayer->GetBiasesBufferSize()) / (1024 * 1024) << "MB" << endl << endl;

            trainedModelSize += (weightsLayer->GetWeightsBufferSize() + weightsLayer->GetBiasesBufferSize()) *
                (weightsLayer->GetParallelismMode() == ParallelismMode::Model ? layerTier.size() : 1);
        }

        totalMemoryConsumption += layerTier[0]->GetMemoryConsumptionSize();
    }

    totalMemoryConsumption += m_dataParallelTiersWeightGradientsHelperBufferSize + m_dataParallelTiersBiasGradientsHelperBufferSize;

    ConsoleHelper::SetConsoleForeground(ConsoleColor::DARKCYAN);
    cout << endl << "Total training memory consumption: " << (float)totalMemoryConsumption / (1024 * 1024) << "MB" << endl << endl;
    cout << "Trained model size: " << (float)trainedModelSize / (1024 * 1024) << "MB" << endl << endl;

    ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl << endl;
}

void Trainer::SaveCheckpoint(uint currEpoch, size_t dataTrainedCount, bool finalCheckpoint)
{
    // Saving model.
    if (currEpoch > 2)
    {
        ShipAssert(remove((m_workFolder + "\\" + c_oldCheckpointModelName).c_str()) == 0, "");
    }
    if (currEpoch > 1)
    {
        ShipAssert(rename((m_workFolder + "\\" + c_checkpointModelName).c_str(), (m_workFolder + "\\" + c_oldCheckpointModelName).c_str()) == 0, "");
    }
    m_neuralNet->SaveModelCheckpoint(m_workFolder + "\\" + c_checkpointModelName);
    if (finalCheckpoint)
    {
        if (currEpoch > 1)
        {
            remove((m_workFolder + "\\" + c_oldCheckpointModelName).c_str());
        }

        string modelPath = m_workFolder + "\\" + c_predictionModelName;
        m_neuralNet->SaveModelForPrediction(modelPath);
    }

    // Saving results.
    ofstream resultsFile(m_workFolder + "\\" + c_resultsFileName);
    resultsFile << "Trained epochs: " << currEpoch << endl;
    resultsFile << "Batch size: " << m_batchSize << endl << endl;
    resultsFile << "Training results:    Loss: " << m_loss / dataTrainedCount << "  Accuracy: " << m_accuracy / dataTrainedCount;
    if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
    {
        resultsFile << "  Multiple guess accuracy: " << m_multipleGuessAccuracy / dataTrainedCount;
    }
    resultsFile << endl << endl;
}

void Trainer::LoadCheckpoint()
{
    // Loading model.
    m_neuralNet->LoadModelCheckpoint(m_workFolder + "\\" + c_checkpointModelName);

    // Parse epoch.
    ifstream resultsFile(m_workFolder + "\\" + c_resultsFileName);
    string line;
    getline(resultsFile, line);
    size_t valuePosition = line.find_last_of(":");
    string epochValue = line.substr(valuePosition + 2);
    m_startEpoch = stoul(epochValue) + 1;
    // Parse results.
    getline(resultsFile, line);
    getline(resultsFile, line);
    getline(resultsFile, line);
    line = line.substr(line.find_first_of(":") + 1);
    line = line.substr(line.find_first_of(":") + 2);
    size_t accuracyPosition = line.find_first_of(":");
    string lossValue = line.substr(0, accuracyPosition - string("Accuracy:").length() - 1);
    m_loss = stof(lossValue);
    line = line.substr(accuracyPosition + 2);
    if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
    {
        size_t mulAccuracyPosition = line.find_first_of(":");
        string accuracyValue = line.substr(0, mulAccuracyPosition - string("Multiple guess accuracy:").length() - 1);
        m_accuracy = stof(accuracyValue);
        line = line.substr(mulAccuracyPosition + 2);
        m_multipleGuessAccuracy = stof(line);
    }
    else
    {
        line = line.substr(accuracyPosition + 2);
        m_accuracy = stof(line);
    }
}

void Trainer::LoadImageData(string folder, vector<pair<string, uint> >& data)
{
    ifstream labelsFile(folder + "\\" + DataMaker::c_labelsFileName);
    string imageName;
    uint label;
    while (labelsFile >> imageName >> label)
    {
        // TODO: You should expect zero based label here, same as for text data,
        //       and fix datamaker to create labels file for alexnet with zero based labels.
        data.push_back(make_pair(imageName, label - 1));
    }
}

void Trainer::LoadTextData(string instancesFile, vector<pair<string, uint> >& data)
{
    ifstream labelsFile(instancesFile);
    string instance;
    string features;
    string label;
    while (getline(labelsFile, instance))
    {
        size_t split = instance.find_first_of(' ');
        label = instance.substr(0, split);
        features = instance.substr(split + 1);
        data.push_back(make_pair(features, stoul(label)));
    }
}

void Trainer::LoadTrainData()
{
    InputLayer* inputLayer = m_neuralNet->GetInputLayer();
    if (inputLayer->GetDataType() == DataType::Image)
    {
        LoadImageData(m_trainDataFolder, m_trainData);
    }
    else
    {
        LoadTextData(m_trainDataFolder + "\\" + c_textDataTrainSetFileName, m_trainData);
    }

    // This is technically not required for training to work, but I don't see a point in running training with so few data.
    ShipAssert(m_trainData.size() >= m_neuralNet->GetInputLayer()->GetInputDataCount(), "There is not enough train data!");

    random_shuffle(m_trainData.begin(), m_trainData.end());
}

void Trainer::LoadTestData()
{
    InputLayer* inputLayer = m_neuralNet->GetInputLayer();
    if (inputLayer->GetDataType() == DataType::Image)
    {
        LoadImageData(m_testDataFolder, m_testData);
    }
    else
    {
        LoadTextData(m_testDataFolder + "\\" + c_textDataTestSetFileName, m_testData);
    }

    ShipAssert(m_testData.size() > 0, "There is not enough test data!");
}

void Trainer::LoadBatch(const vector<string>& dataFiles, PropagationMode propagationMode)
{
    InputLayer* inputLayer = m_neuralNet->GetInputLayer();

    // TODO: we should pass some parameter so that we do not call this every time, since we are passing the same dataFiles for numTestPasses times
    inputLayer->SetDataFilesToLoad(dataFiles, propagationMode);
    // Load data from disk to host memory.
    inputLayer->LoadInputs();

    // Wait for first layer to fprop.
    if (!m_allFirstTierLayersFpropped)
    {
        unique_lock<mutex> lock(m_firstTierLayersFropMutex);
        while (!m_allFirstTierLayersFpropped)
        {
            m_firstTierLayersFpropSync.wait(lock);
        }
    }

    // Load data from host to GPU memory.
    inputLayer->DoForwardProp(propagationMode);
}

void Trainer::LoadGradientsToLayer(Layer* layer)
{
    CudaAssert(cudaSetDevice(layer->GetIndexInTier()));
    layer->LoadActivationGradients();
    layer->SynchronizeMemoryOperations();

    m_gradientsToDataLayerLoaded = true;
}

void Trainer::UpdateTiersParameters(uint currEpoch, size_t beginTier, size_t endTier)
{
    const vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

    for (size_t currTier = beginTier; currTier < endTier; ++currTier)
    {
        if (layerTiers[currTier][0]->GetLayerType() != LayerType::Convolutional && layerTiers[currTier][0]->GetLayerType() != LayerType::Standard)
        {
            continue;
        }

        // In tiers with data parallelism we need to sync gradients between layers.
        if (layerTiers[currTier][0]->GetParallelismMode() == ParallelismMode::Data)
        {
            // Casting layers to appropriate type to determine buffers to work on.
            vector<float*> weightsGradientsBuffers, biasesGradientsBuffers;
            size_t weightsGradientsBufferSize = 0, biasesGradientsBufferSize = 0;
            for (size_t layerIndex = 0; layerIndex < layerTiers[currTier].size(); ++layerIndex)
            {
                WeightsLayer* weightsLayer = static_cast<WeightsLayer*>(layerTiers[currTier][layerIndex]);
                if (layerIndex == 0)
                {
                    weightsGradientsBufferSize = weightsLayer->GetWeightsBufferSize();
                    biasesGradientsBufferSize = weightsLayer->GetBiasesBufferSize();
                }
                weightsGradientsBuffers.push_back(weightsLayer->GetWeightsGradientsBuffer());
                biasesGradientsBuffers.push_back(weightsLayer->GetBiasesGradientsBuffer());
            }

            // Sum up weights and biases gradients from all layers into first layer's gradients buffers.
            for (int layerIndex = 1; layerIndex < layerTiers[currTier].size(); ++layerIndex)
            {
                // Copy over weights gradients buffer.
                CudaAssert(cudaMemcpyPeer(m_dataParallelTiersWeightGradientsHelperBuffer, 0, weightsGradientsBuffers[layerIndex],
                    layerIndex, weightsGradientsBufferSize));

                // Add them to first layer's weights gradients buffer.
                CalculateElementWiseSum(weightsGradientsBuffers[0], m_dataParallelTiersWeightGradientsHelperBuffer,
                    (uint)(weightsGradientsBufferSize / sizeof(float)), weightsGradientsBuffers[0], 0);

                // Copy over biases gradients buffer.
                CudaAssert(cudaMemcpyPeer(m_dataParallelTiersBiasGradientsHelperBuffer, 0, biasesGradientsBuffers[layerIndex],
                    layerIndex, biasesGradientsBufferSize));

                // Add them to first layer's biases gradients buffer.
                CalculateElementWiseSum(biasesGradientsBuffers[0], m_dataParallelTiersBiasGradientsHelperBuffer,
                    (uint)(biasesGradientsBufferSize / sizeof(float)), biasesGradientsBuffers[0], 0);
            }

            CudaAssert(cudaStreamSynchronize(0));

            layerTiers[currTier][0]->UpdateLayerParameters((float)currEpoch / m_numEpochs);

            // Copy updated parameters from first layer to other layers.
            for (int layerIndex = 1; layerIndex < layerTiers[currTier].size(); ++layerIndex)
            {
                WeightsLayer* weightsLayer1 = static_cast<WeightsLayer*>(layerTiers[currTier][0]);
                WeightsLayer* weightsLayer2 = static_cast<WeightsLayer*>(layerTiers[currTier][layerIndex]);

                // Copy over weights buffer.
                CudaAssert(cudaMemcpyPeer(weightsLayer2->GetWeightsBuffer(), layerIndex, weightsLayer1->GetWeightsBuffer(),
                    0, weightsLayer1->GetWeightsBufferSize()));

                // Copy over biases buffer.
                CudaAssert(cudaMemcpyPeer(weightsLayer2->GetBiasesBuffer(), layerIndex, weightsLayer1->GetBiasesBuffer(),
                    0, weightsLayer1->GetBiasesBufferSize()));
            }
        }
        else
        {
            // Update each layer's parameters.
            vector<thread> updateThreads;
            for (Layer* layer : layerTiers[currTier])
            {
                updateThreads.push_back(thread([this, layer, currEpoch]
                    {
                        CudaAssert(cudaSetDevice(layer->GetIndexInTier()));
                        layer->UpdateLayerParameters((float)currEpoch / m_numEpochs);
                    }));
            }
            for (size_t updateThreadIndex = 0; updateThreadIndex < updateThreads.size(); ++updateThreadIndex)
            {
                updateThreads[updateThreadIndex].join();
            }
        }
    }
}

bool Trainer::LayersCompatibleForSplit(Layer* firstLayer, Layer* secondLayer)
{
    return firstLayer->GetParallelismMode() == secondLayer->GetParallelismMode() &&
        firstLayer->GetNextLayers().size() == 1 && secondLayer->GetPrevLayers().size() == 1;
}

void Trainer::ForwardPropagateLayers(const vector<Layer*>& layers, PropagationMode propagationMode)
{
    uint splitIndexInTier = layers[0]->GetIndexInTier();
    CudaAssert(cudaSetDevice(splitIndexInTier));

    Layer* prevSplitLastLayer = layers[0]->GetPrevLayers().size() <= splitIndexInTier ? NULL : layers[0]->GetPrevLayers()[splitIndexInTier];
    Layer* prevTierFirstLayer = layers[0]->GetPrevLayers()[0];

    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex)
    {
        Layer* layer = layers[layerIndex];

        // Check if we should start preloading inputs for next propagation of layers with model parallelism connected with
        // layers with data parallelism, in parallel with computation in these layers.
        if (layerIndex == 0 && layer->GetParallelismMode() == ParallelismMode::Model &&
            prevSplitLastLayer != NULL && prevSplitLastLayer->GetParallelismMode() == ParallelismMode::Model)
        {
            Layer* prevSplitFirstLayer = prevSplitLastLayer;
            while (prevSplitFirstLayer->GetLayerType() != LayerType::Input && !prevSplitFirstLayer->GetPrevLayers().empty() &&
                LayersCompatibleForSplit(prevSplitFirstLayer->GetPrevLayers()[0], prevSplitFirstLayer))
            {
                prevSplitFirstLayer = prevSplitFirstLayer->GetPrevLayers()[0];
            }

            if (prevSplitFirstLayer->GetLayerType() != LayerType::Input && !prevSplitFirstLayer->GetPrevLayers().empty() &&
                prevSplitFirstLayer->GetPrevLayers()[0]->GetParallelismMode() == ParallelismMode::Data &&
                prevSplitFirstLayer->GetInputLayerIndexInTier() < (int)layer->GetPrevLayers().size() - 1)
            {
                // Start preloading inputs for next propagation through this layer.
                prevSplitFirstLayer->SynchronizeCalculations();
                prevSplitFirstLayer->LoadInputs();
            }
        }

        // Load input if needed.
        if (layer->GetInputLayerIndexInTier() < 0)
        {
            layer->LoadInputs();

            // No need for sync with input layer, since it's propagation is worked on in different thread
            // which is joined before computation thread starts.
            if (layer->HoldsInputData())
            {
                // Making sure inputs are loaded before computation.
                layer->SynchronizeMemoryOperations();
            }
        }
        else
        {
            // Skipping load of inputs if they are preloaded, but need to sync to make sure preloading is finished.
            if (layer->GetInputLayerIndexInTier() != layer->GetIndexInTier())
            {
                layer->SynchronizeMemoryOperations();
            }
        }

        if (layer->GetParallelismMode() == ParallelismMode::Model && prevTierFirstLayer->GetParallelismMode() == ParallelismMode::Data)
        {
            layer->IncreaseInputLayerIndexInTier();
        }

        // Do forward propagation.
        layer->DoForwardProp(propagationMode);

        // Check if we should start preloading input data for next batch propagation.
        if (layerIndex == 0 && prevTierFirstLayer->GetLayerType() == LayerType::Input)
        {
            layer->SynchronizeCalculations();

            lock_guard<mutex> lock(m_firstTierLayersFropMutex);
            ++m_numFirstTierLayersFpropped;
            if (m_numFirstTierLayersFpropped == prevTierFirstLayer->GetNextLayers().size())
            {
                // Notify that all first tier layers fpropped.
                m_allFirstTierLayersFpropped = true;
                m_numFirstTierLayersFpropped = 0;
                m_firstTierLayersFpropSync.notify_one();
            }
        }

        // Gathering batch results if this is output layer.
        if (propagationMode == PropagationMode::Train && layer->GetLayerType() == LayerType::Output)
        {
            layer->SynchronizeCalculations();

            OutputLayer* outputLayer = static_cast<OutputLayer*>(layer);
            // No need for locking here since there can be only one output layer.
            m_loss += outputLayer->GetLoss();
            m_accuracy += outputLayer->GetAccuracy();
            if (outputLayer->ShouldCalculateMultipleGuessAccuracy())
            {
                m_multipleGuessAccuracy += outputLayer->GetMultipleGuessAccuracy();
            }
        }
    }

    // Making sure calculations are finished before moving on to next tier.
    layers.back()->SynchronizeCalculations();
}

void Trainer::BackwardPropagateLayers(const vector<Layer*>& layers)
{
    CudaAssert(cudaSetDevice(layers[0]->GetIndexInTier()));

    // Layers are sorted from last to first, imagine everything in this method written as if we are going backwards through the network.
    Layer* prevSplitFirstLayer = layers[0]->GetNextLayers().empty() ? NULL : layers[0]->GetNextLayers()[0];

    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex)
    {
        Layer* layer = layers[layerIndex];

        // Skipping load of activation gradients if they are preloaded.
        if (!(layerIndex == 0 && layer->GetParallelismMode() == ParallelismMode::Data && prevSplitFirstLayer != NULL &&
            prevSplitFirstLayer->GetParallelismMode() == ParallelismMode::Model && layer->GetIndexInTier() < layer->GetTierSize() - 1))
        {
            layer->LoadActivationGradients();

            if (layer->HoldsActivationGradients())
            {
                // Making sure activation gradients are loaded before computation.
                layer->SynchronizeMemoryOperations();
            }
        }

        // We need to delay doing backward prop on model parallelized layer which is currently sending gradients to data parallelized layer before it.
        if (layerIndex == layers.size() - 1 && layer->GetParallelismMode() == ParallelismMode::Model &&
            layer->GetPrevLayers()[0]->GetParallelismMode() == ParallelismMode::Data && layer->GetInputLayerIndexInTier() > 0 && !m_gradientsToDataLayerLoaded)
        {
            lock_guard<mutex> lock(m_gradientsToDataLayerLoadThreadJoinMutex);
            if (!m_gradientsToDataLayerLoaded)
            {
                m_gradientsToDataLayerLoadThread.join();
                m_joinedGradientsToDataLayerLoadThread = true;
            }
        }

        // Do backward propagation.
        layer->DoBackwardProp();
    }

    // Making sure calculations are finished before moving on to next tier.
    layers.back()->SynchronizeCalculations();
}

vector<vector<Layer*> > Trainer::CreateLayerSplits(size_t currTier, size_t& nextTier, int increment, function<bool(size_t)> stopCondition)
{
    const vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

    // Split is group of layers from adjacent tiers that can work together in one pass.
    // If those adjacent tiers contain multiple layers, than we have multiple splits to work on in parallel during one pass.
    vector<vector<Layer*> > layerSplits;
    for (size_t split = 0; split < layerTiers[currTier].size(); ++split)
    {
        vector<Layer*> layerSplit;
        layerSplit.push_back(layerTiers[currTier][split]);
        layerSplits.push_back(layerSplit);
    }
    for (nextTier = currTier + increment; stopCondition(nextTier); nextTier += increment)
    {
        bool layersCompatible = currTier < nextTier ? LayersCompatibleForSplit(layerTiers[currTier][0], layerTiers[nextTier][0]) :
            LayersCompatibleForSplit(layerTiers[nextTier][0], layerTiers[currTier][0]);

        if (layerTiers[currTier].size() == layerTiers[nextTier].size() && layersCompatible)
        {
            for (size_t split = 0; split < layerSplits.size(); ++split)
            {
                layerSplits[split].push_back(layerTiers[nextTier][split]);
            }
        }
        else
        {
            break;
        }
    }

    return layerSplits;
}

void Trainer::PropagateBatchForward(size_t currTier, size_t& nextTier, PropagationMode propagationMode)
{
    size_t layerTiersCount = m_neuralNet->GetLayerTiers().size();

    // Make splits.
    vector<vector<Layer*> > layerSplits = CreateLayerSplits(currTier, nextTier, 1, [layerTiersCount](size_t tier) { return tier < layerTiersCount; });

    // Propagate on split.
    vector<thread> propagationThreads;
    for (size_t split = 0; split < layerSplits.size(); ++split)
    {
        propagationThreads.push_back(thread(&Trainer::ForwardPropagateLayers, this, ref(layerSplits[split]), propagationMode));
    }
    for (size_t split = 0; split < layerSplits.size(); ++split)
    {
        propagationThreads[split].join();
    }
}

void Trainer::PropagateBatchBackward(uint currEpoch, size_t currTier, size_t& nextTier, Direction& direction)
{
    const vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

    // In case where layers with data parallelism are input to layers with model parallelism,
    // we need to propagate multiple times through layers with model parallelism.
    if (currTier < layerTiers.size() - 1 && layerTiers[currTier][0]->GetParallelismMode() == ParallelismMode::Data &&
        layerTiers[currTier + 1][0]->GetParallelismMode() == ParallelismMode::Model)
    {
        if (layerTiers[currTier + 1][0]->GetInputLayerIndexInTier() < (int)layerTiers[currTier].size() - 1)
        {
            // We have to propagate again through previous tiers since not all data has been propagated through them.
            nextTier = currTier + 1;
            direction = Direction::FORWARD;
            m_neuralNet->GetOutputLayer()->MoveLabelsOffset();

            // Start loading gradients into layer whose activities we propagated on in model layers.
            Layer* propagatedLayer = layerTiers[currTier][(size_t)layerTiers[currTier + 1][0]->GetInputLayerIndexInTier()];
            if (m_gradientsToDataLayerLoadThread.joinable() && !m_joinedGradientsToDataLayerLoadThread)
            {
                m_gradientsToDataLayerLoadThread.join();
            }
            m_joinedGradientsToDataLayerLoadThread = false;
            m_gradientsToDataLayerLoaded = false;
            m_gradientsToDataLayerLoadThread = thread(&Trainer::LoadGradientsToLayer, this, propagatedLayer);

            // Update parameters in previous tiers.
            UpdateTiersParameters(currEpoch, nextTier, layerTiers.size() - 1);

            return;
        }
        else
        {
            if (!m_joinedGradientsToDataLayerLoadThread)
            {
                m_gradientsToDataLayerLoadThread.join();
                m_joinedGradientsToDataLayerLoadThread = true;
            }

            // All data has been propagated through previous tiers, clearing their track record.
            for (Layer* layer : layerTiers[currTier + 1])
            {
                layer->ResetInputLayerIndexInTier();
            }
        }
    }

    // Make splits.
    vector<vector<Layer*> > layerSplits = CreateLayerSplits(currTier, nextTier, -1, [](size_t tier) { return tier > 0; });

    // Propagate on split.
    vector<thread> propagationThreads;
    for (size_t split = 0; split < layerSplits.size(); ++split)
    {
        propagationThreads.push_back(thread(&Trainer::BackwardPropagateLayers, this, ref(layerSplits[split])));
    }
    for (size_t split = 0; split < layerSplits.size(); ++split)
    {
        propagationThreads[split].join();
    }
}

void Trainer::TrainBatch(uint currEpoch)
{
    const vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
    Direction direction = Direction::FORWARD;
    size_t currTier = 1;
    while (currTier != 0)
    {
        size_t nextTier;

        // Propagate.
        if (direction == Direction::FORWARD)
        {
            PropagateBatchForward(currTier, nextTier, PropagationMode::Train);
        }
        else
        {
            PropagateBatchBackward(currEpoch, currTier, nextTier, direction);
        }

        // Move on to next tier.
        if (nextTier == layerTiers.size())
        {
            currTier = layerTiers.size() - 1;
            direction = Direction::BACKWARD;
        }
        else
        {
            currTier = nextTier;
        }
    }

    // Update parameters in all tiers.
    UpdateTiersParameters(currEpoch, 0, layerTiers.size() - 1);
}

void Trainer::PrintResults(uint percentDone, size_t dataCount, PropagationMode propagationMode)
{
    ConsoleHelper::SetConsoleForeground(percentDone < 100 ? ConsoleColor::GRAY : ConsoleColor::GREEN);

    cout << percentDone << "% done, results:   ";
    if (percentDone < 100)
    {
        cout << " ";
    }
    if (propagationMode == PropagationMode::Train)
    {
        cout << "Loss: " << m_loss / dataCount << "  ";
    }
    cout << "Accuracy: " << m_accuracy / dataCount;
    if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
    {
        cout << "  Multiple guess accuracy: " << m_multipleGuessAccuracy / dataCount;
    }
    if (percentDone == 100)
    {
        cout << "    [" << GetCurrentTimeStamp() << "]";
    }
    cout << endl;
}

void Trainer::TrainNetwork()
{
    ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl;
    if (m_loadFromCheckpoint)
    {
        cout << "    Network training from checkpoint started  [" << GetCurrentTimeStamp() << "]" << endl;
        cout << "    Last results:   Loss: " << m_loss << "  Accuracy: " << m_accuracy;
        if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
        {
            cout << "  Multiple guess accuracy: " << m_multipleGuessAccuracy;
        }
        cout << endl;
    }
    else
    {
        cout << "    Network training started  [" << GetCurrentTimeStamp() << "]" << endl;
    }
    cout << "**********************************************************************************************************************************" << endl;

    vector<string> dataFiles;
    vector<uint> dataLabels;
    vector<string> nextDataFiles;
    vector<uint> nextDataLabels;
    thread trainThread;
    InputLayer* inputLayer = m_neuralNet->GetInputLayer();
    OutputLayer* outputLayer = m_neuralNet->GetOutputLayer();
    size_t inputBatchSize = m_neuralNet->GetInputLayer()->GetInputDataCount();

    for (uint currEpoch = m_startEpoch; currEpoch <= m_numEpochs; ++currEpoch)
    {
        ConsoleHelper::SetConsoleForeground(ConsoleColor::DARKCYAN);
        cout << endl << "Training epoch " << currEpoch << endl;
        cout << "------------------------------------------------------------------------------------------------" << endl;

        m_loss = 0.f;
        m_accuracy = 0.f;
        m_multipleGuessAccuracy = 0.f;
        size_t dataTrainedCount = 0;
        uint percentDone = 0;
        uint percentStep = 10;

        // Load first batch of data.
        for (size_t dataIndex = 0; dataIndex < min(inputBatchSize, m_trainData.size()); ++dataIndex)
        {
            dataFiles.push_back(m_trainData[dataIndex].first);
            dataLabels.push_back(m_trainData[dataIndex].second);
        }
        m_allFirstTierLayersFpropped = true;
        LoadBatch(dataFiles, PropagationMode::Train);

        // Train on data, batch per batch.
        for (size_t dataIndex = inputBatchSize; dataIndex < m_trainData.size(); ++dataIndex)
        {
            nextDataFiles.push_back(m_trainData[dataIndex].first);
            nextDataLabels.push_back(m_trainData[dataIndex].second);

            if ((dataIndex + 1 ) % inputBatchSize == 0 || dataIndex == m_trainData.size() - 1)
            {
                // Upload data labels to output layer.
                outputLayer->LoadDataLabels(dataLabels);

                // Run training on current batch.
                m_allFirstTierLayersFpropped = false;
                trainThread = thread(&Trainer::TrainBatch, this, currEpoch);
                dataTrainedCount += inputBatchSize;

                dataFiles = nextDataFiles;
                dataLabels = nextDataLabels;
                // Run preloading of next batch, if it's not last.
                if (dataIndex < m_trainData.size() - 1)
                {
                    LoadBatch(dataFiles, PropagationMode::Train);
                    nextDataFiles.clear();
                    nextDataLabels.clear();
                }

                trainThread.join();

                // Printing results.				
                if (dataTrainedCount > (percentDone + percentStep) / 100.f * m_trainData.size())
                {
                    percentDone += percentStep;
                    PrintResults(percentDone, dataTrainedCount, PropagationMode::Train);
                }
            }
        }

        // Train on last batch, if there are enough data loaded.
        if (dataFiles.size() >= m_neuralNet->GetMaxNetworkTierSize())
        {
            // We need to have equal number of data trained in each split in case of data parallelism.
            size_t numDataToExclude = dataFiles.size() % m_neuralNet->GetMaxNetworkTierSize();
            for (size_t i = 0; i < numDataToExclude; ++i)
            {
                dataFiles.pop_back();
                dataLabels.pop_back();
            }

            outputLayer->LoadDataLabels(dataLabels);
            if (m_trainData.size() > inputBatchSize)
            {
                // Batch was not preloaded so we have to load it first.
                LoadBatch(dataFiles, PropagationMode::Train);
            }
            TrainBatch(currEpoch);
            dataTrainedCount += dataFiles.size();
        }
        dataFiles.clear();
        dataLabels.clear();
        nextDataFiles.clear();
        nextDataLabels.clear();

        PrintResults(100, dataTrainedCount, PropagationMode::Train);

        SaveCheckpoint(currEpoch, dataTrainedCount, currEpoch == m_numEpochs);
    }

    ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << "    Network training ended  [" << GetCurrentTimeStamp() << "]" << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << endl;
}

void Trainer::TestBatch()
{
    size_t currTier = 1;
    while (currTier < m_neuralNet->GetLayerTiers().size())
    {
        size_t nextTier;

        // Propagate.
        PropagateBatchForward(currTier, nextTier, PropagationMode::Test);

        // Move on to next tier.
        currTier = nextTier;
    }
}

void Trainer::TestBatchInPasses(vector<string>& dataFiles, vector<uint>& dataLabels, const vector<string>& nextDataFiles, const vector<uint>& nextDataLabels,
    uint numTestPasses, size_t& dataTestedCount)
{
    // Upload data labels to output layer.
    OutputLayer* outputLayer = m_neuralNet->GetOutputLayer();
    outputLayer->LoadDataLabels(dataLabels);

    dataTestedCount += dataFiles.size();

    for (uint testPass = 0; testPass < numTestPasses; ++testPass)
    {
        // Run testing on current batch.
        m_allFirstTierLayersFpropped = false;
        thread testThread = thread(&Trainer::TestBatch, this);

        bool preloadNextBatch = true;
        if (testPass == numTestPasses - 1)
        {
            // We can't preload next batch during network propagation if it has different data count,
            // because it would update input data counts in layers in middle of propagation.
            preloadNextBatch = dataFiles.size() == nextDataFiles.size();
            dataFiles = nextDataFiles;
            dataLabels = nextDataLabels;
        }

        if (preloadNextBatch)
        {
            LoadBatch(dataFiles, PropagationMode::Test);
        }

        testThread.join();
    }

    // Gathering batch results.
    m_accuracy += outputLayer->GetAccuracy();
    if (outputLayer->ShouldCalculateMultipleGuessAccuracy())
    {
        m_multipleGuessAccuracy += outputLayer->GetMultipleGuessAccuracy();
    }
}

void Trainer::TestNetwork()
{
    ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << "    Network testing started  [" << GetCurrentTimeStamp() << "]" << endl;
    cout << "**********************************************************************************************************************************" << endl;

    vector<string> dataFiles;
    vector<uint> dataLabels;
    vector<string> nextDataFiles;
    vector<uint> nextDataLabels;
    thread testThread;
    InputLayer* inputLayer = m_neuralNet->GetInputLayer();
    OutputLayer* outputLayer = m_neuralNet->GetOutputLayer();
    bool calculateMultipleGuessAccuracy = outputLayer->ShouldCalculateMultipleGuessAccuracy();

    m_accuracy = 0.f;
    m_multipleGuessAccuracy = 0.f;
    size_t dataTestedCount = 0;
    uint percentDone = 0;
    uint percentStep = 10;
    uint numTestPasses = inputLayer->GetNumTestPasses();

    // Load first batch of data.
    size_t batchSize = (size_t)m_batchSize;
    for (size_t dataIndex = 0; dataIndex < min(batchSize, m_testData.size()); ++dataIndex)
    {
        dataFiles.push_back(m_testData[dataIndex].first);
        dataLabels.push_back(m_testData[dataIndex].second);
    }
    m_allFirstTierLayersFpropped = true;
    LoadBatch(dataFiles, PropagationMode::Test);

    // Test on data, batch per batch.
    for (size_t dataIndex = batchSize; dataIndex < m_testData.size(); ++dataIndex)
    {
        nextDataFiles.push_back(m_testData[dataIndex].first);
        nextDataLabels.push_back(m_testData[dataIndex].second);
        if ((dataIndex + 1) % batchSize == 0 || dataIndex == m_testData.size() - 1)
        {
            TestBatchInPasses(dataFiles, dataLabels, nextDataFiles, nextDataLabels, numTestPasses, dataTestedCount);

            nextDataFiles.clear();
            nextDataLabels.clear();

            if (dataTestedCount > (percentDone + percentStep) / 100.f * m_testData.size())
            {
                percentDone += percentStep;
                PrintResults(percentDone, dataTestedCount, PropagationMode::Test);
            }
        }
    }

    // Test on last batch.
    if (m_testData.size() > batchSize && m_testData.size() % batchSize > 0)
    {
        // Batch was not preloaded so we have to load it first.
        LoadBatch(dataFiles, PropagationMode::Test);
    }
    TestBatchInPasses(dataFiles, dataLabels, nextDataFiles, nextDataLabels, numTestPasses, dataTestedCount);
    PrintResults(100, dataTestedCount, PropagationMode::Test);

    // Saving test results.
    ofstream resultsFile(m_workFolder + "\\" + c_resultsFileName, ios::app);
    resultsFile << "Test results:    Accuracy: " << m_accuracy / dataTestedCount;
    if (calculateMultipleGuessAccuracy)
    {
        resultsFile << "  Multiple guess accuracy: " << m_multipleGuessAccuracy / dataTestedCount;
    }
    resultsFile << endl;
    resultsFile.close();

    ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
    cout << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << "    Network testing ended  [" << GetCurrentTimeStamp() << "]" << endl;
    cout << "**********************************************************************************************************************************" << endl;
    cout << endl;
}

void Trainer::ResetDevices()
{
    // TODO: if maxTierSize is 1, then reset only m_defaultGpu device

    int numGpus;
    CudaAssert(cudaGetDeviceCount(&numGpus));
    for (int i = 0; i < numGpus; ++i)
    {
        CudaAssert(cudaSetDevice(i));
        CudaAssert(cudaDeviceReset());
    }

    CudaAssert(cudaSetDevice(0));
}

void Trainer::RunTraining()
{
    ShipAssert(m_numEpochs > 0, "Number of epochs to train is not specified.");

    // Creating work folder.
    ShipAssert(_mkdir(m_workFolder.c_str()) == 0 || errno == EEXIST, "Problem creating work directory \"" + m_workFolder + "\".");

    // Training network.
    SetDefaultDevice();
    InitializeNetwork(ParsingMode::Training);
    ValidateConfiguration();
    InitializeTrainer();
    CalculateMemoryConsumption();
    LoadTrainData();
    TrainNetwork();
}

void Trainer::RunTesting()
{
    SetDefaultDevice();
    InitializeNetwork(ParsingMode::Prediction);
    LoadTestData();
    TestNetwork();
}

void Trainer::RunTrainingWithTesting()
{
    RunTraining();

    delete m_neuralNet;
    m_neuralNet = NULL;

    // Just in case we left something on devices.
    ResetDevices();

    // To avoid freeing these buffers in trainer destructor, since they are destroyed when devices are reset.
    m_dataParallelTiersWeightGradientsHelperBuffer = NULL;
    m_dataParallelTiersBiasGradientsHelperBuffer = NULL;

    // Testing uses half of the memory since there are no gradients buffers involved, so we can double up the batch size.
    m_batchSize *= 2;

    RunTesting();
}