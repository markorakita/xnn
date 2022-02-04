// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network features extractor.
// Created: 04/03/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/featurizer.cuh"

#include <fstream>
#include <iostream>

#include "../neuralnetwork/include/configurationparser.cuh"
#include "../neuralnetwork/include/neuralnet.cuh"
#include "../neuralnetwork/layers/include/inputlayer.cuh"
#include "../utils/include/asserts.cuh"
#include "../utils/include/consolehelper.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/cudahelper.cuh"
#include "../utils/include/utils.cuh"

const string Featurizer::c_featuresFileName = "features.txt";

const string Featurizer::c_configurationSignature = "-configfile";
const string Featurizer::c_inputFolderSignature = "-inputfolder";
const string Featurizer::c_inputDataNamesListSignature = "-inputdatalist";
const string Featurizer::c_modelFileSignature = "-modelfile";
const string Featurizer::c_batchSizeSignature = "-batchsize";
const string Featurizer::c_targetLayerSignature = "-layer";

Featurizer::Featurizer()
{
	m_neuralNet = NULL;
	m_hostTargetLayerActivations = NULL;

	m_configurationParser = new ConfigurationParser();
	
	m_batchSize = 0;
	m_targetLayerIndex = 0;
}

Featurizer::~Featurizer()
{
	if (m_neuralNet != NULL)
	{
		delete m_neuralNet;
	}

	delete m_configurationParser;

	if (m_hostTargetLayerActivations != NULL)
	{
		CudaAssert(cudaFreeHost(m_hostTargetLayerActivations));
	}
}

bool Featurizer::ParseArguments(int argc, char *argv[])
{
	if (!ParseArgument(argc, argv, c_inputFolderSignature, m_inputFolder) ||
		!ParseArgument(argc, argv, c_inputDataNamesListSignature, m_inputDataNamesListFile))
	{
		return false;
	}

	uint targetLayer;
	if (!ParseArgument(argc, argv, c_configurationSignature, m_networkConfigurationFile) ||
		!ParseArgument(argc, argv, c_modelFileSignature, m_modelFile) ||
		!ParseArgument(argc, argv, c_batchSizeSignature, m_batchSize) ||
		!ParseArgument(argc, argv, c_targetLayerSignature, targetLayer))
	{
		return false;
	}

	m_targetLayerIndex = targetLayer;

	return true;
}

void Featurizer::InitializeNetwork()
{
	m_neuralNet = m_configurationParser->ParseNetworkFromConfiguration(ParsingMode::Prediction, m_networkConfigurationFile, m_inputFolder, m_batchSize, false);

	// TODO: Implement support for models trained on multiple GPUs.
	ShipAssert(m_neuralNet->GetMaxNetworkTierSize() == 1, "Current implementation of featurizer only works with models trained on single GPU.");

	m_neuralNet->LoadModelForPrediction(m_modelFile);
}

void Featurizer::LoadData()
{
	// Loading data.
	ifstream dataList(m_inputDataNamesListFile);
	string fileName;
	while (getline(dataList, fileName))
	{
		m_data.push_back(fileName);
	}
}

void Featurizer::LoadBatch(const vector<string>& dataFiles)
{
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();

	inputLayer->SetDataFilesToLoad(dataFiles, PropagationMode::Featurization);
	
	// Load data from disk to host memory.
	inputLayer->LoadInputs();

	// Load data from host to GPU memory.
	inputLayer->DoForwardProp(PropagationMode::Featurization);
}

void Featurizer::ExtractFeaturesOnBatch()
{
	const vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
	for (size_t currTier = 1; currTier <= m_targetLayerIndex; ++currTier)
	{
		Layer* layer = layerTiers[currTier][0];
		
		layer->LoadInputs();
		if (layer->HoldsInputData())
		{
			// Making sure inputs are loaded before computation.
			layer->SynchronizeMemoryOperations();
		}

		layer->DoForwardProp(PropagationMode::Featurization);
	}

	Layer* targetLayer = layerTiers[m_targetLayerIndex][0];
	targetLayer->SynchronizeCalculations();

	// Copy results to host.
	if (m_hostTargetLayerActivations == NULL)
	{
		CudaAssert(cudaMallocHost<float>(&m_hostTargetLayerActivations, targetLayer->GetActivationBufferSize(), cudaHostAllocPortable));
	}
	CudaAssert(cudaMemcpy(m_hostTargetLayerActivations, targetLayer->GetActivationDataBuffer(), targetLayer->GetActivationBufferSize(), cudaMemcpyDeviceToHost));

	// Append results to file.
	ofstream featuresFile(m_inputFolder + "\\" + c_featuresFileName, ios::app);
	uint dataCount = targetLayer->GetActivationDataCount();
	uint activationBufferLength = (uint)(targetLayer->GetActivationBufferSize() / sizeof(float));
	for (uint i = 0; i < dataCount; ++i)
	{
		featuresFile << m_hostTargetLayerActivations[i];
		for (uint j = i + dataCount; j < activationBufferLength; j += dataCount)
		{
			featuresFile << " " << m_hostTargetLayerActivations[j];
		}
		featuresFile << endl;
	}
}

void Featurizer::ExtractFeatures()
{
	ConsoleHelper::SetConsoleForeground(ConsoleColor::WHITE);
	cout << endl;
	cout << "**********************************************************************************************************************************" << endl;
	cout << "    Features extraction started  [" << GetCurrentTimeStamp() << "]" << endl;
	cout << "**********************************************************************************************************************************" << endl;

	vector<string> dataFiles;
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();
	size_t batchSize = (size_t)m_batchSize;
	size_t currStep = 1;
	size_t percentPerStep = 10;

	// Run features extraction on data, batch per batch.
	ConsoleHelper::SetConsoleForeground(ConsoleColor::GRAY);
	for (size_t dataIndex = 0; dataIndex < m_data.size(); ++dataIndex)
	{
		dataFiles.push_back(m_data[dataIndex]);
		if ((dataIndex + 1) % batchSize == 0 || dataIndex == m_data.size() - 1)
		{
			// Load data for current batch.
			LoadBatch(dataFiles);

			// Run extraction on current batch.
			ExtractFeaturesOnBatch();

			dataFiles.clear();
		}

		if (dataIndex > currStep * percentPerStep * m_data.size() / 100)
		{
			cout << "Done: " << currStep * percentPerStep << "%" << endl;
			++currStep;
		}
	}

	ConsoleHelper::SetConsoleForeground(ConsoleColor::GREEN);
	cout << endl << "Features extraction finished!" << endl << endl << endl;
}

void Featurizer::RunExtraction()
{
	InitializeNetwork();
	LoadData();
	ExtractFeatures();
}