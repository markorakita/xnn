// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural networks configuration parser.
// Created: 03/17/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/configurationparser.cuh"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "include/activationfunctions.cuh"
#include "include/neuralnet.cuh"
#include "layers/include/convolutionallayer.cuh"
#include "layers/include/dropoutlayer.cuh"
#include "layers/include/inputlayer.cuh"
#include "layers/include/maxpoollayer.cuh"
#include "layers/include/outputlayer.cuh"
#include "layers/include/responsenormalizationlayer.cuh"
#include "layers/include/softmaxlayer.cuh"
#include "layers/include/standardlayer.cuh"
#include "../utils/include/asserts.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/cudahelper.cuh"
#include "../utils/include/utils.cuh"

const string ConfigurationParser::c_layerTypeParam = "layer";
const string ConfigurationParser::c_tierSizeParam = "tierSize";
const string ConfigurationParser::c_dataTypeParam = "dataType";
const string ConfigurationParser::c_numChannelsParam = "numChannels";
const string ConfigurationParser::c_inputDataWidthParam = "inputDataWidth";
const string ConfigurationParser::c_inputDataHeightParam = "inputDataHeight";
const string ConfigurationParser::c_originalDataWidthParam = "originalDataWidth";
const string ConfigurationParser::c_originalDataHeightParam = "originalDataHeight";
const string ConfigurationParser::c_doRandomFlipsParam = "doRandomFlips";
const string ConfigurationParser::c_normalizeInputsParam = "normalizeInputs";
const string ConfigurationParser::c_inputMeansParam = "inputMeans";
const string ConfigurationParser::c_inputStDevsParam = "inputStDevs";
const string ConfigurationParser::c_numTestPatchesParam = "numTestPatches";
const string ConfigurationParser::c_testOnFlipsParam = "testOnFlips";
const string ConfigurationParser::c_parallelismParam = "parallelism";
const string ConfigurationParser::c_prevLayersParam = "prevLayers";
const string ConfigurationParser::c_weightsInitializationParam = "weightsInitialization";
const string ConfigurationParser::c_weightsInitialValueParam = "weightsInitialValue";
const string ConfigurationParser::c_weightsRangeStartParam = "weightsRangeStart";
const string ConfigurationParser::c_weightsRangeEndParam = "weightsRangeEnd";
const string ConfigurationParser::c_weightsMeanParam = "weightsMean";
const string ConfigurationParser::c_weightsStdDevParam = "weightsStdDev";
const string ConfigurationParser::c_biasesInitializationParam = "biasesInitialization";
const string ConfigurationParser::c_biasesInitialValueParam = "biasesInitialValue";
const string ConfigurationParser::c_biasesRangeStartParam = "biasesRangeStart";
const string ConfigurationParser::c_biasesRangeEndParam = "biasesRangeEnd";
const string ConfigurationParser::c_biasesMeanParam = "biasesMean";
const string ConfigurationParser::c_biasesStdDevParam = "biasesStdDev";
const string ConfigurationParser::c_numFiltersParam = "numFilters";
const string ConfigurationParser::c_filterWidthParam = "filterWidth";
const string ConfigurationParser::c_filterHeightParam = "filterHeight";
const string ConfigurationParser::c_filterPaddingXParam = "paddingX";
const string ConfigurationParser::c_filterPaddingYParam = "paddingY";
const string ConfigurationParser::c_filterStrideParam = "stride";
const string ConfigurationParser::c_weightsMomentumParam = "weightsMomentum";
const string ConfigurationParser::c_weightsDecayParam = "weightsDecay";
const string ConfigurationParser::c_weightsStartingLRParam = "weightsStartingLR";
const string ConfigurationParser::c_weightsLRStepParam = "weightsLRStep";
const string ConfigurationParser::c_weightsLRFactorParam = "weightsLRFactor";
const string ConfigurationParser::c_biasesMomentumParam = "biasesMomentum";
const string ConfigurationParser::c_biasesDecayParam = "biasesDecay";
const string ConfigurationParser::c_biasesStartingLRParam = "biasesStartingLR";
const string ConfigurationParser::c_biasesLRStepParam = "biasesLRStep";
const string ConfigurationParser::c_biasesLRFactorParam = "biasesLRFactor";
const string ConfigurationParser::c_activationTypeParam = "activationType";
const string ConfigurationParser::c_activationAlphaParam = "activationAlpha";
const string ConfigurationParser::c_reNormDepthParam = "depth";
const string ConfigurationParser::c_reNormBiasParam = "bias";
const string ConfigurationParser::c_reNormAlphaCoeffParam = "alphaCoeff";
const string ConfigurationParser::c_reNormBetaCoeffParam = "betaCoeff";
const string ConfigurationParser::c_numNeuronsParam = "numNeurons";
const string ConfigurationParser::c_lossFunctionParam = "lossFunction";
const string ConfigurationParser::c_numGuessesParam = "numGuesses";
const string ConfigurationParser::c_dropProbabilityParam = "dropProbability";

const string ConfigurationParser::c_prevLayersOptionAll = "all";

const uint ConfigurationParser::c_tierSizeDefaultValue = 1;
const uint ConfigurationParser::c_numChannelsDefaultValue = 1;
const uint ConfigurationParser::c_inputDataHeightDefaultValue = 1;
const bool ConfigurationParser::c_doRandomFlipsDefaultValue = false;
const bool ConfigurationParser::c_normalizeInputsDefaultValue = false;
const uint ConfigurationParser::c_numTestPatchesDefaultValue = 1;
const bool ConfigurationParser::c_testOnFlipsDefaultValue = false;
const uint ConfigurationParser::c_filterPaddingXDefaultValue = 0;
const uint ConfigurationParser::c_filterPaddingYDefaultValue = 0;
const uint ConfigurationParser::c_filterStrideDefaultValue = 1;
const float ConfigurationParser::c_weightsDefaultInitialValue = 0.f;
const float ConfigurationParser::c_weightsRangeStartDefaultValue = -0.1f;
const float ConfigurationParser::c_weightsRangeEndDefaultValue = 0.1f;
const float ConfigurationParser::c_weightsMeanDefaultValue = 0.f;
const float ConfigurationParser::c_weightsStdDevDefaultValue = 0.01f;
const float ConfigurationParser::c_biasesDefaultInitialValue = 1.f;
const float ConfigurationParser::c_weightsMomentumDefaultValue = 0.f;
const float ConfigurationParser::c_weightsDecayDefaultValue = 0.f;
const float ConfigurationParser::c_dropProbabilityDefaultValue = 0.5f;
const float ConfigurationParser::c_activationAlphaDefaultValue = 0.f;

ConfigurationParser::ConfigurationParser()
{
	// Initializing to default values, just to please the Gods of IntelliSense.
	m_neuralNet = NULL;
	m_parsingMode = ParsingMode::Training;
	m_batchSize = 0;
	m_initializeLayersParams = false;
	m_maxNetworkTierSize = 0;
	ResetWeightsInitializationParams();
}

NeuralNet* ConfigurationParser::ParseNetworkFromConfiguration(ParsingMode parsingMode, const string& configurationFile, const string& dataFolder, uint batchSize,
	bool initializeLayersParams)
{
	m_parsingMode = parsingMode;
	m_dataFolder = dataFolder;
	m_batchSize = batchSize;
	m_initializeLayersParams = initializeLayersParams;
	m_layersTiers.clear();
	m_tiersLines.clear();

	ParseTierLines(configurationFile);
	FindMaxNetworkTierSize();

	m_neuralNet = new NeuralNet(m_maxNetworkTierSize);

	ParseLayersTiers();
	
	// Reverting back to default device.
	CudaAssert(cudaSetDevice(0));

	for (vector<Layer*>& layersTier: m_layersTiers)
	{
		m_neuralNet->AddLayersTier(layersTier);
	}
	
	return m_neuralNet;
}

string ConfigurationParser::TrimLine(const string& line)
{
	if (line == "")
	{
		return line;
	}

	string trimmedLine;

	// Trim leading whitespace.
	size_t firstNonWs = line.find_first_not_of(" \t");
	if (firstNonWs != string::npos)
	{
		trimmedLine = line.substr(firstNonWs);
	}

	// Trim trailing whitespace.
	size_t lastNonWs = trimmedLine.find_last_not_of(" \t");
	if (lastNonWs != string::npos)
	{
		trimmedLine = trimmedLine.substr(0, lastNonWs + 1);
	}

	return trimmedLine;
}

void ConfigurationParser::ParseTierLines(const string& configurationFile)
{
	ifstream configuration(configurationFile);
	string line;
	vector<string> currTierLines;
	while (getline(configuration, line))
	{
		string trimmedLine = TrimLine(line);

		if (trimmedLine.rfind("//", 0) == 0)
		{
			continue;
		}
		else if (trimmedLine.rfind(c_layerTypeParam + ":", 0) == 0)
		{
			if (!currTierLines.empty())
			{
				m_tiersLines.push_back(currTierLines);
				currTierLines.clear();
			}

			currTierLines.push_back(trimmedLine);
		}
		else if (!currTierLines.empty())
		{
			if (trimmedLine == "" || trimmedLine[0] < 'a' || trimmedLine[0] > 'z')
			{
				m_tiersLines.push_back(currTierLines);
				currTierLines.clear();
			}
			else
			{
				currTierLines.push_back(trimmedLine);
			}
		}
	}

	if (!currTierLines.empty())
	{
		m_tiersLines.push_back(currTierLines);
	}
}

bool ConfigurationParser::GetParameterValueStrFromLine(const string& line, const string& parameterName, string& parameterValueStr)
{
	if (line.rfind(parameterName + ":", 0) == 0 || line.rfind(parameterName + " ", 0) == 0)
	{
		parameterValueStr = TrimLine(line.substr(line.find_first_of(":", parameterName.length()) + 1));
		return true;
	}

	return false;
}

bool ConfigurationParser::ParseParameterUint(const string& line, const string& parameterName, uint& parameterValue)
{
	string parameterValueStr;
	if (!GetParameterValueStrFromLine(line, parameterName, parameterValueStr))
	{
		return false;
	}

	parameterValue = stoul(parameterValueStr);

	return true;
}

bool ConfigurationParser::ParseParameterFloat(const string& line, const string& parameterName, float& parameterValue)
{
	string parameterValueStr;
	if (!GetParameterValueStrFromLine(line, parameterName, parameterValueStr))
	{
		return false;
	}

	parameterValue = stof(parameterValueStr);

	return true;
}

bool ConfigurationParser::ParseParameterBool(const string& line, const string& parameterName, bool& parameterValue)
{
	string parameterValueStr;
	if (!GetParameterValueStrFromLine(line, parameterName, parameterValueStr))
	{
		return false;
	}

	parameterValue = parameterValueStr == "yes";

	return true;
}

bool ConfigurationParser::ParseParameterString(const string& line, const string& parameterName, string& parameterValue)
{
	string parameterValueStr;
	if (!GetParameterValueStrFromLine(line, parameterName, parameterValueStr))
	{
		return false;
	}

	parameterValue = ConvertToLowercase(parameterValueStr);

	return true;
}

void ConfigurationParser::FindMaxNetworkTierSize()
{
	m_maxNetworkTierSize = 1;
	
	if (m_parsingMode == ParsingMode::Training)
	{
		for (vector<string>& tierLines : m_tiersLines)
		{
			for (string& line : tierLines)
			{
				uint tierSize = 1;
				ParseParameterUint(line, c_tierSizeParam, tierSize);
				m_maxNetworkTierSize = max((uint)m_maxNetworkTierSize, tierSize);
			}
		}
	}
}

LayerType ConfigurationParser::GetLayerType(const string& layerTypeName)
{
	if (layerTypeName == "input")
	{
		return LayerType::Input;
	}
	else if (layerTypeName == "convolutional")
	{
		return LayerType::Convolutional;
	}
	else if (layerTypeName == "responsenormalization")
	{
		return LayerType::ResponseNormalization;
	}
	else if (layerTypeName == "maxpool")
	{
		return LayerType::MaxPool;
	}
	else if (layerTypeName == "standard")
	{
		return LayerType::Standard;
	}
	else if (layerTypeName == "dropout")
	{
		return LayerType::Dropout;
	}
	else if (layerTypeName == "softmax")
	{
		return LayerType::SoftMax;
	}
	else if (layerTypeName == "output")
	{
		return LayerType::Output;
	}
	else
	{
		ShipAssert(false, "Unknown layer type name: " + layerTypeName);
		return LayerType::Standard;
	}
}

ActivationType ConfigurationParser::GetActivationType(const string& activationTypeName)
{
	if (activationTypeName == "linear")
	{
		return ActivationType::Linear;
	}
	else if (activationTypeName == "relu")
	{
		return ActivationType::ReLU;
	}
	else if (activationTypeName == "elu")
	{
		return ActivationType::ELU;
	}
	else if (activationTypeName == "leakyrelu" || activationTypeName == "lrelu")
	{
		return ActivationType::LeakyReLU;
	}
	else if (activationTypeName == "sigmoid")
	{
		return ActivationType::Sigmoid;
	}
	else if (activationTypeName == "tanh")
	{
		return ActivationType::Tanh;
	}
	else
	{
		ShipAssert(false, "Unknown activation type name: " + activationTypeName);
		return ActivationType::Linear;
	}
}

LossFunctionType ConfigurationParser::GetLossFunctionType(const string& lossFunctionName)
{
	if (lossFunctionName == "crossentropy")
	{
		return LossFunctionType::CrossEntropy;
	}
	if (lossFunctionName == "logisticregression")
	{
		return LossFunctionType::LogisticRegression;
	}
	else
	{
		ShipAssert(false, "Unknown loss function name: " + lossFunctionName);
		return LossFunctionType::LogisticRegression;
	}
}

DataType ConfigurationParser::GetDataType(const string& dataTypeName)
{
	if (dataTypeName == "image")
	{
		return DataType::Image;
	}
	else if (dataTypeName == "text")
	{
		return DataType::Text;
	}
	else
	{
		ShipAssert(false, "Unknown data type name: " + dataTypeName);
		return DataType::Text;
	}
}

ParallelismMode ConfigurationParser::GetParallelismMode(const string& parallelismModeName)
{
	if (parallelismModeName == "data")
	{
		return ParallelismMode::Data;
	}
	else if (parallelismModeName == "model")
	{
		return ParallelismMode::Model;
	}
	else
	{
		ShipAssert(false, "Unknown parallelism mode name: " + parallelismModeName);
		return ParallelismMode::Model;
	}
}

ConfigurationParser::ParamInitializationMode ConfigurationParser::GetParamInitializationMode(const string& paramInitializationModeName)
{
	if (paramInitializationModeName == "constant")
	{
		return ParamInitializationMode::Constant;
	}
	else if (paramInitializationModeName == "uniform")
	{
		return ParamInitializationMode::Uniform;
	}
	else if (paramInitializationModeName == "gaussian" || paramInitializationModeName == "normal")
	{
		return ParamInitializationMode::Gaussian;
	}
	else if (paramInitializationModeName == "xavier")
	{
		return ParamInitializationMode::Xavier;
	}
	else if (paramInitializationModeName == "he")
	{
		return ParamInitializationMode::He;
	}
	else
	{
		ShipAssert(false, "Unknown parameters initialization mode name: " + paramInitializationModeName);
		return ParamInitializationMode::He;
	}
}

void ConfigurationParser::ParseLayersTiers()
{
	for (size_t tierIndex = 0; tierIndex < m_tiersLines.size(); ++tierIndex)
	{
		const vector<string>& tierLines = m_tiersLines[tierIndex];
		string layerTypeName;
		ParseParameterString(tierLines[0], c_layerTypeParam, layerTypeName);
		LayerType tierLayerType = GetLayerType(layerTypeName);

		if (tierIndex == 0)
		{			
			ShipAssert(tierLayerType == LayerType::Input, "First layer in the network should be input layer!");
		}
		else if (tierLayerType == LayerType::Input)
		{
			ShipAssert(tierIndex == 0, "Only first layer in the network can be input layer!");
		}
		else if (tierIndex == m_tiersLines.size() - 1)
		{
			ShipAssert(tierLayerType == LayerType::Output, "Last layer in the network should be output layer!");
		}
		else if (tierLayerType == LayerType::Output)
		{
			ShipAssert(tierIndex == m_tiersLines.size() - 1, "Only last layer in the network can be output layer!");
		}

		vector<Layer*> layerTier = ParseLayersTier(tierIndex, tierLayerType);

		m_layersTiers.push_back(layerTier);
	}
}

vector<Layer*> ConfigurationParser::FindPrevLayers(ParallelismMode currTierParallelismMode, uint layerIndexInTier, uint currTierSize, size_t prevTierIndex, const string& prevLayersParam)
{
	vector<Layer*> prevLayers;

	if (m_parsingMode == ParsingMode::Prediction || (prevLayersParam != c_prevLayersOptionAll && currTierParallelismMode == m_layersTiers[prevTierIndex][0]->GetParallelismMode() &&
		currTierSize == m_layersTiers[prevTierIndex].size()))
	{
		prevLayers.push_back(m_layersTiers[prevTierIndex][layerIndexInTier]);
	}
	else
	{
		prevLayers = m_layersTiers[prevTierIndex];
	}

	return prevLayers;
}

bool ConfigurationParser::ShouldHoldActivationGradients(ParallelismMode currTierParallelismMode, uint currTierSize, size_t currTierIndex, uint layerIndexInTier)
{
	if (m_parsingMode == ParsingMode::Prediction || currTierIndex == m_tiersLines.size() - 1)
	{
		return false;
	}

	uint nextTierSize = c_tierSizeDefaultValue;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode nextTierParallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;
	for (const string& line : m_tiersLines[currTierIndex + 1])
	{
		if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, nextTierSize))
		{
			parsedTierSize = true;
		}
		else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
		{
			parsedParallelism = true;
			nextTierParallelismMode = GetParallelismMode(parallelismValue);
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
	}

	return !((currTierSize == nextTierSize && (currTierSize == 1 || (currTierParallelismMode == nextTierParallelismMode && prevLayersParam != c_prevLayersOptionAll))) ||
		(nextTierSize == 1 && layerIndexInTier == 0 && currTierParallelismMode == ParallelismMode::Model && nextTierParallelismMode == ParallelismMode::Model));
}

void ConfigurationParser::FindInputParams(const vector<Layer*>& prevLayers, uint layerIndexInTier, uint tierSize, ParallelismMode parallelismMode,
	uint& inputNumChannels, uint& inputDataWidth, uint& inputDataHeight, uint& inputDataCount, bool& holdsInputData)
{
	if (prevLayers[0]->GetLayerType() == LayerType::Input)
	{
		InputLayer* inputLayer = static_cast<InputLayer*>(prevLayers[0]);
		inputNumChannels = inputLayer->GetActivationNumChannels();
		inputDataWidth = inputLayer->GetActivationDataWidth();
		inputDataHeight = inputLayer->GetActivationDataHeight();
		if (parallelismMode == ParallelismMode::Data)
		{
			inputDataCount = inputLayer->GetInputDataCount() / tierSize;
		}
		else
		{
			inputDataCount = inputLayer->GetInputDataCount();
		}
	}
	else if (prevLayers[0]->GetParallelismMode() == ParallelismMode::Data)
	{
		inputNumChannels = prevLayers[0]->GetActivationNumChannels();
		inputDataWidth = prevLayers[0]->GetActivationDataWidth();
		inputDataHeight = prevLayers[0]->GetActivationDataHeight();
		inputDataCount = prevLayers[0]->GetActivationDataCount();
	}
	else if (prevLayers[0]->GetLayerType() == LayerType::Convolutional || prevLayers[0]->GetLayerType() == LayerType::ResponseNormalization ||
		prevLayers[0]->GetLayerType() == LayerType::MaxPool)
	{
		inputNumChannels = prevLayers[0]->GetActivationNumChannels();
		for (size_t i = 1; i < prevLayers.size(); ++i)
		{
			inputNumChannels += prevLayers[i]->GetActivationNumChannels();
		}
		inputDataWidth = prevLayers[0]->GetActivationDataWidth();
		inputDataHeight = prevLayers[0]->GetActivationDataHeight();
		inputDataCount = prevLayers[0]->GetActivationDataCount();
	}
	else
	{
		inputNumChannels = prevLayers[0]->GetActivationNumChannels();
		inputDataWidth = prevLayers[0]->GetActivationDataWidth();
		for (size_t i = 1; i < prevLayers.size(); ++i)
		{
			inputDataWidth += prevLayers[i]->GetActivationDataWidth();
		}
		inputDataHeight = prevLayers[0]->GetActivationDataHeight();
		inputDataCount = prevLayers[0]->GetActivationDataCount();
	}

	holdsInputData = prevLayers.size() > 1 ||
		(prevLayers[0]->GetIndexInTier() != layerIndexInTier && (prevLayers[0]->GetLayerType() != LayerType::Input || parallelismMode == ParallelismMode::Model));
}

void ConfigurationParser::ResetWeightsInitializationParams()
{
	m_weightsInitializationMode = ParamInitializationMode::Constant;
	m_parsedWeightsInitialization = false;
	m_weightsInitialValue = c_weightsDefaultInitialValue;
	m_parsedWeightsInitialValue = false;
	m_weightsRangeStart = c_weightsRangeStartDefaultValue;
	m_parsedWeightsRangeStart = false;
	m_weightsRangeEnd = c_weightsRangeEndDefaultValue;
	m_parsedWeightsRangeEnd = false;
	m_weightsMean = c_weightsMeanDefaultValue;
	m_parsedWeightsMean = false;
	m_weightsStdDev = c_weightsStdDevDefaultValue;
	m_parsedWeightsStdDev = false;
	m_biasesInitializationMode = ParamInitializationMode::Constant;
	m_parsedBiasesInitialization = false;
	m_biasesInitialValue = c_biasesDefaultInitialValue;
	m_parsedBiasesInitialValue = false;
	m_biasesRangeStart = c_weightsRangeStartDefaultValue;
	m_parsedBiasesRangeStart = false;
	m_biasesRangeEnd = c_weightsRangeEndDefaultValue;
	m_parsedBiasesRangeEnd = false;
	m_biasesMean = c_weightsMeanDefaultValue;
	m_parsedBiasesMean = false;
	m_biasesStdDev = c_weightsStdDevDefaultValue;
	m_parsedBiasesStdDev = false;
}

void ConfigurationParser::ParseWeightsInitializationParams(const string& line)
{
	string weightsInitializationModeValue, biasesInitializationModeValue;
	if (!m_parsedWeightsInitialization && ParseParameterString(line, c_weightsInitializationParam, weightsInitializationModeValue))
	{
		m_parsedWeightsInitialization = true;
		m_weightsInitializationMode = GetParamInitializationMode(weightsInitializationModeValue);
	}
	else if (!m_parsedWeightsInitialValue && ParseParameterFloat(line, c_weightsInitialValueParam, m_weightsInitialValue))
	{
		m_parsedWeightsInitialValue = true;
	}
	else if (!m_parsedWeightsRangeStart && ParseParameterFloat(line, c_weightsRangeStartParam, m_weightsRangeStart))
	{
		m_parsedWeightsRangeStart = true;
	}
	else if (!m_parsedWeightsRangeEnd && ParseParameterFloat(line, c_weightsRangeEndParam, m_weightsRangeEnd))
	{
		m_parsedWeightsRangeEnd = true;
	}
	else if (!m_parsedWeightsMean && ParseParameterFloat(line, c_weightsMeanParam, m_weightsMean))
	{
		m_parsedWeightsMean = true;
	}
	else if (!m_parsedWeightsStdDev && ParseParameterFloat(line, c_weightsStdDevParam, m_weightsStdDev))
	{
		m_parsedWeightsStdDev = true;
	}
	else if (!m_parsedBiasesInitialization && ParseParameterString(line, c_biasesInitializationParam, biasesInitializationModeValue))
	{
		m_parsedBiasesInitialization = true;
		m_biasesInitializationMode = GetParamInitializationMode(biasesInitializationModeValue);
	}
	else if (!m_parsedBiasesInitialValue && ParseParameterFloat(line, c_biasesInitialValueParam, m_biasesInitialValue))
	{
		m_parsedBiasesInitialValue = true;
	}
	else if (!m_parsedBiasesRangeStart && ParseParameterFloat(line, c_biasesRangeStartParam, m_biasesRangeStart))
	{
		m_parsedBiasesRangeStart = true;
	}
	else if (!m_parsedBiasesRangeEnd && ParseParameterFloat(line, c_biasesRangeEndParam, m_biasesRangeEnd))
	{
		m_parsedBiasesRangeEnd = true;
	}
	else if (!m_parsedBiasesMean && ParseParameterFloat(line, c_biasesMeanParam, m_biasesMean))
	{
		m_parsedBiasesMean = true;
	}
	else if (!m_parsedBiasesStdDev && ParseParameterFloat(line, c_biasesStdDevParam, m_biasesStdDev))
	{
		m_parsedBiasesStdDev = true;
	}
}

void ConfigurationParser::InitializeLayerWeights(WeightsLayer* weightsLayer)
{
	if (m_weightsInitializationMode == ParamInitializationMode::Constant)
	{
		weightsLayer->InitializeWeightsToConstant(m_weightsInitialValue);
	}
	else if (m_weightsInitializationMode == ParamInitializationMode::Uniform)
	{
		weightsLayer->InitializeWeightsFromUniformDistribution(m_weightsRangeStart, m_weightsRangeEnd);
	}
	else if (m_weightsInitializationMode == ParamInitializationMode::Gaussian)
	{
		weightsLayer->InitializeWeightsFromNormalDistribution(m_weightsMean, m_weightsStdDev);
	}
	else if (m_weightsInitializationMode == ParamInitializationMode::Xavier)
	{
		weightsLayer->InitializeWeightsXavier();
	}
	else if (m_weightsInitializationMode == ParamInitializationMode::He)
	{
		weightsLayer->InitializeWeightsHe();
	}
	else
	{
		ShipAssert(false, "Unsupported weights initialization mode!");
	}
}

void ConfigurationParser::InitializeLayerBiases(WeightsLayer* weightsLayer)
{
	if (m_biasesInitializationMode == ParamInitializationMode::Constant)
	{
		weightsLayer->InitializeBiasesToConstant(m_biasesInitialValue);
	}
	else if (m_biasesInitializationMode == ParamInitializationMode::Uniform)
	{
		weightsLayer->InitializeBiasesFromUniformDistribution(m_biasesRangeStart, m_biasesRangeEnd);
	}
	else if (m_biasesInitializationMode == ParamInitializationMode::Gaussian)
	{
		weightsLayer->InitializeBiasesFromNormalDistribution(m_biasesMean, m_biasesStdDev);
	}
	else
	{
		ShipAssert(false, "Unsupported biases initialization mode!");
	}
}

void ConfigurationParser::ParseInputLayerTier(vector<Layer*>& outLayerTier)
{
	string dataTypeValue;
	DataType dataType;
	bool parsedDataType = false;
	uint numChannels = c_numChannelsDefaultValue;
	bool parsedNumChannels = false;
	uint inputDataWidth;
	bool parsedInputDataWidth = false;
	uint inputDataHeight = c_inputDataHeightDefaultValue;
	bool parsedInputDataHeight = false;
	uint originalDataWidth;
	bool parsedOriginalDataWidth = false;
	uint originalDataHeight;
	bool parsedOriginalDataHeight = false;
	bool doRandomFlips = c_doRandomFlipsDefaultValue;
	bool parsedDoRandomFlips = false;
	bool normalizeInputs = c_normalizeInputsDefaultValue;
	bool parsedNormalizeInputs = false;
	string inputMeansValue;
	bool parsedInputMeans = false;
	string inputStDevsValue;
	bool parsedInputStDevs = false;
	uint numTestPatches = c_numTestPatchesDefaultValue;
	bool parsedNumTestPatches = false;
	bool testOnFlips = c_testOnFlipsDefaultValue;
	bool parsedTestOnFlips = false;

	for (const string& line : m_tiersLines[0])
	{
		if (!parsedDataType && ParseParameterString(line, c_dataTypeParam, dataTypeValue))
		{
			parsedDataType = true;
		}
		else if (!parsedNumChannels && ParseParameterUint(line, c_numChannelsParam, numChannels))
		{
			parsedNumChannels = true;
		}
		else if (!parsedInputDataWidth && ParseParameterUint(line, c_inputDataWidthParam, inputDataWidth))
		{
			parsedInputDataWidth = true;
		}
		else if (!parsedInputDataHeight && ParseParameterUint(line, c_inputDataHeightParam, inputDataHeight))
		{
			parsedInputDataHeight = true;
		}
		else if (!parsedOriginalDataWidth && ParseParameterUint(line, c_originalDataWidthParam, originalDataWidth))
		{
			parsedOriginalDataWidth = true;
		}
		else if (!parsedOriginalDataHeight && ParseParameterUint(line, c_originalDataHeightParam, originalDataHeight))
		{
			parsedOriginalDataHeight = true;
		}
		else if (!parsedDoRandomFlips && ParseParameterBool(line, c_doRandomFlipsParam, doRandomFlips))
		{
			parsedDoRandomFlips = true;
		}
		else if (!parsedNormalizeInputs && ParseParameterBool(line, c_normalizeInputsParam, normalizeInputs))
		{
			parsedNormalizeInputs = true;
		}
		else if (!parsedInputMeans && ParseParameterString(line, c_inputMeansParam, inputMeansValue))
		{
			parsedInputMeans = true;
		}
		else if (!parsedInputStDevs && ParseParameterString(line, c_inputStDevsParam, inputStDevsValue))
		{
			parsedInputStDevs = true;
		}
		else if (!parsedNumTestPatches && ParseParameterUint(line, c_numTestPatchesParam, numTestPatches))
		{
			parsedNumTestPatches = true;
		}
		else if (!parsedTestOnFlips && ParseParameterBool(line, c_testOnFlipsParam, testOnFlips))
		{
			parsedTestOnFlips = true;
		}
	}

	ShipAssert(parsedDataType, "Can't parse data type for Input layer!");
	dataType = GetDataType(dataTypeValue);
	if (dataType == DataType::Image)
	{
		ShipAssert(parsedNumChannels, "Can't parse number of channels for Input layer!");
	}

	ShipAssert(parsedInputDataWidth, "Can't parse input data width for Input layer!");
	if (!parsedOriginalDataWidth)
	{
		originalDataWidth = inputDataWidth;
	}
	if (!parsedOriginalDataHeight)
	{
		originalDataHeight = inputDataHeight;
	}

	vector<float> inputMeans, inputStDevs;
	if (normalizeInputs)
	{
		ShipAssert(parsedInputMeans, "Can't parse input means for Input layer!");
		ShipAssert(parsedInputStDevs, "Can't parse input standard deviations for Input layer!");

		replace(inputMeansValue.begin(), inputMeansValue.end(), ',', ' ');
		istringstream inputMeansParser(inputMeansValue);
		float mean;
		while (inputMeansParser >> mean)
		{
			inputMeans.push_back(mean);
		}
		ShipAssert(inputMeans.size() == 1 || inputMeans.size() == numChannels, "Invalid number of input means for Input layer!");
		if (inputMeans.size() == 1)
		{
			for (size_t i = 1; i < numChannels; ++i)
			{
				inputMeans.push_back(inputMeans[0]);
			}
		}

		replace(inputStDevsValue.begin(), inputStDevsValue.end(), ',', ' ');
		istringstream inputStDevsParser(inputStDevsValue);
		float stDev;
		while (inputStDevsParser >> stDev)
		{
			inputStDevs.push_back(stDev);
		}
		ShipAssert(inputStDevs.size() == 1 || inputStDevs.size() == numChannels, "Invalid number of input standard deviations for Input layer!");
		if (inputStDevs.size() == 1)
		{
			for (size_t i = 1; i < numChannels; ++i)
			{
				inputStDevs.push_back(inputStDevs[0]);
			}
		}
	}

	// Finding number of input batches.
	// TODO: We should change this whole logic, instead of looking just at the first tier we should look at the whole network.
	uint numInputBatches = 1;
	if (m_parsingMode == ParsingMode::Training)
	{
		uint nextTierSize = 1;
		bool parsedTierSize = false;
		string parallelismValue;
		bool parsedParallelism = false;
		for (const string& line : m_tiersLines[1])
		{
			if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, nextTierSize))
			{
				parsedTierSize = true;
			}
			else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
			{
				parsedParallelism = true;
			}
		}

		if (parsedParallelism && GetParallelismMode(parallelismValue) == ParallelismMode::Data)
		{
			numInputBatches = nextTierSize;
		}
	}

	InputLayer* inputLayer = new InputLayer(m_dataFolder, dataType, m_neuralNet->GetDeviceMemoryStreams(), numChannels, inputDataWidth, inputDataHeight,
		(uint)numInputBatches * m_batchSize, originalDataWidth, originalDataHeight, doRandomFlips, numInputBatches, normalizeInputs, inputMeans, inputStDevs,
		numTestPatches, testOnFlips);

	inputLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

	outLayerTier.push_back(inputLayer);
}

void ConfigurationParser::ParseConvolutionalLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize = c_tierSizeDefaultValue;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	uint numFilters;
	bool parsedNumFilters = false;
	uint filterWidth;
	bool parsedFilterWidth = false;
	uint filterHeight;
	bool parsedFilterHeight = false;
	uint paddingX = c_filterPaddingXDefaultValue;
	bool parsedPaddingX = false;
	uint paddingY = c_filterPaddingYDefaultValue;
	bool parsedPaddingY = false;
	uint stride = c_filterStrideDefaultValue;
	bool parsedStride = false;
	float weightsMomentum = c_weightsMomentumDefaultValue;
	bool parsedWeightsMomentum = false;
	float weightsDecay = c_weightsDecayDefaultValue;
	bool parsedWeightsDecay = false;
	float weightsStartingLR;
	bool parsedWeightsStartingLR = false;
	float weightsLRStep;
	bool parsedWeightsLRStep = false;
	float weightsLRFactor;
	bool parsedWeightsLRFactor = false;
	float biasesMomentum = c_weightsMomentumDefaultValue;
	bool parsedBiasesMomentum = false;
	float biasesDecay = c_weightsDecayDefaultValue;
	bool parsedBiasesDecay = false;
	float biasesStartingLR;
	bool parsedBiasesStartingLR = false;
	float biasesLRStep;
	bool parsedBiasesLRStep = false;
	float biasesLRFactor;
	bool parsedBiasesLRFactor = false;
	string activationTypeValue;
	ActivationType activationType;
	bool parsedActivationType = false;
	float activationAlpha = c_activationAlphaDefaultValue;
	bool parsedActivationAlpha = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	ResetWeightsInitializationParams();

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, tierSize))
		{
			parsedTierSize = true;
		}
		else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
		{
			parsedParallelism = true;
		}
		else if (!parsedNumFilters && ParseParameterUint(line, c_numFiltersParam, numFilters))
		{
			parsedNumFilters = true;
		}
		else if (!parsedFilterWidth && ParseParameterUint(line, c_filterWidthParam, filterWidth))
		{
			parsedFilterWidth = true;
		}
		else if (!parsedFilterHeight && ParseParameterUint(line, c_filterHeightParam, filterHeight))
		{
			parsedFilterHeight = true;
		}
		else if (!parsedPaddingX && ParseParameterUint(line, c_filterPaddingXParam, paddingX))
		{
			parsedPaddingX = true;
		}
		else if (!parsedPaddingY && ParseParameterUint(line, c_filterPaddingYParam, paddingY))
		{
			parsedPaddingY = true;
		}
		else if (!parsedStride && ParseParameterUint(line, c_filterStrideParam, stride))
		{
			parsedStride = true;
		}
		else if (!parsedWeightsMomentum && ParseParameterFloat(line, c_weightsMomentumParam, weightsMomentum))
		{
			parsedWeightsMomentum = true;
		}
		else if (!parsedWeightsDecay && ParseParameterFloat(line, c_weightsDecayParam, weightsDecay))
		{
			parsedWeightsDecay = true;
		}
		else if (!parsedWeightsStartingLR && ParseParameterFloat(line, c_weightsStartingLRParam, weightsStartingLR))
		{
			parsedWeightsStartingLR = true;
		}
		else if (!parsedWeightsLRStep && ParseParameterFloat(line, c_weightsLRStepParam, weightsLRStep))
		{
			parsedWeightsLRStep = true;
		}
		else if (!parsedWeightsLRFactor && ParseParameterFloat(line, c_weightsLRFactorParam, weightsLRFactor))
		{
			parsedWeightsLRFactor = true;
		}
		else if (!parsedBiasesMomentum && ParseParameterFloat(line, c_biasesMomentumParam, biasesMomentum))
		{
			parsedBiasesMomentum = true;
		}
		else if (!parsedBiasesDecay && ParseParameterFloat(line, c_biasesDecayParam, biasesDecay))
		{
			parsedBiasesDecay = true;
		}
		else if (!parsedBiasesStartingLR && ParseParameterFloat(line, c_biasesStartingLRParam, biasesStartingLR))
		{
			parsedBiasesStartingLR = true;
		}
		else if (!parsedBiasesLRStep && ParseParameterFloat(line, c_biasesLRStepParam, biasesLRStep))
		{
			parsedBiasesLRStep = true;
		}
		else if (!parsedBiasesLRFactor && ParseParameterFloat(line, c_biasesLRFactorParam, biasesLRFactor))
		{
			parsedBiasesLRFactor = true;
		}
		else if (!parsedActivationType && ParseParameterString(line, c_activationTypeParam, activationTypeValue))
		{
			parsedActivationType = true;
		}
		else if (!parsedActivationAlpha && ParseParameterFloat(line, c_activationAlphaParam, activationAlpha))
		{
			parsedActivationAlpha = true;
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
		else
		{
			ParseWeightsInitializationParams(line);
		}
	}

	ShipAssert(parsedNumFilters, "Can't parse number of filters for Convolutional layer!");
	ShipAssert(parsedFilterWidth, "Can't parse filter width for Convolutional layer!");
	ShipAssert(parsedFilterHeight, "Can't parse filter height for Convolutional layer!");
	ShipAssert(parsedWeightsStartingLR, "Can't parse weights starting learning rate for Convolutional layer!");
	ShipAssert(parsedWeightsLRStep, "Can't parse weights learning rate step for Convolutional layer!");
	ShipAssert(parsedWeightsLRFactor, "Can't parse weights learning rate factor for Convolutional layer!");
	ShipAssert(parsedBiasesStartingLR, "Can't parse biases starting learning rate for Convolutional layer!");
	ShipAssert(parsedBiasesLRStep, "Can't parse biases learning rate step for Convolutional layer!");
	ShipAssert(parsedBiasesLRFactor, "Can't parse biases learnign rate factor for Convolutional layer!");
	ShipAssert(parsedActivationType, "Can't parse activation type for Convolutional layer!");

	if (parsedParallelism)
	{
		parallelismMode = GetParallelismMode(parallelismValue);

		ShipAssert(parallelismMode != ParallelismMode::Data || tierSize > 1, "Can't have data parallelism in Convolutional layer tier of size 1!");
	}
	activationType = GetActivationType(activationTypeValue);

	if (m_parsingMode == ParsingMode::Prediction)
	{
		if (parallelismMode == ParallelismMode::Model)
		{
			// TODO: BUG HERE!!! Filter weights are sorted one weight per filter, so they can't be just merged together from multiple tiers.
			numFilters *= tierSize;
		}
		tierSize = c_tierSizeDefaultValue;
		parallelismMode = ParallelismMode::Model;
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		ConvolutionalLayer* convLayer = new ConvolutionalLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], m_neuralNet->GetCurandStatesBuffers()[layerIndex], layerIndex, tierSize,
			inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData, numFilters, filterWidth, filterHeight,
			inputNumChannels, weightsMomentum, weightsDecay, weightsLRStep, weightsStartingLR, weightsLRFactor, biasesMomentum, biasesDecay,
			biasesLRStep, biasesStartingLR, biasesLRFactor, paddingX, paddingY, stride, activationType, activationAlpha, holdsActivationGradients);

		convLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

		if (m_initializeLayersParams)
		{
			if (parallelismMode == ParallelismMode::Model || layerIndex == 0)
			{
				CudaAssert(cudaSetDevice(layerIndex));

				InitializeLayerWeights(convLayer);
				InitializeLayerBiases(convLayer);
			}
			else
			{
				convLayer->CopyWeightsFromLayer(static_cast<ConvolutionalLayer*>(outLayerTier[0]));
				convLayer->CopyBiasesFromLayer(static_cast<ConvolutionalLayer*>(outLayerTier[0]));
			}
		}

		for (Layer* prevLayer : prevLayers)
		{
			convLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(convLayer);
		}

		outLayerTier.push_back(convLayer);
	}

	CudaAssert(cudaSetDevice(0));
}

void ConfigurationParser::ParseResponseNormalizationLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize = c_tierSizeDefaultValue;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	uint depth;
	bool parsedDepth = false;
	float bias;
	bool parsedBias = false;
	float alphaCoeff;
	bool parsedAlphaCoeff = false;
	float betaCoeff;
	bool parsedBetaCoeff = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, tierSize))
		{
			parsedTierSize = true;
		}
		else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
		{
			parsedParallelism = true;
		}
		else if (!parsedDepth && ParseParameterUint(line, c_reNormDepthParam, depth))
		{
			parsedDepth = true;
		}
		else if (!parsedBias && ParseParameterFloat(line, c_reNormBiasParam, bias))
		{
			parsedBias = true;
		}
		else if (!parsedAlphaCoeff && ParseParameterFloat(line, c_reNormAlphaCoeffParam, alphaCoeff))
		{
			parsedAlphaCoeff = true;
		}
		else if (!parsedBetaCoeff && ParseParameterFloat(line, c_reNormBetaCoeffParam, betaCoeff))
		{
			parsedBetaCoeff = true;
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
	}

	ShipAssert(parsedDepth, "Can't parse depth for Response Normalization layer!");
	ShipAssert(depth > 0, "Depth should have positive integer value for Response Normalization layer!");
	ShipAssert(parsedBias, "Can't parse bias for Response Normalization layer!");
	ShipAssert(parsedAlphaCoeff, "Can't parse alpha coefficient for Response Normalization layer!");
	ShipAssert(parsedBetaCoeff, "Can't parse beta coefficient for Response Normalization layer!");

	if (parsedParallelism)
	{
		parallelismMode = GetParallelismMode(parallelismValue);

		ShipAssert(parallelismMode != ParallelismMode::Data || tierSize > 1, "Can't have data parallelism in Response Normalization layer tier of size 1!");
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		ResponseNormalizationLayer* reNormLayer = new ResponseNormalizationLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], layerIndex, tierSize, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount,
			holdsInputData, depth, bias, alphaCoeff, betaCoeff, holdsActivationGradients);

		reNormLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

		for (Layer* prevLayer : prevLayers)
		{
			reNormLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(reNormLayer);
		}

		outLayerTier.push_back(reNormLayer);
	}
}

void ConfigurationParser::ParseMaxPoolLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize = c_tierSizeDefaultValue;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	uint unitWidth;
	bool parsedUnitWidth = false;
	uint unitHeight;
	bool parsedUnitHeight = false;
	uint paddingX = c_filterPaddingXDefaultValue;
	bool parsedPaddingX = false;
	uint paddingY = c_filterPaddingYDefaultValue;
	bool parsedPaddingY = false;
	uint unitStride = c_filterStrideDefaultValue;
	bool parsedUnitStride = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, tierSize))
		{
			parsedTierSize = true;
		}
		else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
		{
			parsedParallelism = true;
		}
		else if (!parsedUnitWidth && ParseParameterUint(line, c_filterWidthParam, unitWidth))
		{
			parsedUnitWidth = true;
		}
		else if (!parsedUnitHeight && ParseParameterUint(line, c_filterHeightParam, unitHeight))
		{
			parsedUnitHeight = true;
		}
		else if (!parsedPaddingX && ParseParameterUint(line, c_filterPaddingXParam, paddingX))
		{
			parsedPaddingX = true;
		}
		else if (!parsedPaddingY && ParseParameterUint(line, c_filterPaddingYParam, paddingY))
		{
			parsedPaddingY = true;
		}
		else if (!parsedUnitStride && ParseParameterUint(line, c_filterStrideParam, unitStride))
		{
			parsedUnitStride = true;
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
	}

	ShipAssert(parsedUnitWidth, "Can't parse unit width for Max Pool layer!");
	ShipAssert(parsedUnitHeight, "Can't parse unit height for Max Pool layer!");

	if (parsedParallelism)
	{
		parallelismMode = GetParallelismMode(parallelismValue);

		ShipAssert(parallelismMode != ParallelismMode::Data || tierSize > 1, "Can't have data parallelism in Max Pool layer tier of size 1!");
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		MaxPoolLayer* maxPoolLayer = new MaxPoolLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], layerIndex, tierSize, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData,
			unitWidth, unitHeight, paddingX, paddingY, unitStride, holdsActivationGradients);

		maxPoolLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

		for (Layer* prevLayer : prevLayers)
		{
			maxPoolLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(maxPoolLayer);
		}

		outLayerTier.push_back(maxPoolLayer);
	}
}

void ConfigurationParser::ParseStandardLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize = c_tierSizeDefaultValue;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	uint numNeurons;
	bool parsedNumNeurons = false;
	float weightsMomentum = c_weightsMomentumDefaultValue;
	bool parsedWeightsMomentum = false;
	float weightsDecay = c_weightsDecayDefaultValue;
	bool parsedWeightsDecay = false;
	float weightsStartingLR;
	bool parsedWeightsStartingLR = false;
	float weightsLRStep;
	bool parsedWeightsLRStep = false;
	float weightsLRFactor;
	bool parsedWeightsLRFactor = false;
	float biasesMomentum = c_weightsMomentumDefaultValue;
	bool parsedBiasesMomentum = false;
	float biasesDecay = c_weightsDecayDefaultValue;
	bool parsedBiasesDecay = false;
	float biasesStartingLR;
	bool parsedBiasesStartingLR = false;
	float biasesLRStep;
	bool parsedBiasesLRStep = false;
	float biasesLRFactor;
	bool parsedBiasesLRFactor = false;
	string activationTypeValue;
	ActivationType activationType;
	bool parsedActivationType = false;
	float activationAlpha = c_activationAlphaDefaultValue;
	bool parsedActivationAlpha = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	ResetWeightsInitializationParams();

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, tierSize))
		{
			parsedTierSize = true;
		}
		else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
		{
			parsedParallelism = true;
		}
		else if (!parsedNumNeurons && ParseParameterUint(line, c_numNeuronsParam, numNeurons))
		{
			parsedNumNeurons = true;
		}
		else if (!parsedWeightsMomentum && ParseParameterFloat(line, c_weightsMomentumParam, weightsMomentum))
		{
			parsedWeightsMomentum = true;
		}
		else if (!parsedWeightsDecay && ParseParameterFloat(line, c_weightsDecayParam, weightsDecay))
		{
			parsedWeightsDecay = true;
		}
		else if (!parsedWeightsStartingLR && ParseParameterFloat(line, c_weightsStartingLRParam, weightsStartingLR))
		{
			parsedWeightsStartingLR = true;
		}
		else if (!parsedWeightsLRStep && ParseParameterFloat(line, c_weightsLRStepParam, weightsLRStep))
		{
			parsedWeightsLRStep = true;
		}
		else if (!parsedWeightsLRFactor && ParseParameterFloat(line, c_weightsLRFactorParam, weightsLRFactor))
		{
			parsedWeightsLRFactor = true;
		}
		else if (!parsedBiasesMomentum && ParseParameterFloat(line, c_biasesMomentumParam, biasesMomentum))
		{
			parsedBiasesMomentum = true;
		}
		else if (!parsedBiasesDecay && ParseParameterFloat(line, c_biasesDecayParam, biasesDecay))
		{
			parsedBiasesDecay = true;
		}
		else if (!parsedBiasesStartingLR && ParseParameterFloat(line, c_biasesStartingLRParam, biasesStartingLR))
		{
			parsedBiasesStartingLR = true;
		}
		else if (!parsedBiasesLRStep && ParseParameterFloat(line, c_biasesLRStepParam, biasesLRStep))
		{
			parsedBiasesLRStep = true;
		}
		else if (!parsedBiasesLRFactor && ParseParameterFloat(line, c_biasesLRFactorParam, biasesLRFactor))
		{
			parsedBiasesLRFactor = true;
		}
		else if (!parsedActivationType && ParseParameterString(line, c_activationTypeParam, activationTypeValue))
		{
			parsedActivationType = true;
		}
		else if (!parsedActivationAlpha && ParseParameterFloat(line, c_activationAlphaParam, activationAlpha))
		{
			parsedActivationAlpha = true;
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
		else
		{
			ParseWeightsInitializationParams(line);
		}
	}

	ShipAssert(parsedNumNeurons, "Can't parse number of neurons for Standard layer!");
	ShipAssert(parsedWeightsStartingLR, "Can't parse weights starting learning rate for Standard layer!");
	ShipAssert(parsedWeightsLRStep, "Can't parse weights learning rate step for Standard layer!");
	ShipAssert(parsedWeightsLRFactor, "Can't parse weights learning rate factor for Standard layer!");
	ShipAssert(parsedBiasesStartingLR, "Can't parse biases starting learning rate for Standard layer!");
	ShipAssert(parsedBiasesLRStep, "Can't parse biases learning rate step for Standard layer!");
	ShipAssert(parsedBiasesLRFactor, "Can't parse biases learnign rate factor for Standard layer!");
	ShipAssert(parsedActivationType, "Can't parse activation type for Standard layer!");

	if (parsedParallelism)
	{
		parallelismMode = GetParallelismMode(parallelismValue);

		ShipAssert(parallelismMode != ParallelismMode::Data || tierSize > 1, "Can't have data parallelism in Standard layer tier of size 1!");
	}
	activationType = GetActivationType(activationTypeValue);

	if (m_parsingMode == ParsingMode::Prediction)
	{
		if (parallelismMode == ParallelismMode::Model)
		{
			numNeurons *= tierSize;
		}
		tierSize = c_tierSizeDefaultValue;
		parallelismMode = ParallelismMode::Model;
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		StandardLayer* standardLayer = new StandardLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], m_neuralNet->GetCublasHandles()[layerIndex], m_neuralNet->GetCurandStatesBuffers()[layerIndex],
			layerIndex, tierSize, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData, numNeurons, weightsMomentum, weightsDecay,
			weightsLRStep, weightsStartingLR, weightsLRFactor, biasesMomentum, biasesDecay, biasesLRStep, biasesStartingLR, biasesLRFactor, activationType,
			activationAlpha, holdsActivationGradients);

		standardLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

		if (m_initializeLayersParams)
		{
			if (parallelismMode == ParallelismMode::Model || layerIndex == 0)
			{
				CudaAssert(cudaSetDevice(layerIndex));

				InitializeLayerWeights(standardLayer);
				InitializeLayerBiases(standardLayer);
			}
			else
			{
				standardLayer->CopyWeightsFromLayer(static_cast<StandardLayer*>(outLayerTier[0]));
				standardLayer->CopyBiasesFromLayer(static_cast<StandardLayer*>(outLayerTier[0]));
			}
		}

		for (Layer* prevLayer : prevLayers)
		{
			standardLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(standardLayer);
		}

		outLayerTier.push_back(standardLayer);
	}

	CudaAssert(cudaSetDevice(0));
}

void ConfigurationParser::ParseDropoutLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize = c_tierSizeDefaultValue;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	float dropProbability = c_dropProbabilityDefaultValue;
	bool parsedDropProbability = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		if (!parsedTierSize && ParseParameterUint(line, c_tierSizeParam, tierSize))
		{
			parsedTierSize = true;
		}
		else if (!parsedParallelism && ParseParameterString(line, c_parallelismParam, parallelismValue))
		{
			parsedParallelism = true;
		}
		else if (!parsedDropProbability && ParseParameterFloat(line, c_dropProbabilityParam, dropProbability))
		{
			parsedDropProbability = true;
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
	}

	if (parsedParallelism)
	{
		parallelismMode = GetParallelismMode(parallelismValue);

		ShipAssert(parallelismMode != ParallelismMode::Data || tierSize > 1, "Can't have data parallelism in Dropout layer tier of size 1!");
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		DropoutLayer* dropoutLayer = new DropoutLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], m_neuralNet->GetCurandStatesBuffers()[layerIndex], layerIndex, tierSize, inputNumChannels,
			inputDataWidth, inputDataHeight, inputDataCount, holdsInputData, dropProbability, false, holdsActivationGradients);

		dropoutLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

		for (Layer* prevLayer : prevLayers)
		{
			dropoutLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(dropoutLayer);
		}

		outLayerTier.push_back(dropoutLayer);
	}
}

void ConfigurationParser::ParseSoftMaxLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	string prevLayersParam;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		if (ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			break;
		}
	}

	uint tierSize = c_tierSizeDefaultValue;
	uint layerIndex = 0;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
	uint inputNumChannels;
	uint inputDataWidth;
	uint inputDataHeight;
	uint inputDataCount;
	bool holdsInputData;
	FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
	bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

	SoftMaxLayer* softMaxLayer = new SoftMaxLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
		m_neuralNet->GetDeviceMemoryStreams()[layerIndex], inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, holdsInputData);

	softMaxLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

	for (Layer* prevLayer : prevLayers)
	{
		softMaxLayer->AddPrevLayer(prevLayer);
		prevLayer->AddNextLayer(softMaxLayer);
	}

	outLayerTier.push_back(softMaxLayer);
}

void ConfigurationParser::ParseOutputLayerTier(vector<Layer*>& outLayerTier)
{
	string lossFunctionName;
	LossFunctionType lossFunction;
	bool parsedLossFunction = false;
	uint numGuesses;
	bool parsedNumGuesses = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines.back();
	for (const string& line : tierLines)
	{
		if (!parsedLossFunction && ParseParameterString(line, c_lossFunctionParam, lossFunctionName))
		{
			parsedLossFunction = true;
		}
		else if (!parsedNumGuesses && ParseParameterUint(line, c_numGuessesParam, numGuesses))
		{
			parsedNumGuesses = true;
		}
		else if (!parsedPrevLayers && ParseParameterString(line, c_prevLayersParam, prevLayersParam))
		{
			parsedPrevLayers = true;
		}
	}

	ShipAssert(parsedLossFunction, "Can't parse loss function for Output layer!");

	lossFunction = GetLossFunctionType(lossFunctionName);
	if (lossFunction == LossFunctionType::LogisticRegression)
	{
		// There can only be one guess for logistic regression.
		parsedNumGuesses = false;
	}

	uint tierSize = c_tierSizeDefaultValue;
	uint layerIndex = 0;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, m_tiersLines.size() - 2, prevLayersParam);
	uint inputNumChannels;
	uint inputDataWidth;
	uint inputDataHeight;
	uint inputDataCount;
	bool holdsInputData;
	FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);

	InputLayer* inputLayer = static_cast<InputLayer*>(m_layersTiers[0][0]);
	OutputLayer* outputLayer = new OutputLayer(m_neuralNet->GetDeviceCalculationStreams()[layerIndex], m_neuralNet->GetDeviceMemoryStreams()[layerIndex],
		inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, inputLayer->GetInputDataCount(), lossFunction, parsedNumGuesses, numGuesses,
		inputLayer->GetNumTestPasses());

	outputLayer->AllocateBuffers(m_parsingMode == ParsingMode::Training);

	for (Layer* prevLayer : prevLayers)
	{
		if (lossFunction == LossFunctionType::LogisticRegression)
		{
			ShipAssert(prevLayer->GetLayerType() == LayerType::Standard && prevLayer->GetActivationDataSize() == 1,
				"It is expected to have Standard layer with single neuron before Output layer in case of LogisticRegression loss!");
		}
		else if (lossFunction == LossFunctionType::CrossEntropy)
		{
			ShipAssert(prevLayer->GetLayerType() == LayerType::SoftMax, "It is expected to have SoftMax layer before Output layer in case of CrossEntropy loss!");
		}

		outputLayer->AddPrevLayer(prevLayer);
		prevLayer->AddNextLayer(outputLayer);
	}

	outLayerTier.push_back(outputLayer);
}

vector<Layer*> ConfigurationParser::ParseLayersTier(size_t tierIndex, LayerType tierLayerType)
{
	vector<Layer*> layerTier;

	if (tierLayerType == LayerType::Input)
	{
		ParseInputLayerTier(layerTier);
	}
	else if (tierLayerType == LayerType::Convolutional)
	{
		ParseConvolutionalLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::ResponseNormalization)
	{
		ParseResponseNormalizationLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::MaxPool)
	{
		ParseMaxPoolLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::Standard)
	{
		ParseStandardLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::Dropout)
	{
		ParseDropoutLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::SoftMax)
	{
		ParseSoftMaxLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::Output)
	{
		ParseOutputLayerTier(layerTier);
	}

	return layerTier;
}