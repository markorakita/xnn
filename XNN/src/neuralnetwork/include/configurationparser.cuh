// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural networks configuration parser.
// Created: 03/17/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "../../utils/include/deftypes.cuh"

using namespace std;

// Parsing mode.
enum class ParsingMode
{
	Training,
	Prediction
};

enum class ActivationType;
enum class DataType;
enum class LayerType;
enum class LossFunctionType;
enum class ParallelismMode;
class Layer;
class WeightsLayer;
class NeuralNet;

class ConfigurationParser
{
private:
	friend class TestConfigurationParser;

	// Parameters initialization mode.
	enum class ParamInitializationMode
	{
		Constant,
		Uniform,
		Gaussian,
		Xavier,
		He
	};

	// Configuration parameters names.
	static const string c_layerTypeParam;
	static const string c_tierSizeParam;
	static const string c_dataTypeParam;
	static const string c_numChannelsParam;
	static const string c_inputDataWidthParam;
	static const string c_inputDataHeightParam;
	static const string c_originalDataWidthParam;
	static const string c_originalDataHeightParam;
	static const string c_doRandomFlipsParam;
	static const string c_normalizeInputsParam;
	static const string c_inputMeansParam;
	static const string c_inputStDevsParam;
	static const string c_numTestPatchesParam;
	static const string c_testOnFlipsParam;
	static const string c_parallelismParam;
	static const string c_prevLayersParam;
	static const string c_weightsInitializationParam;
	static const string c_weightsInitialValueParam;
	static const string c_weightsRangeStartParam;
	static const string c_weightsRangeEndParam;
	static const string c_weightsMeanParam;
	static const string c_weightsStdDevParam;
	static const string c_biasesInitializationParam;
	static const string c_biasesInitialValueParam;
	static const string c_biasesRangeStartParam;
	static const string c_biasesRangeEndParam;
	static const string c_biasesMeanParam;
	static const string c_biasesStdDevParam;
	static const string c_numFiltersParam;
	static const string c_filterWidthParam;
	static const string c_filterHeightParam;
	static const string c_filterPaddingXParam;
	static const string c_filterPaddingYParam;
	static const string c_filterStrideParam;
	static const string c_weightsMomentumParam;
	static const string c_weightsDecayParam;
	static const string c_weightsStartingLRParam;
	static const string c_weightsLRStepParam;
	static const string c_weightsLRFactorParam;
	static const string c_biasesMomentumParam;
	static const string c_biasesDecayParam;
	static const string c_biasesStartingLRParam;
	static const string c_biasesLRStepParam;
	static const string c_biasesLRFactorParam;
	static const string c_activationTypeParam;
	static const string c_activationAlphaParam;
	static const string c_reNormDepthParam;
	static const string c_reNormBiasParam;
	static const string c_reNormAlphaCoeffParam;
	static const string c_reNormBetaCoeffParam;
	static const string c_numNeuronsParam;
	static const string c_lossFunctionParam;
	static const string c_numGuessesParam;
	static const string c_dropProbabilityParam;

	// Configuration parameter option values.
	static const string c_prevLayersOptionAll;

	// Default parameters values.
	static const uint c_tierSizeDefaultValue;
	static const uint c_numChannelsDefaultValue;
	static const uint c_inputDataHeightDefaultValue;
	static const bool c_doRandomFlipsDefaultValue;
	static const bool c_normalizeInputsDefaultValue;
	static const uint c_numTestPatchesDefaultValue;
	static const bool c_testOnFlipsDefaultValue;
	static const uint c_filterPaddingXDefaultValue;
	static const uint c_filterPaddingYDefaultValue;
	static const uint c_filterStrideDefaultValue;
	static const float c_weightsDefaultInitialValue;
	static const float c_weightsRangeStartDefaultValue;
	static const float c_weightsRangeEndDefaultValue;
	static const float c_weightsMeanDefaultValue;
	static const float c_weightsStdDevDefaultValue;
	static const float c_biasesDefaultInitialValue;
	static const float c_weightsMomentumDefaultValue;
	static const float c_weightsDecayDefaultValue;
	static const float c_dropProbabilityDefaultValue;
	static const float c_activationAlphaDefaultValue;

	// Weights initialization parameters.
	ParamInitializationMode m_weightsInitializationMode;
	bool m_parsedWeightsInitialization;
	float m_weightsInitialValue;
	bool m_parsedWeightsInitialValue;
	float m_weightsRangeStart;
	bool m_parsedWeightsRangeStart;
	float m_weightsRangeEnd;
	bool m_parsedWeightsRangeEnd;
	float m_weightsMean;
	bool m_parsedWeightsMean;
	float m_weightsStdDev;
	bool m_parsedWeightsStdDev;
	ParamInitializationMode m_biasesInitializationMode;
	bool m_parsedBiasesInitialization;
	float m_biasesInitialValue;
	bool m_parsedBiasesInitialValue;
	float m_biasesRangeStart;
	bool m_parsedBiasesRangeStart;
	float m_biasesRangeEnd;
	bool m_parsedBiasesRangeEnd;
	float m_biasesMean;
	bool m_parsedBiasesMean;
	float m_biasesStdDev;
	bool m_parsedBiasesStdDev;

	// Neural network which is parsed.
	NeuralNet* m_neuralNet;

	// Parsing mode.
	ParsingMode m_parsingMode;

	// Folder with data for training.
	string m_dataFolder;

	// Training data batch size.
	uint m_batchSize;

	// Should we initialize parameters in layers.
	bool m_initializeLayersParams;

	// Parsed layers tiers.
	vector<vector<Layer*> > m_layersTiers;

	// Parsed tiers' lines.
	vector<vector<string> > m_tiersLines;

	// Size of network tier with maximal size.
	size_t m_maxNetworkTierSize;

	// Trims line.
	string TrimLine(const string& line);

	// Parses tiers' lines.
	void ParseTierLines(const string& configurationFile);

	// Finds size of network tier with maximal size.
	void FindMaxNetworkTierSize();

	// Gets parameter value string from line.
	bool GetParameterValueStrFromLine(const string& line, const string& parameterName, string& parameterValueStr);

	// Parses unsigned int parameter from line, returns true if successful.
	bool ParseParameterUint(const string& line, const string& parameterName, uint& parameterValue);

	// Parses float parameter from line, returns true if successful.
	bool ParseParameterFloat(const string& line, const string& parameterName, float& parameterValue);

	// Parses boolean parameter from line, returns true if successful.
	bool ParseParameterBool(const string& line, const string& parameterName, bool& parameterValue);

	// Parses string parameter from line, returns true if successful.
	bool ParseParameterString(const string& line, const string& parameterName, string& parameterValue);

	// Gets layer type based on layer type name.
	LayerType GetLayerType(const string& layerTypeName);

	// Gets activation type based on activation type name.
	ActivationType GetActivationType(const string& activationTypeName);

	// Gets loss function type based on loss function name.
	LossFunctionType GetLossFunctionType(const string& lossFunctionName);

	// Gets data type based on data type name.
	DataType GetDataType(const string& dataTypeName);

	// Gets parallelism mode based on parallelism mode name.
	ParallelismMode GetParallelismMode(const string& parallelismModeName);

	// Gets parameters initialization mode based on mode name.
	ParamInitializationMode GetParamInitializationMode(const string& paramInitializationModeName);

	// Parses layers tiers.
	void ParseLayersTiers();

	// Parses layers tier with specified type.
	vector<Layer*> ParseLayersTier(size_t tierIndex, LayerType tierLayerType);

	// Finds previous layers of current tier.
	vector<Layer*> FindPrevLayers(ParallelismMode currTierParallelismMode, uint layerIndexInTier, uint currTierSize, size_t prevTierIndex, const string& prevLayersParam);

	// Finds if layers in current tier should hold activation gradients.
	bool ShouldHoldActivationGradients(ParallelismMode currTierParallelismMode, uint currTierSize, size_t currTierIndex, uint layerIndexInTier);

	// Finds input parameters based on previous layers.
	void FindInputParams(const vector<Layer*>& prevLayers, uint layerIndexInTier, uint tierSize, ParallelismMode parallelismMode, uint& inputNumChannels,
		uint& inputDataWidth, uint& inputDataHeight, uint& inputDataCount, bool& holdsInputData);

	// Resets weights initialization parameters to default values.
	void ResetWeightsInitializationParams();

	// Parses weights initialization parameters.
	void ParseWeightsInitializationParams(const string& line);

	// Initializes layer's weights.
	void InitializeLayerWeights(WeightsLayer* weightsLayer);

	// Initializes layer's biases.
	void InitializeLayerBiases(WeightsLayer* weightsLayer);

	// Parses input layer tier.
	void ParseInputLayerTier(vector<Layer*>& outLayerTier);

	// Parses convolutional layer tier.
	void ParseConvolutionalLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses response normalization layer tier.
	void ParseResponseNormalizationLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses max pool layer tier.
	void ParseMaxPoolLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses standard layer tier.
	void ParseStandardLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses dropout layer tier.
	void ParseDropoutLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses soft max layer tier.
	void ParseSoftMaxLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses output layer tier.
	void ParseOutputLayerTier(vector<Layer*>& outLayerTier);

public:
	// Constructor, just to initialize member variables to please the IntelliSense.
	ConfigurationParser();

	// Parses network configuration and creates network.
	NeuralNet* ParseNetworkFromConfiguration(ParsingMode parsingMode, const string& configurationFile, const string& dataFolder, uint batchSize, bool initializeLayersParams);
};