// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network input layer.
// Created: 12/30/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <random>
#include <string>
#include <vector>

#include "layer.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;

// Data types.
enum class DataType
{
	Image,
	Text
};

/*
	Input layer loads data for the network training.

	Input and output data buffers are matrices with these specifications:
	    num_columns = data_count
		num_rows    = pixels_count (pixels are ordered per channel, R pixels first, then G pixels, then B)
*/
class InputLayer : public Layer
{
private:
	// Device streams for memory operations.
	vector<cudaStream_t> m_deviceMemoryStreams;

	// Folder with data for training.
	string m_dataFolder;

	// Data type.
	DataType m_dataType;

	// Width of data samples.
	uint m_dataWidth;

	// Height of data samples.
	uint m_dataHeight;

	// Should we randomly flip input data.
	bool m_doRandomFlips;

	// Seed engine for generating random crop horizontal positions.
	default_random_engine m_cropPositionXGenerator;

	// Random distribution for generating random crop horizontal positions.
	uniform_int_distribution<uint> m_cropPositionXDistribution;

	// Seed engine for generating random crop vertical positions.
	default_random_engine m_cropPositionYGenerator;

	// Random distribution for generating random crop vertical positions.
	uniform_int_distribution<uint> m_cropPositionYDistribution;

	// Seed engine for generating random crop flip decisions.
	default_random_engine m_cropFlipGenerator;

	// Random distribution for generating random crop flip decisions.
	uniform_int_distribution<uint> m_cropFlipDistribution;

	// Data files to load.
	vector<string> m_dataFilesToLoad;

	// Number of input batches to generate, depending on number of layers in next tier and their parallelism.
	uint m_numInputBatches;

	// Propagation mode.
	PropagationMode m_propagationMode;

	// Should inputs be normalized.
	bool m_normalizeInputs;

	// Input means for input normalization.
	vector<float> m_inputMeans;

	// Input standard deviations for input normalization.
	vector<float> m_inputStDevs;

	// Activation data buffers, one buffer for each of layers in next tier.
	vector<float*> m_activationDataBuffers;

	// Activation data counts, one count for each of layers in next tier.
	vector<uint> m_activationDataCounts;

	// Number of patches we extract from the image during test to get average prediction.
	uint m_numTestPatches;

	// Should we also test on flipped versions of patches.
	bool m_testOnFlips;

	// Number of test passes for one image.
	uint m_numTestPasses;

	// Counter of test passes.
	uint m_testPassCounter;

	// Calculates patch position.
	void CalculatePatch(uint& cropPositionX, uint numPatchesX, uint patchX, uint& cropPositionY, uint numPatchesY, uint patchY);

	// Calculates position from which to crop patch for test pass.
	void CalculateTestPatchPosition(uint& cropPositionX, uint& cropPositionY, bool& flip);

	// Setups data positions for load.
	void SetupDataPositions(int partIndex, size_t inputBatchIndex, size_t& startIndex, size_t& endIndex, float** inputDataBuffer, vector<string>& dataFilesToLoad);

	// Loads part of input image files to input data buffer which position depends of how many inputs we are loading.
	void LoadImageInputsPart(int partIndex, size_t inputBatchIndex, uint cropPositionX, uint cropPositionY, bool flip);

	// Loads part of input text files to input data buffer which position depends of how many inputs we are loading.
	void LoadTextInputsPart(int partIndex, size_t inputBatchIndex);

public:
	// Constructor.
	InputLayer(string dataFolder, DataType dataType, const vector<cudaStream_t>& deviceMemoryStreams, uint inputNumChannels, uint inputDataWidth,
		uint inputDataHeight, uint inputDataCount, uint dataWidth, uint dataHeight, bool doRandomFlips, uint numInputBatches, bool normalizeInputs,
		const vector<float>& inputMeans, const vector<float>& inputStDevs, uint numTestPatches, bool testOnFlips);

	// Allocates internal data buffers used in this layer.
	virtual void AllocateBuffers(bool allocateTrainBuffers);

	// Destructor.
	virtual ~InputLayer();

	// Gets input data type.
	DataType GetDataType() const { return m_dataType; }

	// Sets data files to load.
	void SetDataFilesToLoad(const vector<string>& dataFiles, PropagationMode propagationMode);

	// Gets activation data buffer, dedicated to layer with specified index in next tier.
	float* GetActivationDataBuffer(uint indexInTier) { return m_activationDataBuffers[indexInTier]; }

	// Gets input data count.
	uint GetInputDataCount() const { return m_inputDataCount; }

	// Gets activation data counts, dedicated to layer with specified index in next tier.
	uint GetActivationDataCount(uint indexInTier) { return m_activationDataCounts[indexInTier]; }

	// Gets number of test passes for one image.
	uint GetNumTestPasses() const { return m_numTestPasses; }

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};