// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "../../utils/include/deftypes.cuh"

using namespace std;

class Layer;
class InputLayer;
class OutputLayer;

typedef struct cublasContext* cublasHandle_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct curandStateXORWOW curandState;

class NeuralNet
{
private:
	// Layers are organized into tiers.
	// Tier contains layers of same type which work in parallel.
	vector<vector<Layer*> > m_layersTiers;

	// Size of network tier with maximal size.
	size_t m_maxNetworkTierSize;

	// Streams this network uses for device calculations.
	vector<cudaStream_t> m_deviceCalculationStreams;

	// Streams this network uses for device memory operations.
	vector<cudaStream_t> m_deviceMemoryStreams;

	// Handles this network uses for cuBLAS operations.
	vector<cublasHandle_t> m_cublasHandles;

	// Buffers for cuRAND states this network uses for cuRAND operations.
	vector<curandState*> m_curandStatesBuffers;

	// Initializes states in cuRAND buffer.
	void InitCurandStatesBuffer(curandState* curandStatesBuffer, cudaStream_t deviceCalculationStream);

	// Saves trained network model to disk.
	void SaveModel(string modelFile, bool saveUpdateBuffers);

	// Loads trained network model from disk.
	void LoadModel(string modelFile, bool loadUpdateBuffers);

public:
	// Number of blocks to use for cuRAND operations.
	static const uint c_numCurandBlocks;

	// Number of threads to use for cuRAND operations.
	static const uint c_numCurandThreadsPerBlock;

	// Constructs network with specified capacity.
	NeuralNet(size_t maxNetworkTierSize);

	// Destructor.
	~NeuralNet();

	// Gets input layer of the network.
	InputLayer* GetInputLayer() const;

	// Gets output layer of the network.
	OutputLayer* GetOutputLayer() const;

	// Gets layer tiers.
	const vector<vector<Layer*> >& GetLayerTiers() const { return m_layersTiers; }

	// Gets streams this network uses for device calculations.
	const vector<cudaStream_t>& GetDeviceCalculationStreams() const { return m_deviceCalculationStreams; }

	// Gets streams this network uses for device memory operations.
	const vector<cudaStream_t>& GetDeviceMemoryStreams() const { return m_deviceMemoryStreams; }

	// Gets handles this network uses for cuBLAS operations.
	const vector<cublasHandle_t>& GetCublasHandles() const { return m_cublasHandles; }

	// Gets buffers for cuRAND states this network uses for cuRAND operations.
	const vector<curandState*>& GetCurandStatesBuffers() const { return m_curandStatesBuffers; }

	// Adds layer tier.
	void AddLayersTier(const vector<Layer*>& layerTier) { m_layersTiers.push_back(layerTier); }

	// Gets size of network tier with maximal size.
	size_t GetMaxNetworkTierSize() const { return m_maxNetworkTierSize; }

	// Saves trained network model checkpoint to disk.
	void SaveModelCheckpoint(string modelFile);

	// Saves trained network model for prediction to disk.
	void SaveModelForPrediction(string modelFile);

	// Loads saved network model checkpoint from disk.
	void LoadModelCheckpoint(string modelFile);

	// Loads saved network model for prediction from disk.
	void LoadModelForPrediction(string modelFile);
};