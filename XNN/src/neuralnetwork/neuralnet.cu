// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/neuralnet.cuh"

#include <chrono>
#include <fstream>

#include <curand_kernel.h>

#include "layers/include/dropoutlayer.cuh"
#include "layers/include/inputlayer.cuh"
#include "layers/include/outputlayer.cuh"
#include "layers/include/weightslayer.cuh"
#include "../utils/include/cublasasserts.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/cudahelper.cuh"

const uint NeuralNet::c_numCurandBlocks = 96;
const uint NeuralNet::c_numCurandThreadsPerBlock = 128;

NeuralNet::NeuralNet(size_t maxNetworkTierSize)
{
	m_maxNetworkTierSize = maxNetworkTierSize;

	for (size_t tierLayer = 0; tierLayer < m_maxNetworkTierSize; ++tierLayer)
	{
		CudaAssert(cudaSetDevice((int)tierLayer));

		// Initialize calculation stream.
		cudaStream_t deviceCalculationStream;
		CudaAssert(cudaStreamCreateWithFlags(&deviceCalculationStream, cudaStreamNonBlocking));
		m_deviceCalculationStreams.push_back(deviceCalculationStream);

		// Initialize memory stream.
		cudaStream_t deviceMemoryStream;
		CudaAssert(cudaStreamCreateWithFlags(&deviceMemoryStream, cudaStreamNonBlocking));
		m_deviceMemoryStreams.push_back(deviceMemoryStream);

		// Initialize cuBLAS handles.
		cublasHandle_t cublasHandle;
		CudaCublasAssert(cublasCreate(&cublasHandle));
		m_cublasHandles.push_back(cublasHandle);

		// Initialize cuRAND state buffers.
		curandState* curandStateBuffer;
		CudaAssert(cudaMalloc<curandState>(&curandStateBuffer, (size_t)c_numCurandBlocks * c_numCurandThreadsPerBlock * sizeof(curandState)));
		InitCurandStatesBuffer(curandStateBuffer, deviceCalculationStream);
		m_curandStatesBuffers.push_back(curandStateBuffer);

		// We need to sync whole device since cublas uses stream 0 to create handles,
		// but this is called once per network so we don't care.
		CudaAssert(cudaDeviceSynchronize());
	}

	// Reverting back to default device.
	CudaAssert(cudaSetDevice(0));
}

/*
	Initializes one cuRAND state per thread.
*/
__global__ void InitializeCurandStates(curandState* curandStatesBuffer, unsigned long long seedValue)
{
	const uint c_stateIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// Initializing each state with different subsequence, to get more statistically uncorrelated sequences in different cuRAND states.
	curand_init(seedValue, c_stateIndex, 0, &curandStatesBuffer[c_stateIndex]);
}

void NeuralNet::InitCurandStatesBuffer(curandState* curandStatesBuffer, cudaStream_t deviceCalculationStream)
{
	dim3 blockDimensions(c_numCurandThreadsPerBlock);
	dim3 gridDimensions(c_numCurandBlocks);
	// Making it less likely for statistically correlated sequences of random values across different cuRAND state buffers,
	// since they are all initialized in approximately same time.
	unsigned long long seedValue = 2 * chrono::system_clock::now().time_since_epoch().count() + 1;
	LAUNCH_KERNEL_ASYNC(InitializeCurandStates, gridDimensions, blockDimensions, deviceCalculationStream)(curandStatesBuffer, seedValue);
	CudaAssert(cudaGetLastError());
}

NeuralNet::~NeuralNet()
{
	// Delete layers.
	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		for (size_t layerIndex = 0; layerIndex < m_layersTiers[tier].size(); ++layerIndex)
		{
			delete m_layersTiers[tier][layerIndex];
		}
	}

	// Destroy streams.
	for (size_t stream = 0; stream < m_deviceCalculationStreams.size(); ++stream)
	{
		CudaAssert(cudaStreamDestroy(m_deviceCalculationStreams[stream]));
		CudaAssert(cudaStreamDestroy(m_deviceMemoryStreams[stream]));
	}

	// Destroy cuBLAS handles.
	for (size_t handle = 0; handle < m_cublasHandles.size(); ++handle)
	{
		CudaCublasAssert(cublasDestroy(m_cublasHandles[handle]));
	}

	// Destroy cuRAND state buffers.
	for (size_t buffer = 0; buffer < m_curandStatesBuffers.size(); ++buffer)
	{
		CudaAssert(cudaFree(m_curandStatesBuffers[buffer]));
	}
}

InputLayer* NeuralNet::GetInputLayer() const
{
	return static_cast<InputLayer*>(m_layersTiers[0][0]);
}

OutputLayer* NeuralNet::GetOutputLayer() const
{
	return static_cast<OutputLayer*>(m_layersTiers.back()[0]);
}

void NeuralNet::SaveModel(string modelFile, bool saveUpdateBuffers)
{
	ofstream modelStream(modelFile, ofstream::out | ofstream::binary);

	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Convolutional || m_layersTiers[tier][0]->GetLayerType() == LayerType::Standard)
		{
			vector<WeightsLayer*> weightsLayers;
			if (m_layersTiers[tier][0]->GetParallelismMode() == ParallelismMode::Data)
			{
				weightsLayers.push_back(static_cast<WeightsLayer*>(m_layersTiers[tier][0]));
			}
			else
			{
				for (Layer* layer : m_layersTiers[tier])
				{
					weightsLayers.push_back(static_cast<WeightsLayer*>(layer));
				}
			}

			// Writing weights.
			float* tempWeightsBuffer;
			CudaAssert(cudaMallocHost<float>(&tempWeightsBuffer, weightsLayers[0]->GetWeightsBufferSize()));
			for (WeightsLayer* weightsLayer : weightsLayers)
			{
				CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
				CudaAssert(cudaMemcpy(tempWeightsBuffer, weightsLayer->GetWeightsBuffer(), weightsLayer->GetWeightsBufferSize(), cudaMemcpyDeviceToHost));
				modelStream.write(reinterpret_cast<const char*>(tempWeightsBuffer), weightsLayer->GetWeightsBufferSize());
			}
			CudaAssert(cudaFreeHost(tempWeightsBuffer));

			// Writing biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, weightsLayers[0]->GetBiasesBufferSize()));
			for (WeightsLayer* weightsLayer : weightsLayers)
			{
				CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
				CudaAssert(cudaMemcpy(tempBiasesBuffer, weightsLayer->GetBiasesBuffer(), weightsLayer->GetBiasesBufferSize(), cudaMemcpyDeviceToHost));
				modelStream.write(reinterpret_cast<const char*>(tempBiasesBuffer), weightsLayer->GetBiasesBufferSize());
			}
			CudaAssert(cudaFreeHost(tempBiasesBuffer));

			if (saveUpdateBuffers)
			{
				// Writing weights update buffers.
				float* tempWeightsUpdatesBuffer;
				CudaAssert(cudaMallocHost<float>(&tempWeightsUpdatesBuffer, weightsLayers[0]->GetWeightsBufferSize()));
				for (WeightsLayer* weightsLayer : weightsLayers)
				{
					CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
					CudaAssert(cudaMemcpy(tempWeightsUpdatesBuffer, weightsLayer->GetWeightsUpdateBuffer(), weightsLayer->GetWeightsBufferSize(), cudaMemcpyDeviceToHost));
					modelStream.write(reinterpret_cast<const char*>(tempWeightsUpdatesBuffer), weightsLayer->GetWeightsBufferSize());
				}
				CudaAssert(cudaFreeHost(tempWeightsUpdatesBuffer));

				// Writing biases update buffers.
				float* tempBiasesUpdatesBuffer;
				CudaAssert(cudaMallocHost<float>(&tempBiasesUpdatesBuffer, weightsLayers[0]->GetBiasesBufferSize()));
				for (WeightsLayer* weightsLayer : weightsLayers)
				{
					CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
					CudaAssert(cudaMemcpy(tempBiasesUpdatesBuffer, weightsLayer->GetBiasesUpdateBuffer(), weightsLayer->GetBiasesBufferSize(), cudaMemcpyDeviceToHost));
					modelStream.write(reinterpret_cast<const char*>(tempBiasesUpdatesBuffer), weightsLayer->GetBiasesBufferSize());
				}
				CudaAssert(cudaFreeHost(tempBiasesUpdatesBuffer));
			}
		}
	}

	CudaAssert(cudaSetDevice(0));

	modelStream.close();
}

void NeuralNet::SaveModelCheckpoint(string modelFile)
{
	SaveModel(modelFile, true);
}

void NeuralNet::SaveModelForPrediction(string modelFile)
{
	SaveModel(modelFile, false);
}

void NeuralNet::LoadModel(string modelFile, bool loadUpdateBuffers)
{
	ifstream modelStream(modelFile, ifstream::in | ifstream::binary);

	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Convolutional || m_layersTiers[tier][0]->GetLayerType() == LayerType::Standard)
		{
			vector<WeightsLayer*> weightsLayers;
			for (Layer* layer : m_layersTiers[tier])
			{
				weightsLayers.push_back(static_cast<WeightsLayer*>(layer));
			}

			// Reading weights.
			float* tempWeightsBuffer;
			CudaAssert(cudaMallocHost<float>(&tempWeightsBuffer, weightsLayers[0]->GetWeightsBufferSize()));
			if (weightsLayers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempWeightsBuffer), weightsLayers[0]->GetWeightsBufferSize());
				for (WeightsLayer* weightsLayer : weightsLayers)
				{
					CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
					weightsLayer->CopyWeightsFromHost(tempWeightsBuffer);
				}
			}
			else
			{
				for (WeightsLayer* weightsLayer : weightsLayers)
				{
					CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempWeightsBuffer), weightsLayer->GetWeightsBufferSize());
					weightsLayer->CopyWeightsFromHost(tempWeightsBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempWeightsBuffer));

			// Reading biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, weightsLayers[0]->GetBiasesBufferSize()));
			if (weightsLayers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), weightsLayers[0]->GetBiasesBufferSize());
				for (WeightsLayer* weightsLayer : weightsLayers)
				{
					CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
					weightsLayer->CopyBiasesFromHost(tempBiasesBuffer);
				}
			}
			else
			{
				for (WeightsLayer* weightsLayer : weightsLayers)
				{
					CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), weightsLayer->GetBiasesBufferSize());
					weightsLayer->CopyBiasesFromHost(tempBiasesBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempBiasesBuffer));

			if (loadUpdateBuffers)
			{
				// Reading weights update buffer.
				float* tempWeightsUpdateBuffer;
				CudaAssert(cudaMallocHost<float>(&tempWeightsUpdateBuffer, weightsLayers[0]->GetWeightsBufferSize()));
				if (weightsLayers[0]->GetParallelismMode() == ParallelismMode::Data)
				{
					modelStream.read(reinterpret_cast<char*>(tempWeightsUpdateBuffer), weightsLayers[0]->GetWeightsBufferSize());
					for (WeightsLayer* weightsLayer : weightsLayers)
					{
						CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
						weightsLayer->CopyWeightsUpdateFromHost(tempWeightsUpdateBuffer);
					}
				}
				else
				{
					for (WeightsLayer* weightsLayer : weightsLayers)
					{
						CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
						modelStream.read(reinterpret_cast<char*>(tempWeightsUpdateBuffer), weightsLayer->GetWeightsBufferSize());
						weightsLayer->CopyWeightsUpdateFromHost(tempWeightsUpdateBuffer);
					}
				}
				CudaAssert(cudaFreeHost(tempWeightsUpdateBuffer));

				// Reading biases update buffer.
				float* tempBiasesUpdateBuffer;
				CudaAssert(cudaMallocHost<float>(&tempBiasesUpdateBuffer, weightsLayers[0]->GetBiasesBufferSize()));
				if (weightsLayers[0]->GetParallelismMode() == ParallelismMode::Data)
				{
					modelStream.read(reinterpret_cast<char*>(tempBiasesUpdateBuffer), weightsLayers[0]->GetBiasesBufferSize());
					for (WeightsLayer* weightsLayer : weightsLayers)
					{
						CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
						weightsLayer->CopyBiasesUpdateFromHost(tempBiasesUpdateBuffer);
					}
				}
				else
				{
					for (WeightsLayer* weightsLayer : weightsLayers)
					{
						CudaAssert(cudaSetDevice(weightsLayer->GetIndexInTier()));
						modelStream.read(reinterpret_cast<char*>(tempBiasesUpdateBuffer), weightsLayer->GetBiasesBufferSize());
						weightsLayer->CopyBiasesUpdateFromHost(tempBiasesUpdateBuffer);
					}
				}
				CudaAssert(cudaFreeHost(tempBiasesUpdateBuffer));
			}
		}
	}

	CudaAssert(cudaSetDevice(0));

	modelStream.close();
}

void NeuralNet::LoadModelCheckpoint(string modelFile)
{
	LoadModel(modelFile, true);
}

void NeuralNet::LoadModelForPrediction(string modelFile)
{
	LoadModel(modelFile, false);
}