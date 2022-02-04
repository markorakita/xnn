// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network input layer.
// Created: 12/30/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/inputlayer.cuh"

#include <iostream>
#include <sstream>
#include <thread>

#include "../../data/include/imagedata.cuh"
#include "../../dataparsers/include/dataparser.cuh"
#include "../../dataparsers/include/dataparserfactory.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/config.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/utils.cuh"

InputLayer::InputLayer(string dataFolder, DataType dataType, const vector<cudaStream_t>& deviceMemoryStreams, uint inputNumChannels, uint inputDataWidth,
	uint inputDataHeight, uint inputDataCount, uint dataWidth, uint dataHeight, bool doRandomFlips, uint numInputBatches, bool normalizeInputs,
	const vector<float>& inputMeans, const vector<float>& inputStDevs, uint numTestPatches, bool testOnFlips)
{
	m_layerType = LayerType::Input;
	m_parallelismMode = ParallelismMode::Model;
	m_dataFolder = dataFolder;
	m_dataType = dataType;
	m_deviceMemoryStreams = deviceMemoryStreams;
	m_indexInTier = 0;
	m_tierSize = 1;
	m_numInputBatches = numInputBatches;

	m_normalizeInputs = normalizeInputs;
	m_inputMeans = inputMeans;
	m_inputStDevs = inputStDevs;

	m_numTestPatches = numTestPatches;
	m_testOnFlips = testOnFlips;
	m_numTestPasses = m_numTestPatches * (m_testOnFlips ? 2 : 1);
	m_testPassCounter = 0;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_dataWidth = dataWidth;
	m_dataHeight = dataHeight;
	m_doRandomFlips = doRandomFlips;
	m_holdsInputData = true;

	if (dataType == DataType::Image)
	{
		if (m_dataWidth != m_inputDataWidth)
		{
			m_cropPositionXGenerator = default_random_engine((uint)chrono::system_clock::now().time_since_epoch().count());
			m_cropPositionXDistribution = uniform_int_distribution<uint>(0, m_dataWidth - m_inputDataWidth);
		}
		if (m_dataHeight != m_inputDataHeight)
		{
			m_cropPositionYGenerator = default_random_engine((uint)chrono::system_clock::now().time_since_epoch().count());
			m_cropPositionYDistribution = uniform_int_distribution<uint>(0, m_dataHeight - m_inputDataHeight);
		}
		if (m_doRandomFlips)
		{
			m_cropFlipGenerator = default_random_engine((uint)chrono::system_clock::now().time_since_epoch().count());
			m_cropFlipDistribution = uniform_int_distribution<uint>(1, 100);
		}
	}

	m_inputBufferSize = (size_t)m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);

	m_holdsActivationGradients = false;
}

void InputLayer::AllocateBuffers(bool allocateTrainBuffers)
{
	CudaAssert(cudaSetDevice(0));

	// Allocating input data buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize, cudaHostAllocPortable));

	// Allocating activation data buffers.
	if (m_numInputBatches > 1)
	{
		uint dataPerInput = m_inputDataCount / m_numInputBatches;
		for (uint i = 0; i < m_numInputBatches; ++i)
		{
			uint dataPerCurrInput = dataPerInput;
			if (i < m_inputDataCount % m_numInputBatches)
			{
				++dataPerCurrInput;
			}
			m_activationDataCounts.push_back(dataPerCurrInput);

			CudaAssert(cudaSetDevice((int)i));
			float* activationDataBuffer;
			size_t activationDataBufferSize = (size_t)dataPerCurrInput * m_inputNumChannels * m_inputDataSize * sizeof(float);
			CudaAssert(cudaMalloc<float>(&activationDataBuffer, activationDataBufferSize));
			m_memoryConsumptionSize += activationDataBufferSize;
			m_activationDataBuffers.push_back(activationDataBuffer);
		}

		// Reverting device back to 0.
		CudaAssert(cudaSetDevice(0));
	}
	else
	{
		m_activationBufferSize = m_inputBufferSize;
		CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));
		m_memoryConsumptionSize += m_activationBufferSize;

		m_activationDataBuffers.push_back(m_activationDataBuffer);
		m_activationDataCounts.push_back(m_inputDataCount);
	}
}

InputLayer::~InputLayer()
{
	if (m_inputDataBuffer != NULL)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
		m_inputDataBuffer = NULL;
	}

	if (m_numInputBatches > 1)
	{
		for (size_t i = 0; i < m_activationDataBuffers.size(); ++i)
		{
			CudaAssert(cudaFree(m_activationDataBuffers[i]));
		}

		// Doing this so it doesn't try to free it in the layer default destructor.
		m_activationDataBuffer = NULL;
	}
}

void InputLayer::SetDataFilesToLoad(const vector<string>& dataFiles, PropagationMode propagationMode)
{
	m_dataFilesToLoad = dataFiles;
	m_propagationMode = propagationMode;

	// Reinitialize layer if needed.
	if (m_dataFilesToLoad.size() != m_inputDataCount)
	{
		Reinitialize((uint)m_dataFilesToLoad.size());

		uint dataPerInput = m_inputDataCount / m_numInputBatches;
		if (m_numInputBatches > 1)
		{
			for (size_t i = 0; i < m_numInputBatches; ++i)
			{
				m_activationDataCounts[i] = dataPerInput;
				if (i < m_inputDataCount % m_numInputBatches)
				{
					++m_activationDataCounts[i];
				}
			}
		}
		else
		{
			m_activationDataCounts[0] = m_inputDataCount;
		}
	}
}

void InputLayer::CalculatePatch(uint& cropPositionX, uint numPatchesX, uint patchX, uint& cropPositionY, uint numPatchesY, uint patchY)
{
	cropPositionX = (patchX - 1) * (m_dataWidth - m_inputDataWidth) / (numPatchesX - 1);
	cropPositionY = (patchY - 1) * (m_dataHeight - m_inputDataHeight) / (numPatchesY - 1);
}

void InputLayer::CalculateTestPatchPosition(uint& cropPositionX, uint& cropPositionY, bool& flip)
{
	++m_testPassCounter;

	flip = false;
	if (m_testOnFlips)
	{
		flip = m_testPassCounter > m_numTestPatches;
	}

	uint pass = m_testPassCounter;
	if (flip)
	{
		pass -= m_numTestPatches;
	}

	if (m_numTestPatches == 1)
	{
		CalculatePatch(cropPositionX, 3, 2, cropPositionY, 3, 2);
	}
	else if (m_numTestPatches == 2)
	{
		CalculatePatch(cropPositionX, 2, pass, cropPositionY, 3, 2);
	}
	else if (m_numTestPatches == 3)
	{
		CalculatePatch(cropPositionX, 3, pass, cropPositionY, 3, 2);
	}
	else if (m_numTestPatches == 4)
	{
		if (pass <= 2)
		{
			CalculatePatch(cropPositionX, 2, pass, cropPositionY, 2, 1);
		}
		else
		{
			CalculatePatch(cropPositionX, 2, pass - 2, cropPositionY, 2, 2);
		}
	}
	else if (m_numTestPatches == 5)
	{
		if (pass <= 2)
		{
			CalculatePatch(cropPositionX, 2, pass, cropPositionY, 2, 1);
		}
		else if (pass <= 4)
		{
			CalculatePatch(cropPositionX, 2, pass - 2, cropPositionY, 2, 2);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, 2, cropPositionY, 3, 2);
		}
	}
	else if (m_numTestPatches == 6)
	{
		if (pass <= 3)
		{
			CalculatePatch(cropPositionX, 3, pass, cropPositionY, 2, 1);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, pass - 3, cropPositionY, 2, 2);
		}
	}
	else if (m_numTestPatches == 7)
	{
		if (pass <= 3)
		{
			CalculatePatch(cropPositionX, 3, pass, cropPositionY, 2, 1);
		}
		else if (pass <= 6)
		{
			CalculatePatch(cropPositionX, 3, pass - 3, cropPositionY, 2, 2);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, 2, cropPositionY, 3, 2);
		}
	}
	else if (m_numTestPatches == 8)
	{
		if (pass <= 4)
		{
			CalculatePatch(cropPositionX, 4, pass, cropPositionY, 2, 1);
		}
		else
		{
			CalculatePatch(cropPositionX, 4, pass - 4, cropPositionY, 2, 2);
		}
	}
	else if (m_numTestPatches == 9)
	{
		if (pass <= 3)
		{
			CalculatePatch(cropPositionX, 3, pass, cropPositionY, 3, 1);
		}
		else if (pass <= 6)
		{
			CalculatePatch(cropPositionX, 3, pass - 3, cropPositionY, 3, 2);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, pass - 6, cropPositionY, 3, 3);
		}
	}
	else
	{
		ShipAssert(false, "Currently not supported!");
	}

	if (m_testPassCounter == m_numTestPasses)
	{
		m_testPassCounter = 0;
	}
}

void InputLayer::SetupDataPositions(int partIndex, size_t inputBatchIndex, size_t& startIndex, size_t& endIndex, float** inputDataBuffer, vector<string>& dataFilesToLoad)
{
	DebugAssert(partIndex < Config::NUM_DATA_LOAD_CPU_CORES, "There is more data parts than CPU cores!");

	// Setting up data positions.
	auto beginIterator = m_dataFilesToLoad.begin();
	for (size_t i = 0; i < inputBatchIndex; ++i)
	{
		beginIterator += m_activationDataCounts[i];
	}
	auto endIterator = beginIterator + m_activationDataCounts[inputBatchIndex];
	dataFilesToLoad = vector<string>(beginIterator, endIterator);
	size_t dataPerCore = dataFilesToLoad.size() / Config::NUM_DATA_LOAD_CPU_CORES;
	startIndex = partIndex * dataPerCore;
	endIndex = partIndex < Config::NUM_DATA_LOAD_CPU_CORES - 1 ? startIndex + dataPerCore : dataFilesToLoad.size();
	*inputDataBuffer = m_inputDataBuffer + (beginIterator - m_dataFilesToLoad.begin()) * m_inputNumChannels * m_inputDataSize;
}

void InputLayer::LoadImageInputsPart(int partIndex, size_t inputBatchIndex, uint cropPositionX, uint cropPositionY, bool flip)
{
	// Setting data positions.
	size_t startIndex;
	size_t endIndex;
	float* inputDataBuffer;
	vector<string> dataFilesToLoad;
	SetupDataPositions(partIndex, inputBatchIndex, startIndex, endIndex, &inputDataBuffer, dataFilesToLoad);

	// Preparing data.
	DataParserFactory dataParserFactory;
	DataParser* dataParser;
	ImageData* image;
	ImageData* inputImage;
	for (size_t i = startIndex; i < endIndex; ++i)
	{
		// Finding file extension.
		string dataExtension = GetExtension(dataFilesToLoad[i]);
		ShipAssert(dataExtension != "", "Encountered data without extension! File name: " + dataFilesToLoad[i]);

		// Finding appropriate parser for that extension.
		dataParser = dataParserFactory.GetDataParser(dataExtension);
		ShipAssert(dataParser != NULL, "Can't create data parser for this data extension! File name: " + dataFilesToLoad[i]);

		// Parsing image.
		// TODO: We should store some vector of loaded images per inputs part, and then only for first test pass we load images, and then for next passes we just crop them differently.
		image = dataParser->LoadImage(m_dataFolder + "\\" + dataFilesToLoad[i]);
		ShipAssert(image->GetNumOfChannels() == m_inputNumChannels, "Encountered image with invalid number of channels! File name: " + dataFilesToLoad[i]);
		
		// Cropping image.
		if (m_dataWidth == m_inputDataWidth && m_dataHeight == m_inputDataHeight && !m_doRandomFlips)
		{
			inputImage = image;
		}
		else
		{
			inputImage = dataParser->CropImage(*image, cropPositionX, cropPositionY, m_inputDataWidth, m_inputDataHeight, flip);
			delete image;
		}
		
		// Copying image data to buffer.
		uchar* croppedImageRowMajorPixels = inputImage->GetRowMajorPixels();
		uint totalPixelsPerChannel = m_activationDataCounts[inputBatchIndex] * m_inputDataSize;
		size_t inputImageBufferLength = inputImage->GetBufferSize() / sizeof(uchar);
		for (size_t pixel = 0; pixel < inputImageBufferLength; ++pixel)
		{
			size_t channel = pixel % m_inputNumChannels;

			float pixelValue = (float)croppedImageRowMajorPixels[pixel];
			if (m_normalizeInputs)
			{
				pixelValue = (pixelValue - m_inputMeans[channel]) / m_inputStDevs[channel];
			}

			// Each column contains pixels of one image, sorted by channel.
			inputDataBuffer[i + (pixel / m_inputNumChannels) * m_activationDataCounts[inputBatchIndex] + channel * totalPixelsPerChannel] = pixelValue;
		}

		delete inputImage;
	}
}

void InputLayer::LoadTextInputsPart(int partIndex, size_t inputBatchIndex)
{
	// Setting data positions.
	size_t startIndex;
	size_t endIndex;
	float* inputDataBuffer;
	vector<string> dataFilesToLoad;
	SetupDataPositions(partIndex, inputBatchIndex, startIndex, endIndex, &inputDataBuffer, dataFilesToLoad);

	// Parsing data.
	for (size_t i = startIndex; i < endIndex; ++i)
	{
		istringstream dataParser(dataFilesToLoad[i]);
		float inputFeature;
		for (size_t j = 0; j < m_inputDataSize; ++j)
		{
			dataParser >> inputFeature;

			if (m_normalizeInputs)
			{
				inputFeature = (inputFeature - m_inputMeans[0]) / m_inputStDevs[0];
			}

			inputDataBuffer[i + j * m_activationDataCounts[inputBatchIndex]] = inputFeature;
		}
	}
}

void InputLayer::LoadInputs()
{
	DebugAssert(!m_dataFilesToLoad.empty(), "Data files to load must be set first!");

	if (m_dataType == DataType::Image)
	{
		uint cropPositionX = 0;
		uint cropPositionY = 0;
		bool flip = false;
		if (m_propagationMode == PropagationMode::Test)
		{
			CalculateTestPatchPosition(cropPositionX, cropPositionY, flip);
		}

		// Loading inputs in batches.
		for (size_t inputBatchIndex = 0; inputBatchIndex < m_numInputBatches; ++inputBatchIndex)
		{
			vector<thread> dataLoadThreads;
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				if (m_propagationMode == PropagationMode::Train)
				{
					if (m_dataWidth != m_inputDataWidth)
					{
						cropPositionX = m_cropPositionXDistribution(m_cropPositionXGenerator);
					}
					if (m_dataHeight != m_inputDataHeight)
					{
						cropPositionY = m_cropPositionYDistribution(m_cropPositionYGenerator);
					}
					if (m_doRandomFlips)
					{
						flip = m_cropFlipDistribution(m_cropFlipGenerator) % 2 == 0;
					}
				}

				dataLoadThreads.push_back(thread(&InputLayer::LoadImageInputsPart, this, i, inputBatchIndex, cropPositionX, cropPositionY, flip));
			}
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				dataLoadThreads[i].join();
			}
		}
	}
	else
	{
		// Loading inputs in batches.
		for (size_t inputBatchIndex = 0; inputBatchIndex < m_numInputBatches; ++inputBatchIndex)
		{
			vector<thread> dataLoadThreads;
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				dataLoadThreads.push_back(thread(&InputLayer::LoadTextInputsPart, this, i, inputBatchIndex));
			}
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				dataLoadThreads[i].join();
			}
		}
	}
}

void InputLayer::DoForwardProp(PropagationMode propagationMode)
{
	if (m_numInputBatches > 1)
	{
		// Forwarding loaded input data to each of activation buffers.
		uint inputOffset = 0;
		for (size_t i = 0; i < m_activationDataBuffers.size(); ++i)
		{
			CudaAssert(cudaSetDevice((int)i));
			uint activationDataBufferLength = m_activationDataCounts[i] * m_inputNumChannels * m_inputDataSize;
			CudaAssert(cudaMemcpyAsync(m_activationDataBuffers[i], m_inputDataBuffer + inputOffset, activationDataBufferLength * sizeof(float),
				cudaMemcpyHostToDevice, m_deviceMemoryStreams[i]));
			CudaAssert(cudaStreamSynchronize(m_deviceMemoryStreams[i]));
			inputOffset += activationDataBufferLength;
		}

		// Reverting device back to 0.
		CudaAssert(cudaSetDevice(0));
	}
	else
	{
		CudaAssert(cudaMemcpyAsync(m_activationDataBuffer, m_inputDataBuffer, m_inputBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStreams[0]));
		CudaAssert(cudaStreamSynchronize(m_deviceMemoryStreams[0]));
	}
}

void InputLayer::DoBackwardProp()
{
	ShipAssert(false, "Shouldn't backpropagate on input layer!");
}