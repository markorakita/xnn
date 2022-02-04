// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Prepares data for training.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/datamaker.cuh"

#include <algorithm>
#include <chrono>
#include <direct.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include <nppcore.h>

#include "../data/include/imagedata.cuh"
#include "../dataparsers/include/dataparser.cuh"
#include "../dataparsers/include/dataparserfactory.cuh"
#include "../utils/include/asserts.cuh"
#include "../utils/include/config.cuh"
#include "../utils/include/cudaasserts.cuh"
#include "../utils/include/cudahelper.cuh"
#include "../utils/include/utils.cuh"

using namespace std::chrono;

const string DataMaker::c_dataInfoFileName = "data info.txt";
const string DataMaker::c_labelsFileName = "labels.txt";

const string DataMaker::c_inputFolderSignature = "-inputfolder";
const string DataMaker::c_inputDataListSignature = "-inputdatalist";
const string DataMaker::c_outputFolderSignature = "-outputfolder";
const string DataMaker::c_imageSizeSignature = "-imagesize";
const string DataMaker::c_numImageChannelsSignature = "-numchannels";

const uint DataMaker::c_defaultImageSize = 256;
const uint DataMaker::c_defaultNumOfImageChannels = 3;

DataMaker::DataMaker()
{
	m_inputFolder = "";
	m_inputDataListFile = "";
	m_outputFolder = "";

	m_imageSize = c_defaultImageSize;
	m_numImageChannels = c_defaultNumOfImageChannels;

	m_deviceMeanImageBufferLength = 0;
}

DataMaker::~DataMaker()
{
	for (size_t i = 0; i < m_deviceMeanImageBuffers.size(); ++i)
	{
		CudaAssert(cudaFree(m_deviceMeanImageBuffers[i]));
	}
}

bool DataMaker::ParseArguments(int argc, char* argv[])
{
	bool parsedInputFolder = ParseArgument(argc, argv, c_inputFolderSignature, m_inputFolder);
	bool parsedInputDataList = ParseArgument(argc, argv, c_inputDataListSignature, m_inputDataListFile);
	if (!(parsedInputFolder || parsedInputDataList))
	{
		return false;
	}
	
	if (!ParseArgument(argc, argv, c_outputFolderSignature, m_outputFolder))
	{
		return false;
	}

	ParseArgument(argc, argv, c_imageSizeSignature, m_imageSize);
	ParseArgument(argc, argv, c_numImageChannelsSignature, m_numImageChannels);
	
	return true;
}

void DataMaker::Initialize(DataMakerMode dataMakerMode)
{
	ShipAssert(_mkdir(m_outputFolder.c_str()) == 0 || errno == EEXIST, "Problem creating directory \"" + m_outputFolder + "\".");

	if (dataMakerMode == DataMakerMode::AlexNet)
	{
		// Allocating memory for calculation of mean image.
		int coresPerGPU = Config::NUM_CPU_CORES / Config::NUM_GPUS;
		m_deviceMeanImageBufferLength = m_imageSize * m_imageSize * m_numImageChannels;
		int currentDevice;
		CudaAssert(cudaGetDevice(&currentDevice));
		for (int i = 0; i < Config::NUM_CPU_CORES; ++i)
		{
			CudaAssert(cudaSetDevice(i / coresPerGPU));
			m_deviceMeanImageBuffers.push_back(NULL);
			CudaAssert(cudaMalloc<uint>(&m_deviceMeanImageBuffers[i], m_deviceMeanImageBufferLength * sizeof(uint)));
			m_meanImagesAppliedCounts.push_back(0);
		}
		CudaAssert(cudaSetDevice(currentDevice));
	}
}

__global__ void AddImageToMeanKernel(uchar* deviceImageBuffer, uint* meanImageBuffer, uint imageBufferSize)
{
	for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < imageBufferSize; i += gridDim.x * blockDim.x)
	{
		meanImageBuffer[i] += deviceImageBuffer[i];
	}
}

void DataMaker::MakeDataPart(const vector<string>& data, string folder, int partIndex, DataMakerMode dataMakerMode)
{
	DebugAssert(partIndex < Config::NUM_CPU_CORES, "There is more data parts then CPU cores!");

	// Setting up the GPU.
	int coresPerGPU = Config::NUM_CPU_CORES / Config::NUM_GPUS;
	CudaAssert(cudaSetDevice(partIndex / coresPerGPU));
	cudaStream_t stream;
	CudaAssert(cudaStreamCreate(&stream));
	nppSetStream(stream);

	// Setting up data positions.
	size_t dataPerCore = data.size() / Config::NUM_CPU_CORES;
	size_t startIndex = partIndex * dataPerCore;
	size_t endIndex = partIndex < Config::NUM_CPU_CORES - 1 ? startIndex + dataPerCore : data.size();
	size_t dataToPrepareCnt = endIndex - startIndex;
	
	// Preparing data.
	DataParserFactory dataParserFactory;
	DataParser* dataParser;
	ImageData* image;
	uint doneCnt = 0;
	uint percentStep = (uint)max(dataPerCore / 20, (size_t)1);
	high_resolution_clock::time_point startTime = high_resolution_clock::now();
	for (size_t i = startIndex; i < endIndex; ++i)
	{
		// Finding file extension.
		string dataExtension = GetExtension(data[i]);
		ShipAssert(dataExtension != "", "Encountered data without extension! File name: " + data[i]);

		// Finding appropriate parser for that extension.
		dataParser = dataParserFactory.GetDataParser(dataExtension);
		ShipAssert(dataParser != NULL, "Can't create data parser for this data extension! File name: " + data[i]);

		// Parsing and preparing data.
		if (dataMakerMode == DataMakerMode::AlexNet)
		{
			image = dataParser->LoadImage(m_inputFolder + "\\" + folder + "\\" + data[i]);
		}
		else
		{
			if (m_inputFolder == "")
			{
				image = dataParser->LoadImage(data[i]);
			}
			else
			{
				image = dataParser->LoadImage(m_inputFolder + "\\" + data[i]);
			}
		}
		ShipAssert(image->GetNumOfChannels() == m_numImageChannels, "Encountered image with invalid number of channels! File name: " + data[i]);

		string imageName = GetFileName(data[i]);
		if (dataMakerMode == DataMakerMode::AlexNet)
		{
			PrepareImageForAlexNet(image, imageName, dataParser, stream, partIndex, folder);
		}
		else if (dataMakerMode == DataMakerMode::CentralCrops)
		{
			PrepareImageCentralCrop(image, imageName, dataParser, stream);
		}

		delete image;

		if (++doneCnt % percentStep == 0)
		{
			high_resolution_clock::time_point endTime = high_resolution_clock::now();
			long long durationInSeconds = duration_cast<seconds>(endTime - startTime).count();
			long long numHours = durationInSeconds / 3600;
			long long numMinutes = (durationInSeconds - numHours * 3600) / 60;
			long long numSeconds = durationInSeconds - numHours * 3600 - numMinutes * 60;
			uint preparedPercent = (uint)(ceil(doneCnt * 100.0 / dataToPrepareCnt));

			lock_guard<mutex> lock(m_outputMutex);
			cout << "datamaker " << partIndex + 1 << " prepared: " << doneCnt << "/" << dataToPrepareCnt <<
				" (" << preparedPercent << "%) " << folder << " samples (Took: " << numHours << "h " << numMinutes << "min " << numSeconds << "s)" << endl;
		}
	}

	// Cleanup.
	CudaAssert(cudaStreamDestroy(stream));
}

void DataMaker::PrepareImageForAlexNet(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream, int partIndex, string folder)
{
	ImageData* resizedImage = dataParser->ResizeImageWithCropCu(*image, m_imageSize, m_imageSize, ResizeMode::ResizeToSmaller, CropMode::CropCenter, stream);
	int numBlocks = Config::MAX_NUM_FULL_BLOCKS;
	int numThreads = Config::MAX_NUM_THREADS;
	ShipAssert(m_deviceMeanImageBufferLength == resizedImage->GetBufferSize() / sizeof(uchar), "Resized image has unexpected buffer size!");
	if (m_deviceMeanImageBufferLength <= Config::MAX_NUM_THREADS)
	{
		numBlocks = 1;
		numThreads = m_deviceMeanImageBufferLength;
	}
	else if (m_deviceMeanImageBufferLength < Config::MAX_NUM_FULL_BLOCKS * Config::MAX_NUM_THREADS)
	{
		numThreads = RoundUp(DivideUp(m_deviceMeanImageBufferLength, Config::MAX_NUM_FULL_BLOCKS), Config::WARP_SIZE);
	}
	LAUNCH_KERNEL_ASYNC(AddImageToMeanKernel, numBlocks, numThreads, stream)(dataParser->GetLastImageDeviceBuffer(), m_deviceMeanImageBuffers[partIndex],
		m_deviceMeanImageBufferLength);
	++m_meanImagesAppliedCounts[partIndex];
	
	dataParser->SaveImage(*resizedImage, m_outputFolder + "\\" + folder + "\\" + imageName);
	delete resizedImage;
}

void DataMaker::PrepareImageCentralCrop(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream)
{
	ImageData* resizedImage = dataParser->ResizeImageWithCropCu(*image, m_imageSize, m_imageSize, ResizeMode::ResizeToSmaller, CropMode::CropCenter, stream);
	dataParser->SaveImage(*resizedImage, m_outputFolder + "\\" + imageName);
	delete resizedImage;
}

void DataMaker::MakeData(string folder, DataMakerMode dataMakerMode)
{
	vector<string> images;
	if (dataMakerMode == DataMakerMode::AlexNet)
	{
		// Creating output folder.
		string outputFolder = m_outputFolder + "\\" + folder;
		ShipAssert(_mkdir(outputFolder.c_str()) == 0 || errno == EEXIST, "Problem creating directory \"" + outputFolder + "\".");

		// Labels file will be copied to output location.
		ifstream labelsFile(m_inputFolder + "\\" + folder + "\\" + c_labelsFileName);
		ofstream dest(outputFolder + "\\" + c_labelsFileName);
		string imageName;
		int label;
		while (labelsFile >> imageName >> label)
		{
			images.push_back(imageName);

			// Writing label info.
			dest << imageName << " " << label << endl;
		}
		dest.close();
		labelsFile.close();
	}
	else
	{
		ShipAssert(m_inputDataListFile != "", "Input data list file path is required for this data maker mode!");

		ifstream datalistFile(m_inputDataListFile);
		string imageNameOrPath;
		while (getline(datalistFile, imageNameOrPath))
		{
			images.push_back(imageNameOrPath);
		}
		datalistFile.close();
	}

	vector<thread> datamakerThreads;
#ifdef NDEBUG	
	int numThreads = Config::NUM_CPU_CORES;
#else
	int numThreads = 1;
#endif

	for (int i = 0; i < numThreads; ++i)
	{
		datamakerThreads.push_back(thread(&DataMaker::MakeDataPart, this, ref(images), folder, i, dataMakerMode));
	}
	for (int i = 0; i < numThreads; ++i)
	{
		datamakerThreads[i].join();
	}

	if (dataMakerMode == DataMakerMode::AlexNet)
	{
		cout << folder << " data prepared..." << endl << endl;
	}
}

void DataMaker::MakeDataInfo()
{
	// Allocating temporary buffer to hold aggregated data from device mean image buffers.
	uint* tempImageBuffer;
	size_t tempImageBufferLength = m_deviceMeanImageBufferLength * sizeof(uint);
	CudaAssert(cudaMallocHost<uint>(&tempImageBuffer, tempImageBufferLength));

	// Aggregating device mean image buffers and calculating mean pixels.
	vector<unsigned long long> meanPixelValues;
	for (size_t i = 0; i < m_numImageChannels; ++i)
	{
		meanPixelValues.push_back(0);
	}
	for (size_t i = 0; i < m_deviceMeanImageBuffers.size(); ++i)
	{
		CudaAssert(cudaMemcpy(tempImageBuffer, m_deviceMeanImageBuffers[i], tempImageBufferLength, cudaMemcpyDeviceToHost));
		for (size_t j = 0; j < m_deviceMeanImageBufferLength; ++j)
		{
			meanPixelValues[j % m_numImageChannels] += tempImageBuffer[j];
		}
	}
	CudaAssert(cudaFreeHost(tempImageBuffer));
	unsigned long long meanImagesAppliedCount = 0;
	for (size_t i = 0; i < m_meanImagesAppliedCounts.size(); ++i)
	{
		meanImagesAppliedCount += m_meanImagesAppliedCounts[i];
	}
	unsigned long long pixelsPerChannelApplied = (unsigned long long)m_imageSize * m_imageSize * meanImagesAppliedCount;
	for (size_t i = 0; i < m_numImageChannels; ++i)
	{
		meanPixelValues[i] /= pixelsPerChannelApplied;
		ShipAssert(meanPixelValues[i] <= 255, "Calculated incorrect mean value! Value: " + to_string(meanPixelValues[i]));
	}

	// Writing data info file.
	ofstream dest(m_outputFolder + "\\" + c_dataInfoFileName);
	dest << "Mean pixel values: " << meanPixelValues[0];
	for (size_t i = 1; i < m_numImageChannels; ++i)
	{
		dest << ", " << meanPixelValues[i];
	}
	dest << endl;
}

void PrintDataMakingTiming(string operationMessage, high_resolution_clock::time_point operationStartTime,
	high_resolution_clock::time_point operationEndTime)
{
	long long durationInSeconds = duration_cast<seconds>(operationEndTime - operationStartTime).count();
	long long numHours = durationInSeconds / 3600;
	long long numMinutes = (durationInSeconds - numHours * 3600) / 60;
	long long numSeconds = durationInSeconds - numHours * 3600 - numMinutes * 60;
	cout << operationMessage << "! (Took: " << numHours << "h " << numMinutes << "min " << numSeconds << "s)" << endl;
}

void DataMaker::MakeDataForAlexNet()
{
	cout << endl;

	DataMakerMode dataMakerMode = DataMakerMode::AlexNet;
	
	high_resolution_clock::time_point startTime = high_resolution_clock::now();
	Initialize(dataMakerMode);
	MakeData("train", dataMakerMode);
	MakeData("test", dataMakerMode);
	MakeDataInfo();
	high_resolution_clock::time_point endTime = high_resolution_clock::now();

	PrintDataMakingTiming("Data is prepared for training of AlexNet", startTime, endTime);
}

void DataMaker::MakeDataCentralCrops()
{
	cout << endl;

	DataMakerMode dataMakerMode = DataMakerMode::CentralCrops;

	high_resolution_clock::time_point startTime = high_resolution_clock::now();
	Initialize(dataMakerMode);
	MakeData("", dataMakerMode);
	high_resolution_clock::time_point endTime = high_resolution_clock::now();

	PrintDataMakingTiming("Data central crops are prepared", startTime, endTime);
}