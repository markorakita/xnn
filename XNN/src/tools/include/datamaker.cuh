// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Prepares data for training or featurization.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "../../utils/include/deftypes.cuh"

using namespace std;

// Data making modes.
enum class DataMakerMode
{
	AlexNet,
	CentralCrops
};

class ImageData;
class DataParser;
typedef struct CUstream_st* cudaStream_t;

class DataMaker
{
private:
	// Data info file name.
	static const string c_dataInfoFileName;

	// Input folder path.
	string m_inputFolder;

	// Input data list file path, which can contain data names from input folder, or full data paths.
	string m_inputDataListFile;

	// Output folder path.
	string m_outputFolder;

	// Size of output images.
	uint m_imageSize;

	// Number of image channels.
	uint m_numImageChannels;

	// Device buffers for calculating mean image.
	vector<uint*> m_deviceMeanImageBuffers;

	// Device buffers length.
	uint m_deviceMeanImageBufferLength;

	// Count of images applied to calculate each mean image.
	vector<uint> m_meanImagesAppliedCounts;

	// Mutex used for output from multiple worker threads.
	mutex m_outputMutex;

	// Initializes data maker.
	void Initialize(DataMakerMode dataMakerMode);

	// Prepares part of data.
	void MakeDataPart(const vector<string>& data, string folder, int partIndex, DataMakerMode dataMakerMode);

	// Prepares image for training/testing of AlexNet.
	void PrepareImageForAlexNet(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream, int partIndex, string folder);

	// Prepares central crop of image.
	void PrepareImageCentralCrop(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream);

	// Prepares data in certain folder.
	void MakeData(string folder, DataMakerMode dataMakerMode);

	// Makes data info file.
	void MakeDataInfo();

public:
	// Argument parameters signatures.
	static const string c_inputFolderSignature;
	static const string c_inputDataListSignature;
	static const string c_outputFolderSignature;
	static const string c_imageSizeSignature;
	static const string c_numImageChannelsSignature;

	// Parameters default values.
	static const uint c_defaultImageSize;
	static const uint c_defaultNumOfImageChannels;

	// Labels file name.
	static const string c_labelsFileName;

	// Constructor.
	DataMaker();

	// Destructor.
	~DataMaker();

	// Parses arguments for data making.
	bool ParseArguments(int argc, char* argv[]);

	// Prepares data for training and testing of AlexNet.
	void MakeDataForAlexNet();

	// Prepares central crops of data for generic network training.
	void MakeDataCentralCrops();
};