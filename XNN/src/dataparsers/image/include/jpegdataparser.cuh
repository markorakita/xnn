// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Jpeg data parser.
// Created: 11/26/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "../../include/dataparser.cuh"
#include "../../../utils/include/deftypes.cuh"

using namespace std;

class ImageData;
typedef struct CUstream_st* cudaStream_t;

class JpegDataParser : public DataParser
{
private:
	// Operations source image device buffer.
	uchar* m_deviceOpSrcBuffer;

	// Operations source image device buffer size.
	size_t m_deviceOpSrcBufferSize;

	// Operations destination image device buffer.
	uchar* m_deviceOpDestBuffer;

	// Operations destination image device buffer size.
	size_t m_deviceOpDestBufferSize;

	// Device buffer of last image that parser operated on.
	// It will always be one of the two buffers above.
	uchar* m_lastImageDeviceBuffer;

	// Allocates memory for operations source image device buffer if it is not allocated, or if its size is less than needed.
	void AllocSrcDeviceMemoryIfNeeded(size_t imageBufferSize);

	// Allocates memory for operations destination image device buffer if it is not allocated, or if its size is less than needed.
	void AllocDestDeviceMemoryIfNeeded(size_t imageBufferSize);

	// Transfers image data to operations device source buffer.
	void TransferImageToOpDeviceBuffer(const ImageData& image, cudaStream_t stream = 0);

	// Calculates resize dimensions.
	void CalculateResizeDimensions(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		int interpolationMode, double& scaleX, double& scaleY, int& destImageWidth, int& destImageHeight);

	// Resizes image from source device buffer to destination device buffer.
	void ResizeOnDeviceBuffers(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream,
		int& destImageWidth, int& destImageHeight);

	// Crops image from destination device buffer to source device buffer.
	void CropOnInvertedDeviceBuffers(uint imageWidth, uint imageHeight, uint numOfChannels, uint croppedSize, uint startPixelX, uint startPixelY,
		cudaStream_t stream);

	// Rectangularly crops image from destination device buffer to source device buffer.
	void CropOnInvertedDeviceBuffers(uint imageWidth, uint imageHeight, uint numOfChannels, uint croppedSize, uint edgePadding, CropMode cropMode,
		cudaStream_t stream);

public:
	// Constructor.
	JpegDataParser();

	// Destructor.
	virtual ~JpegDataParser();

	// Loads image from file.
	virtual ImageData* LoadImage(string inputPath);

	// Saves image to file.
	virtual void SaveImage(const ImageData& image, string outputPath);

	// Resizes image to desired dimensions, using CUDA, on default stream.
	virtual ImageData* ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode);

	// Resizes image to desired dimensions, using CUDA.
	virtual ImageData* ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream);

	// Crops image to desired coordinates and dimensions, with option to flip.
	virtual ImageData* CropImage(const ImageData& image, uint startPixelX, uint startPixelY, uint desiredWidth, uint desiredHeight, bool flipCrop);

	// Resizes image to desired dimensions and crops it to be squared, using CUDA, on default stream.
	virtual ImageData* ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		CropMode cropMode);

	// Resizes image to desired dimensions and crops it to be squared, using CUDA.
	virtual ImageData* ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		CropMode cropMode, cudaStream_t stream);

	// Gets device buffer of last image that parser operated on.
	virtual uchar* GetLastImageDeviceBuffer() const { return m_lastImageDeviceBuffer; }
};