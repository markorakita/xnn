// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Data parser abstract interface.
// Created: 11/26/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "../../utils/include/deftypes.cuh"

using namespace std;

// Resize modes.
enum class ResizeMode
{
	// Image is resized proportionately so that smaller dimension fits desired dimensions.
	ResizeToSmaller,

	// Image is resized proportionately so that larger dimension fits desired dimensions.
	ResizeToLarger,

	// Image is resized so that both dimensions fit desired dimensions.
	ResizeToFit
};

// Crop modes.
enum class CropMode
{
	CropLeft,
	CropTop,
	CropRight,
	CropBottom,
	CropCenter
};

class ImageData;
typedef struct CUstream_st* cudaStream_t;

class DataParser
{
public:
	// Destructor.
	virtual ~DataParser() {}

	// Loads image from file.
	virtual ImageData* LoadImage(string inputPath) = 0;

	// Saves image to file.
	virtual void SaveImage(const ImageData& image, string outputPath) = 0;

	// Resizes image to desired dimensions, using CUDA, on default stream.
	virtual ImageData* ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode) = 0;

	// Resizes image to desired dimensions, using CUDA.
	virtual ImageData* ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream) = 0;

	// Crops image to desired coordinates and dimensions, with option to flip.
	virtual ImageData* CropImage(const ImageData& image, uint startPixelX, uint startPixelY, uint desiredWidth, uint desiredHeight, bool flipCrop) = 0;

	// Resizes image to desired dimensions and crops it to be squared, using CUDA, on default stream.
	virtual ImageData* ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		CropMode cropMode) = 0;

	// Resizes image to desired dimensions and crops it to be squared, using CUDA.
	virtual ImageData* ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		CropMode cropMode, cudaStream_t stream) = 0;

	// Gets device buffer of last image that parser operated on.
	virtual uchar* GetLastImageDeviceBuffer() const = 0;
};