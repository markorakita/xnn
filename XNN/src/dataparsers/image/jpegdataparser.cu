// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Jpeg data parser.
// Created: 11/26/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/jpegdataparser.cuh"

#ifdef _WIN32
	#define XMD_H /* This prevents jpeglib to redefine INT32 */
#endif

// cstdio is needed for jpeglib
#include <cstdio>
#include <jpeglib.h>
#include <npp.h>

#include "../../data/include/imagedata.cuh"
#include "../../utils/include/asserts.cuh"
#include "../../utils/include/config.cuh"
#include "../../utils/include/cudaasserts.cuh"
#include "../../utils/include/cudahelper.cuh"
#include "../../utils/include/nppasserts.cuh"
#include "../../utils/include/utils.cuh"

JpegDataParser::JpegDataParser()
{
	m_deviceOpSrcBuffer = NULL;
	m_deviceOpSrcBufferSize = 0;

	m_deviceOpDestBuffer = NULL;
	m_deviceOpDestBufferSize = 0;

	m_lastImageDeviceBuffer = NULL;
}

JpegDataParser::~JpegDataParser()
{
	CudaAssert(cudaFree(m_deviceOpSrcBuffer));
	CudaAssert(cudaFree(m_deviceOpDestBuffer));
	// m_lastImageDeviceBuffer will always be one of the two buffers above, so no need to free it.
}

void JpegDataParser::AllocSrcDeviceMemoryIfNeeded(size_t imageBufferSize)
{
	if (imageBufferSize <= m_deviceOpSrcBufferSize)
	{
		return;
	}

	m_deviceOpSrcBufferSize = imageBufferSize;
	if (m_deviceOpSrcBuffer != NULL)
	{
		CudaAssert(cudaFree(m_deviceOpSrcBuffer));
	}

	CudaAssert(cudaMalloc<uchar>(&m_deviceOpSrcBuffer, m_deviceOpSrcBufferSize));
}

void JpegDataParser::AllocDestDeviceMemoryIfNeeded(size_t imageBufferSize)
{
	if (imageBufferSize <= m_deviceOpDestBufferSize)
	{
		return;
	}

	m_deviceOpDestBufferSize = imageBufferSize;
	if (m_deviceOpDestBuffer != NULL)
	{
		CudaAssert(cudaFree(m_deviceOpDestBuffer));
	}

	CudaAssert(cudaMalloc<uchar>(&m_deviceOpDestBuffer, m_deviceOpDestBufferSize));
}

ImageData* JpegDataParser::LoadImage(string inputPath)
{
	// Opening image file.
	FILE *imageFile;
	ShipAssert((fopen_s(&imageFile, inputPath.c_str(), "rb")) == 0, "Cannot open jpeg image file \"" + inputPath + "\".");

	// Parsing image.
	struct jpeg_decompress_struct imageInfo;
	struct jpeg_error_mgr errorHandler;
	imageInfo.err = jpeg_std_error(&errorHandler);
	jpeg_create_decompress(&imageInfo);
	jpeg_stdio_src(&imageInfo, imageFile);
	jpeg_read_header(&imageInfo, TRUE);
	switch (imageInfo.jpeg_color_space)
	{
		case JCS_GRAYSCALE:
		case JCS_RGB:
		case JCS_YCbCr:
			imageInfo.out_color_space = JCS_RGB;
			break;
		case JCS_CMYK:
		case JCS_YCCK:
			imageInfo.out_color_space = JCS_CMYK;
			break;
		default:
			ShipAssert(false, "Unsupported jpeg color format: " + to_string(imageInfo.out_color_space));
	}

	jpeg_start_decompress(&imageInfo);

	uint imageWidth = imageInfo.output_width;
	uint imageHeight = imageInfo.output_height;
	uint imageNumChannels = imageInfo.output_components;
	ImageData* image = new ImageData(imageWidth, imageHeight, imageNumChannels);
	uint imageStride = image->GetStride();
	uchar* rowMajorPixels = image->GetRowMajorPixels();
	while (imageInfo.output_scanline < imageHeight)
	{
		JSAMPROW writeLoc = &rowMajorPixels[imageInfo.output_scanline * imageStride];
		jpeg_read_scanlines(&imageInfo, &writeLoc, 1);
	}
	
	// Cleaning up.
	jpeg_finish_decompress(&imageInfo);
	jpeg_destroy_decompress(&imageInfo);
	fclose(imageFile);

	// Ensuring right color format.
	if (imageInfo.out_color_space == JCS_RGB)
	{
		ShipAssert(imageInfo.output_components == 3, "RGB should have 3 components, encountered: " + to_string(imageInfo.output_components));
		return image;
	}
	else if (imageInfo.out_color_space == JCS_CMYK)
	{
		ShipAssert(imageInfo.output_components == 4, "CMYK should have 4 components, encountered: " + to_string(imageInfo.output_components));

		// Transforming into RGB
		const uint rgbNumChannels = 3;
		ImageData* rgbImage = new ImageData(imageWidth, imageHeight, rgbNumChannels);
		uchar* rgbRowMajorPixels = rgbImage->GetRowMajorPixels();
		uint rgbImageStride = rgbImage->GetStride();
		for (uint row = 0; row < image->GetHeight(); ++row)
		{
			for (uint col = 0; col < image->GetWidth(); ++col)
			{
				uint pixelPos = row * imageStride + col * image->GetNumOfChannels();
				uchar cyan = rowMajorPixels[pixelPos];
				uchar magenta = rowMajorPixels[pixelPos + 1];
				uchar yellow = rowMajorPixels[pixelPos + 2];
				uchar key = rowMajorPixels[pixelPos + 3];

				// This is incorrect but pretty close aproximation. Resulting picture will be different,
				// but picture semantic will be intact, and picture semantic is only thing that matters for recognition.
				uint rgbPixelPos = row * rgbImageStride + col * rgbNumChannels;
				rgbRowMajorPixels[rgbPixelPos] = cyan * key / 255;
				rgbRowMajorPixels[rgbPixelPos + 1] = magenta * key / 255;
				rgbRowMajorPixels[rgbPixelPos + 2] = yellow * key / 255;
			}
		}

		delete image;
		return rgbImage;
	}
	else
	{
		ShipAssert(false, "Unexpected output color space: " + to_string(imageInfo.out_color_space));
		return NULL;
	}
}

void JpegDataParser::SaveImage(const ImageData& image, string outputPath)
{
	// Creating output image file.
	FILE *imageFile;
	ShipAssert((fopen_s(&imageFile, outputPath.c_str(), "wb")) == 0, "Cannot create jpeg image file \"" + outputPath + "\".");

	// Packing image.
	const int imageQuality = 95;
	uint imageWidth = image.GetWidth();
	uint imageHeight = image.GetHeight();
	uint imageNumChannels = image.GetNumOfChannels();
	uint imageStride = image.GetStride();
	struct jpeg_compress_struct imageInfo;
	struct jpeg_error_mgr errorHandler;
	imageInfo.err = jpeg_std_error(&errorHandler);
	jpeg_create_compress(&imageInfo);
	jpeg_stdio_dest(&imageInfo, imageFile);
	imageInfo.image_width = imageWidth;
	imageInfo.image_height = imageHeight;
	imageInfo.input_components = imageNumChannels;
	imageInfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&imageInfo);
	jpeg_set_quality(&imageInfo, imageQuality, TRUE);
	jpeg_start_compress(&imageInfo, TRUE);

	// Copying image data.
	uchar* rowMajorPixels = image.GetRowMajorPixels();
	while (imageInfo.next_scanline < imageInfo.image_height)
	{
		JSAMPROW readLoc = &rowMajorPixels[imageInfo.next_scanline * imageStride];
		jpeg_write_scanlines(&imageInfo, &readLoc, 1);
	}	

	// Cleaning up.
	jpeg_finish_compress(&imageInfo);
	jpeg_destroy_compress(&imageInfo);
	fclose(imageFile);
}

void JpegDataParser::TransferImageToOpDeviceBuffer(const ImageData& image, cudaStream_t stream)
{
	size_t imageBufferSize = image.GetBufferSize();
	AllocSrcDeviceMemoryIfNeeded(imageBufferSize);

	CudaAssert(cudaMemcpyAsync(m_deviceOpSrcBuffer, image.GetRowMajorPixels(), imageBufferSize, cudaMemcpyHostToDevice, stream));
}

void JpegDataParser::CalculateResizeDimensions(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
	int interpolationMode, double& scaleX, double& scaleY, int& destImageWidth, int& destImageHeight)
{
	NppiRect srcImageRect;
	srcImageRect.x = 0;
	srcImageRect.y = 0;
	srcImageRect.width = image.GetWidth();
	srcImageRect.height = image.GetHeight();
	
	// Small delta to add during division to avoid getting scales that later when multiplied by width/height give width/height smaller by 1.
	double divDelta = 0.01;
	switch (resizeMode)
	{
		case ResizeMode::ResizeToSmaller:
			scaleX = scaleY = image.GetWidth() <= image.GetHeight() ?
				(desiredWidth + divDelta) / image.GetWidth() :
				(desiredHeight + divDelta) / image.GetHeight();
			break;
		case ResizeMode::ResizeToLarger:
			scaleX = scaleY = image.GetWidth() >= image.GetHeight() ?
				(desiredWidth + divDelta) / image.GetWidth() :
				(desiredHeight + divDelta) / image.GetHeight();
			break;
		case ResizeMode::ResizeToFit:
			scaleX = (desiredWidth + divDelta) / image.GetWidth();
			scaleY = (desiredHeight + divDelta) / image.GetHeight();
			break;
		default:
			ShipAssert(false, "Unknown resize mode encountered!");
	}

	NppiRect destImageRect;
	nppiGetResizeRect(srcImageRect, &destImageRect, scaleX, scaleY, 0, 0, interpolationMode);
	destImageWidth = destImageRect.width;
	destImageHeight = destImageRect.height;

	switch (resizeMode)
	{
		case ResizeMode::ResizeToSmaller:
			ShipAssert((destImageRect.width <= destImageRect.height && destImageRect.width == desiredWidth) ||
				(destImageRect.height < destImageRect.width && destImageRect.height == desiredHeight), "Calculated wrong crop dimensions!");
			break;
		case ResizeMode::ResizeToLarger:
			ShipAssert((destImageRect.width >= destImageRect.height && destImageRect.width == desiredWidth) ||
				(destImageRect.height > destImageRect.width && destImageRect.height == desiredHeight), "Calculated wrong crop dimensions!");
			break;
		case ResizeMode::ResizeToFit:
			ShipAssert(destImageRect.width == desiredWidth && destImageRect.height == desiredHeight, "Calculated wrong crop dimensions!");
	}
}

void JpegDataParser::ResizeOnDeviceBuffers(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream,
	int& destImageWidth, int& destImageHeight)
{
	// Setting up the device source buffer and transfering image data to it.
	TransferImageToOpDeviceBuffer(image, stream);

	// Calculating dimensions.
	double scaleX, scaleY;
	NppiInterpolationMode interpolationMode = NppiInterpolationMode::NPPI_INTER_LANCZOS;
	CalculateResizeDimensions(image, desiredWidth, desiredHeight, resizeMode, interpolationMode, scaleX, scaleY, destImageWidth, destImageHeight);

	// Setting up the device destination buffer
	uint destImageBufferSize = (size_t)image.GetNumOfChannels() * destImageHeight * destImageWidth * sizeof(uchar);
	AllocDestDeviceMemoryIfNeeded(destImageBufferSize);

	NppiSize srcImageSize;
	srcImageSize.width = image.GetWidth();
	srcImageSize.height = image.GetHeight();
	NppiRect srcImageRect;
	srcImageRect.x = 0;
	srcImageRect.y = 0;
	srcImageRect.width = srcImageSize.width;
	srcImageRect.height = srcImageSize.height;
	NppiRect destImageRect;
	destImageRect.x = 0;
	destImageRect.y = 0;
	destImageRect.width = destImageWidth;
	destImageRect.height = destImageHeight;
	
	CudaNppAssert(nppiResizeSqrPixel_8u_C3R(m_deviceOpSrcBuffer, srcImageSize, srcImageSize.width * image.GetNumOfChannels(), srcImageRect,
		m_deviceOpDestBuffer, destImageRect.width * image.GetNumOfChannels(), destImageRect, scaleX, scaleY, 0, 0, interpolationMode));
}

ImageData* JpegDataParser::ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode)
{
	return ResizeImageCu(image, desiredWidth, desiredHeight, resizeMode, 0);
}

ImageData* JpegDataParser::ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream)
{
	int destImageWidth, destImageHeight;
	ResizeOnDeviceBuffers(image, desiredWidth, desiredHeight, resizeMode, stream, destImageWidth, destImageHeight);
	
	// Creating output image.
	ImageData* resizedImage = new ImageData(destImageWidth, destImageHeight, image.GetNumOfChannels());
	CudaAssert(cudaMemcpyAsync(resizedImage->GetRowMajorPixels(), m_deviceOpDestBuffer, resizedImage->GetBufferSize(), cudaMemcpyDeviceToHost, stream));
	m_lastImageDeviceBuffer = m_deviceOpDestBuffer;
	CudaAssert(cudaStreamSynchronize(stream));

	return resizedImage;
}

ImageData* JpegDataParser::CropImage(const ImageData& image, uint startPixelX, uint startPixelY, uint desiredWidth, uint desiredHeight, bool flipCrop)
{
	uint endPixelX = startPixelX + desiredWidth;
	uint endPixelY = startPixelY + desiredHeight;
	uint numOfChannels = image.GetNumOfChannels();
	uint imageStride = image.GetStride();
	uint croppedImageStride = numOfChannels * desiredWidth;

	ShipAssert(endPixelX <= image.GetWidth(), "Can't crop jpeg image, bad crop width dimensions.");
	ShipAssert(endPixelY <= image.GetHeight(), "Can't crop jpeg image, bad crop height dimensions.");
	
	ImageData* croppedImage = new ImageData(desiredWidth, desiredHeight, numOfChannels);
	uchar* imageRowMajorPixels = image.GetRowMajorPixels();
	uchar* croppedImageRowMajorPixels = croppedImage->GetRowMajorPixels();

	// Redundant code to avoid checking for flip deep inside the loop which hurts the performance.
	if (flipCrop)
	{
		for (uint row = startPixelY; row < endPixelY; ++row)
		{
			const uint croppedImageRowOffset = (row - startPixelY) * croppedImageStride;
			const uint imageRowOffset = row * imageStride;
			for (uint col = startPixelX; col < endPixelX; ++col)
			{
				const uint croppedImageTotalOffset = croppedImageRowOffset + (col - startPixelX) * numOfChannels;
				const uint imageTotalOffset = imageRowOffset + (startPixelX + endPixelX - col - 1) * numOfChannels;
				for (uint channel = 0; channel < numOfChannels; ++channel)
				{
					croppedImageRowMajorPixels[croppedImageTotalOffset + channel] = imageRowMajorPixels[imageTotalOffset + channel];
				}
			}
		}
	}
	else
	{
		for (uint row = startPixelY; row < endPixelY; ++row)
		{
			const uint croppedImageRowOffset = (row - startPixelY) * croppedImageStride;
			const uint imageRowOffset = row * imageStride;
			for (uint col = startPixelX; col < endPixelX; ++col)
			{
				const uint croppedImageTotalOffset = croppedImageRowOffset + (col - startPixelX) * numOfChannels;
				const uint imageTotalOffset = imageRowOffset + col * numOfChannels;
				for (uint channel = 0; channel < numOfChannels; ++channel)
				{
					croppedImageRowMajorPixels[croppedImageTotalOffset + channel] = imageRowMajorPixels[imageTotalOffset + channel];
				}
			}
		}
	}

	return croppedImage;
}

// Crop kernel for images of small size (less or equal than maximum number of threads per block).
__global__ void CropKernelSmall(uchar* srcBuffer, uchar* destBuffer, uint startPixelX, uint startPixelY, uint imageStride, uint croppedSize, uint numOfChannels)
{
	uint startPosX = startPixelX * numOfChannels;
	uint endPixelY = startPixelY + croppedSize;
	uint croppedImageStride = croppedSize * numOfChannels;

	for (uint row = startPixelY; row < endPixelY; row += gridDim.x)
	{
		uint posY = row + blockIdx.x;
		if (posY < endPixelY && threadIdx.x < croppedImageStride)
		{			
			destBuffer[(posY - startPixelY) * croppedImageStride + threadIdx.x] = srcBuffer[posY * imageStride + startPosX + threadIdx.x];
		}
	}
}

// Crop kernel for images of medium size (larger than maximum number of threads per block, but smaller or equal than maximum number of threads per SM).
__global__ void CropKernelMedium(uchar* srcBuffer, uchar* destBuffer, uint startPixelX, uint startPixelY, uint imageStride, uint croppedSize, uint numOfChannels)
{
	uint startPosX = startPixelX * numOfChannels;
	uint endPixelY = startPixelY + croppedSize;
	uint croppedImageStride = croppedSize * numOfChannels;

	for (uint row = startPixelY; row < endPixelY; ++row)
	{
		uint posX = blockIdx.x * blockDim.x + threadIdx.x;
		if (posX < croppedImageStride)
		{
			destBuffer[(row - startPixelY) * croppedImageStride + posX] = srcBuffer[row * imageStride + startPosX + posX];
		}
	}
}

// Crop kernel for images of large size (larger than maximum number of threads per SM).
__global__ void CropKernelLarge(uchar* srcBuffer, uchar* destBuffer, uint startPixelX, uint startPixelY, uint imageStride, uint croppedSize, uint numOfChannels)
{
	uint startPosX = startPixelX * numOfChannels;
	uint endPixelY = startPixelY + croppedSize;
	uint croppedImageStride = croppedSize * numOfChannels;

	for (uint row = startPixelY; row < endPixelY; ++row)
	{
		for (uint posX = blockIdx.x * blockDim.x + threadIdx.x; posX < croppedImageStride; posX += gridDim.x * blockDim.x)
		{
			destBuffer[(row - startPixelY) * croppedImageStride + posX] = srcBuffer[row * imageStride + startPosX + posX];
		}
	}
}

void JpegDataParser::CropOnInvertedDeviceBuffers(uint imageWidth, uint imageHeight, uint numOfChannels, uint croppedSize, uint startPixelX, uint startPixelY,
	cudaStream_t stream)
{
	ShipAssert(startPixelX + croppedSize <= imageWidth, "Can't crop jpeg image, bad crop width dimensions.");
	ShipAssert(startPixelY + croppedSize <= imageHeight, "Can't crop jpeg image, bad crop height dimensions.");

	// Inverting source and destination buffers to save time of calling memcpy, source buffer will be used to store cropped image.
	AllocSrcDeviceMemoryIfNeeded((size_t)croppedSize * croppedSize * numOfChannels * sizeof(uchar));

	uint croppedImageStride = croppedSize * numOfChannels;
	if (croppedImageStride <= Config::MAX_NUM_THREADS)
	{
		uint numThreads = RoundUp(croppedImageStride, Config::WARP_SIZE);
		uint numBlocks = min(Config::MAX_NUM_FULL_BLOCKS * Config::MAX_NUM_THREADS / croppedImageStride, (uint)Config::MAX_NUM_BLOCKS);
		LAUNCH_KERNEL_ASYNC(CropKernelSmall, numBlocks, numThreads, stream)(m_deviceOpDestBuffer, m_deviceOpSrcBuffer, startPixelX, startPixelY,
			imageWidth * numOfChannels, croppedSize, numOfChannels);
	}
	else if (croppedImageStride <= Config::MAX_NUM_FULL_BLOCKS * Config::MAX_NUM_THREADS)
	{
		uint numThreads = RoundUp(DivideUp(croppedImageStride, Config::MAX_NUM_FULL_BLOCKS), Config::WARP_SIZE);
		uint numBlocks = Config::MAX_NUM_FULL_BLOCKS;
		LAUNCH_KERNEL_ASYNC(CropKernelMedium, numBlocks, numThreads, stream)(m_deviceOpDestBuffer, m_deviceOpSrcBuffer, startPixelX, startPixelY,
			imageWidth * numOfChannels, croppedSize, numOfChannels);
	}
	else
	{
		uint numThreads = Config::MAX_NUM_THREADS;
		uint numBlocks = Config::MAX_NUM_FULL_BLOCKS;
		LAUNCH_KERNEL_ASYNC(CropKernelLarge, numBlocks, numThreads, stream)(m_deviceOpDestBuffer, m_deviceOpSrcBuffer, startPixelX, startPixelY,
			imageWidth * numOfChannels, croppedSize, numOfChannels);
	}
	CudaAssert(cudaGetLastError());
}

void JpegDataParser::CropOnInvertedDeviceBuffers(uint imageWidth, uint imageHeight, uint numOfChannels, uint croppedSize, uint edgePadding, CropMode cropMode,
	cudaStream_t stream)
{
	uint startPixelX, startPixelY;
	switch (cropMode)
	{
		case CropMode::CropLeft:
			startPixelX = edgePadding;
			startPixelY = (imageHeight - croppedSize) / 2;
			break;
		case CropMode::CropTop:
			startPixelX = (imageWidth - croppedSize) / 2;
			startPixelY = edgePadding;
			break;
		case CropMode::CropRight:
			startPixelX = imageWidth - croppedSize - edgePadding;
			startPixelY = (imageHeight - croppedSize) / 2;
			break;
		case CropMode::CropBottom:
			startPixelX = (imageWidth - croppedSize) / 2;
			startPixelY = imageHeight - croppedSize - edgePadding;
			break;
		case CropMode::CropCenter:
			startPixelX = (imageWidth - croppedSize) / 2;
			startPixelY = (imageHeight - croppedSize) / 2;
			break;
		default:
			ShipAssert(false, "Unknown crop mode encountered!");
	}

	CropOnInvertedDeviceBuffers(imageWidth, imageHeight, numOfChannels, croppedSize, startPixelX, startPixelY, stream);
}

ImageData* JpegDataParser::ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
	CropMode cropMode)
{
	return ResizeImageWithCropCu(image, desiredWidth, desiredHeight, resizeMode, cropMode, 0);
}

ImageData* JpegDataParser::ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
	CropMode cropMode, cudaStream_t stream)
{
	int destImageWidth, destImageHeight;
	ResizeOnDeviceBuffers(image, desiredWidth, desiredHeight, resizeMode, stream, destImageWidth, destImageHeight);

	// Cropping on device.
	uint croppedImageWidth = destImageWidth;
	uint croppedImageHeight = destImageHeight;
	uchar* destinationBuffer = m_deviceOpDestBuffer;
	bool horizontalCropMode = cropMode == CropMode::CropLeft || cropMode == CropMode::CropCenter || cropMode == CropMode::CropRight;
	bool verticalCropMode = cropMode == CropMode::CropTop || cropMode == CropMode::CropCenter || cropMode == CropMode::CropBottom;
	if ((horizontalCropMode && destImageWidth > destImageHeight) || (verticalCropMode && destImageHeight > destImageWidth))
	{
		croppedImageWidth = croppedImageHeight = min(destImageWidth, destImageHeight);
		CropOnInvertedDeviceBuffers(destImageWidth, destImageHeight, image.GetNumOfChannels(), croppedImageWidth, 0, cropMode, stream);
		
		// This has to be done after cropping, since device source operations buffer could be reallocated during cropping if it is too small.
		destinationBuffer = m_deviceOpSrcBuffer;
	}

	// Creating output image.
	ImageData* resizedImage = new ImageData(croppedImageWidth, croppedImageHeight, image.GetNumOfChannels());
	CudaAssert(cudaMemcpyAsync(resizedImage->GetRowMajorPixels(), destinationBuffer, resizedImage->GetBufferSize(), cudaMemcpyDeviceToHost, stream));
	m_lastImageDeviceBuffer = destinationBuffer;
	CudaAssert(cudaStreamSynchronize(stream));

	return resizedImage;
}