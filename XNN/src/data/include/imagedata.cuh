// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Image data class.
// Created: 11/29/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../utils/include/deftypes.cuh"

using namespace std;

class ImageData
{
private:
	// Image width.
	uint m_width;

	// Image height.
	uint m_height;

	// Number of channels.
	uint m_numberOfChannels;

	// Image stride.
	uint m_stride;

	// Pixels in row major order, in RGB mode.
	uchar* m_rowMajorPixels;

	// Image buffer size.
	size_t m_bufferSize;

public:
	// Constructor.
	ImageData(uint width, uint height, uint numberOfChannels);

	// Destructor.
	~ImageData();

	// Returns width of image.
	uint GetWidth() const { return m_width; }

	// Returns height of image.
	uint GetHeight() const { return m_height; }

	// Returns number of channels in image.
	uint GetNumOfChannels() const { return m_numberOfChannels; }

	// Returns image stride.
	uint GetStride() const { return m_stride; }

	// Returns image buffer size.
	size_t GetBufferSize() const { return m_bufferSize; }

	// Returns pointer to pixels of image in row major order, in RGB mode.
	uchar* GetRowMajorPixels() const { return m_rowMajorPixels; }
};