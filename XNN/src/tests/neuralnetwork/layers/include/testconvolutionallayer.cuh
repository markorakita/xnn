// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for convolutional layer.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestConvolutionalLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
		uint filterHeight, int paddingX, int paddingY, uint stride);
	bool TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
		uint filterHeight, int paddingX, int paddingY, uint stride);

	// Prints out computation info for debug purposes.
	void PrintComputationInfo(size_t activationDifferentPixelIndex, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount,
		uint numFilters, uint filterWidth, uint filterHeight, int paddingX, int paddingY, uint stride, float* inputDataBuffer, float* filtersBuffer,
		float differentActivationPixelMock, float differentActivationPixelRegular);

	// Tests.
	bool TestDoForwardProp();
	bool TestDoBackwardProp();

public:
	// Constructor.
	TestConvolutionalLayer();
};