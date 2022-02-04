// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for max pool layer.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestMaxPoolLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
		uint unitHeight, int paddingLeft, int paddingTop, uint unitStride);
	bool TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
		uint unitHeight, int paddingLeft, int paddingTop, uint unitStride);

	// Tests.
	bool TestDoForwardProp();
	bool TestDoBackwardProp();

public:
	// Constructor.
	TestMaxPoolLayer();
};