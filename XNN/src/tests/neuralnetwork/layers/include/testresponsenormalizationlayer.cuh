// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for response normalization layer.
// Created: 02/11/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestResponseNormalizationLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth, float bias,
		float alphaCoeff, float betaCoeff);
	bool TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth, float bias,
		float alphaCoeff, float betaCoeff);

	// Tests.
	bool TestDoForwardProp();
	bool TestDoBackwardProp();

public:
	// Constructor.
	TestResponseNormalizationLayer();
};