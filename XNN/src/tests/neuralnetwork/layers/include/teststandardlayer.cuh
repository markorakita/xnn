// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for standard layer.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestStandardLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons);
	bool TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons);

	// Tests.
	bool TestDoForwardProp();
	bool TestForwardPropSpeed();
	bool TestDoBackwardProp();

public:
	// Constructor.
	TestStandardLayer();
};