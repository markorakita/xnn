// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for output layer.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestOutputLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleCrossEntropyForwardProp(uint inputDataSize, uint inputDataCount);
	bool TestSingleLogisticRegressionForwardProp(uint inputDataCount);

	// Tests.
	bool TestDoForwardProp();

public:
	// Constructor.
	TestOutputLayer();
};