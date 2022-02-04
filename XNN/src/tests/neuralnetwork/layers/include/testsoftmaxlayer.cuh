// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestSoftMaxLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleForwardProp(uint inputDataSize, uint inputDataCount);
	bool TestSingleBackwardProp(uint inputDataSize, uint inputDataCount);

	// Tests.
	bool TestDoForwardProp();
	bool TestDoBackwardProp();

public:
	// Input activations mean and standard deviation to be used in tests.
	static const float c_inputActivationsMean;
	static const float c_inputActivationsStDev;

	// Constructor.
	TestSoftMaxLayer();
};