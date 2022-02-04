// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestDropoutLayer : public AbstractTester
{
private:
	// Helper functions.
	bool TestSingleForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability);
	bool TestSingleBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability);

	// Tests.
	bool TestDoForwardProp();
	bool TestDoBackwardProp();

public:
	// Constructor.
	TestDropoutLayer();
};