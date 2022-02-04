// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for abstract layer.
// Created: 12/07/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../include/abstracttester.cuh"

using namespace std;

class TestLayer : public AbstractTester
{
private:
	// Tests.
	bool TestInitializeBufferFromUniformDistribution();
	bool TestInitializeBufferFromNormalDistribution();
	bool TestInitializeBufferToConstant();

public:
	// Constructor.
	TestLayer();
};