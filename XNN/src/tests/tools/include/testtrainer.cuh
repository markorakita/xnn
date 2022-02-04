// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for the networks trainer.
// Created: 11/27/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../include/abstracttester.cuh"

class Trainer;
class MockInputLayer;
class OutputLayer;

class TestTrainer : public AbstractTester
{
private:
	bool CheckStandardSingleVsMultiGpuPropagationForward(Trainer* singleGpuTrainer, Trainer* multiGpuTrainer, MockInputLayer* mockInputLayer,
		OutputLayer* singleGpuOutputLayer, OutputLayer* multiGpuOutputLayer);

	// Tests.
	bool TestStandardSingleVsMultiGpuTraining();

public:
	// Constructor.
	TestTrainer();
};