// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for neural network activation functions.
// Created: 03/22/2021.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../include/abstracttester.cuh"
#include "../../../utils/include/deftypes.cuh"

using namespace std;

enum class ActivationType;
class NeuralNet;

class TestActivationFunctions : public AbstractTester
{
private:
	static const float c_activationAlpha;
	static const uint c_numActivations;

	// Helper functions.
	void ApplyTestActivations(ActivationType activationType, float* hostActivationsBufferBF, float* deviceActivationsBuffer,
		NeuralNet* neuralNet);
	bool TestApplyActivation(ActivationType activationType, float maxComparisonDiff);
	bool TestCalculateActivationGradient(ActivationType activationType, float maxComparisonDiff);

	// Tests.
	bool TestApplyReLUActivation();
	bool TestApplyELUActivation();
	bool TestApplyLeakyReLUActivation();
	bool TestApplySigmoidActivation();
	bool TestApplyTanhActivation();
	bool TestCalculateReLUActivationGradient();
	bool TestCalculateELUActivationGradient();
	bool TestCalculateLeakyReLUActivationGradient();
	bool TestCalculateSigmoidActivationGradient();
	bool TestCalculateTanhActivationGradient();

public:
	// Constructor.
	TestActivationFunctions();
};