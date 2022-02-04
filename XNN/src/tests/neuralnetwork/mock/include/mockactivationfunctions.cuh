// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network activation functions, used in tests.
// Created: 03/20/2021.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../utils/include/deftypes.cuh"

using namespace std;

enum class ActivationType;

// Applies activation to preactivations, brute force version.
void ApplyActivationBF(ActivationType activationType, float activationAlpha, float* preactivations, uint numPreactivations, float* activations);

// Calculates gradients of activations to preactivations, brute force version.
void CalculatePreactivationGradientsBF(ActivationType activationType, float activationAlpha, float* activationGradients, float* activations,
	uint numActivations, float* preactivationGradients);