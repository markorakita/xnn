// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network activation functions.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../utils/include/deftypes.cuh"

using namespace std;

typedef struct CUstream_st* cudaStream_t;

// Activation types.
enum class ActivationType
{
	Linear,
	ReLU,
	ELU,
	LeakyReLU,
	Sigmoid,
	Tanh
};

// Applies activation to preactivations.
void ApplyActivation(ActivationType activationType, float activationAlpha, float* preactivations, uint numPreactivations, float* activations,
	cudaStream_t deviceCalculationStream);

// Calculates gradients of activations to preactivations.
void CalculatePreactivationGradients(ActivationType activationType, float activationAlpha, float* activationGradients, float* activations,
	uint numActivations, float* preactivationGradients, cudaStream_t deviceCalculationStream);