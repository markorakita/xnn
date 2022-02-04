// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network activation functions, used in tests.
// Created: 03/20/2021.
// ----------------------------------------------------------------------------------------------------

#include "include/mockactivationfunctions.cuh"

#include "../../../neuralnetwork/include/activationfunctions.cuh"
#include "../../../utils/include/asserts.cuh"

void ApplyActivationBF(ActivationType activationType, float activationAlpha, float* preactivations, uint numPreactivations, float* activations)
{
	for (size_t i = 0; i < numPreactivations; ++i)
	{
		if (activationType == ActivationType::Linear)
		{
			activations[i] = preactivations[i];
		}
		else if (activationType == ActivationType::ReLU)
		{
			activations[i] = max(preactivations[i], 0.f);
		}
		else if (activationType == ActivationType::ELU)
		{
			activations[i] = preactivations[i] >= 0.f ? preactivations[i] : activationAlpha * ((float)exp(preactivations[i]) - 1.f);
		}
		else if (activationType == ActivationType::LeakyReLU)
		{
			activations[i] = preactivations[i] >= 0.f ? preactivations[i] : activationAlpha * preactivations[i];
		}
		else if (activationType == ActivationType::Sigmoid)
		{
			activations[i] = preactivations[i] >= 0.f ? 1.f / (1.f + (float)exp(-preactivations[i])) :
				(1.f - 1.f / (1.f + (float)exp(preactivations[i])));
		}
		else if (activationType == ActivationType::Tanh)
		{
			activations[i] = preactivations[i] >= 0.f ? (2.f / (1.f + (float)exp(-2.f * preactivations[i])) - 1.f) :
				(1.f - 2.f / (1.f + (float)exp(2.f * preactivations[i])));
		}
		else
		{
			ShipAssert(false, "Unknown activation type!");
		}
	}
}

void CalculatePreactivationGradientsBF(ActivationType activationType, float activationAlpha, float* activationGradients, float* activations,
	uint numActivations, float* preactivationGradients)
{
	for (size_t i = 0; i < numActivations; ++i)
	{
		if (activationType == ActivationType::Linear)
		{
			preactivationGradients[i] = activationGradients[i];
		}
		else if (activationType == ActivationType::ReLU)
		{
			preactivationGradients[i] = activations[i] > 0.f ? activationGradients[i] : 0.f;
		}
		else if (activationType == ActivationType::ELU)
		{
			preactivationGradients[i] = activations[i] > 0.f ? activationGradients[i] : activationGradients[i] * (activations[i] + activationAlpha);
		}
		else if (activationType == ActivationType::LeakyReLU)
		{
			preactivationGradients[i] = activations[i] > 0.f ? activationGradients[i] : activationGradients[i] * activationAlpha;
		}
		else if (activationType == ActivationType::Sigmoid)
		{
			preactivationGradients[i] = activationGradients[i] * activations[i] * (1.f - activations[i]);
		}
		else if (activationType == ActivationType::Tanh)
		{
			preactivationGradients[i] = activationGradients[i] * (1.f - activations[i] * activations[i]);
		}
		else
		{
			ShipAssert(false, "Unknown activation type!");
		}
	}
}