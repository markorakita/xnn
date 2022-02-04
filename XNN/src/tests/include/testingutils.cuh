// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions for testing.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

using namespace std;

// Compares two buffers with results and returns whether we got correct result, or if we didn't returns first difference etc.
void CompareBuffers(const float* regularBuffer, const float* mockBuffer, size_t buffersLength, float maxDiff, float maxDiffPercentage, float maxDiffPercentageThreshold,
	bool& correctResult, size_t& numDifferences, float& firstDifference, float& firstDifferentMock, float& firstDifferentReg, bool& foundDifferentFromZeroMock,
	bool& foundDifferentFromZeroReg);