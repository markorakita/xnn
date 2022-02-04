// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>

#include "deftypes.cuh"

using namespace std;

// Returns lower case version of string.
string ConvertToLowercase(string inputString);

// Returns extension of file.
string GetExtension(string fileName);

// Returns file name from file path.
string GetFileName(string filePath);

// Returns file name from file path, without extension.
string GetFileNameWithoutExtension(string filePath);

// Parses argument from arguments list with specified signature to out variable.
bool ParseArgument(int argc, char *argv[], string signature, string& out, bool convertToLowercase = false);
bool ParseArgument(int argc, char *argv[], string signature, uint& out);
bool ParseArgument(int argc, char* argv[], string signature, int& out);

// Sets out as true if there is argument in arguments list with specified signature.
void ParseArgument(int argc, char *argv[], string signature, bool& out);

// Divides two numbers so that quotient times divisor is larger or equal than dividend.
inline uint DivideUp(uint dividend, uint divisor)
{
	return (dividend + divisor - 1) / divisor;
}

// Rounds up the number so it is divisible by base.
inline uint RoundUp(uint number, uint base)
{
	return DivideUp(number, base) * base;
}

// Gets current time stamp.
string GetCurrentTimeStamp();