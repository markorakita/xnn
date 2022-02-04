// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for configuration parser.
// Created: 10/09/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../include/abstracttester.cuh"

using namespace std;

class TestConfigurationParser : public AbstractTester
{
private:
	// Tests.
	bool TestTrimLine();
	bool TestGetParameterValueStrFromLine();
	bool TestParseParameterUint();
	bool TestParseParameterFloat();
	bool TestParseParameterBool();
	bool TestParseParameterString();

public:
	// Constructor.
	TestConfigurationParser();
};