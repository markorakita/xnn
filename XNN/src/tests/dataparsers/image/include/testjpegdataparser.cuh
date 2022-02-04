// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for jpeg data parser.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>

#include "../../../include/abstracttester.cuh"
#include "../../../../utils/include/deftypes.cuh"

using namespace std;

class TestJpegDataParser : public AbstractTester
{
private:
	// Input folder for test images.
	string m_inputFolder;

	// Output folder for test operations.
	string m_outputFolder;

	// Checks if input and output folders are specified.
	bool CheckInputOutputFolders();

	// Tests.
	bool TestResizeImageCu();
	bool TestCropImage();
	bool TestResizeImageWithCropCu();

public:
	// Constructor.
	TestJpegDataParser(string inputFolder, string outputFolder);

	// Runs specific test.
	virtual bool RunTest(string testName);

	// Runs all tests.
	virtual bool RunAllTests();
};