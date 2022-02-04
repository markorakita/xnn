// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Driver for all tests.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>

using namespace std;

class AbstractTester;

class TestsDriver
{
private:
	// Input folder for test resources.
	string m_inputFolder;

	// Output folder for test operations.
	string m_outputFolder;

	// Component to test.
	string m_componentToRun;

	// Specific test name to test.
	string m_testToRun;

	// Map of all testers by component name.
	map<string, AbstractTester*> m_testers;

	// Registers testers.
	void RegisterTesters();

public:
	static const string c_outputFolderSignature;
	static const string c_componentToRunSignature;
	static const string c_testToRunSignature;

	// Destructor,
	~TestsDriver();

	// Parses arguments for testing.
	bool ParseArguments(int argc, char *argv[]);

	// Runs tests.
	void RunTests();
};