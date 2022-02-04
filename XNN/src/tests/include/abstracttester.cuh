// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract tester interface.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <functional>
#include <map>
#include <string>

using namespace std;

class AbstractTester
{
private:
	// Prints test result.
	void PrintTestResult(bool testResult, string testName);

protected:
	// Mapping from test name to test function.
	map<string, function<bool()>> m_tests;

public:
	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual bool RunTest(string testName);

	// Runs all tests.
	virtual bool RunAllTests();
};