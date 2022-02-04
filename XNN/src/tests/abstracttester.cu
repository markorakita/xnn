// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract tester interface.
// Created: 10/08/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/abstracttester.cuh"

#include <iostream>

#include "../utils/include/asserts.cuh"
#include "../utils/include/consolehelper.cuh"

void AbstractTester::PrintTestResult(bool testResult, string testName)
{
    if (testResult)
    {
        ConsoleHelper::SetConsoleForeground(ConsoleColor::GREEN);
        cout << "Test " << testName << " passed!" << endl;
        ConsoleHelper::RevertConsoleColors();
    }
    else
    {
        ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
        cout << "Test " << testName << " failed!" << endl;
        ConsoleHelper::RevertConsoleColors();
    }
}

bool AbstractTester::HasTest(string testName)
{
    auto test = m_tests.find(testName);
    return test != m_tests.end();
}

bool AbstractTester::RunTest(string testName)
{
    auto test = m_tests.find(testName);
    ShipAssert(test != m_tests.end(), "Test " + testName + " not found!");

    bool testResult = (test->second)();
    PrintTestResult(testResult, test->first);

    return testResult;
}

bool AbstractTester::RunAllTests()
{
    bool allTestsPassed = true;
    for (auto test = m_tests.begin(); test != m_tests.end(); ++test)
    {
        cout << "Running test " << test->first << endl;
        bool testResult = (test->second)();
        PrintTestResult(testResult, test->first);

        allTestsPassed = allTestsPassed && testResult;
    }

    return allTestsPassed;
}