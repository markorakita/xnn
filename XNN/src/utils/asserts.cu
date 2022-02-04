// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/asserts.cuh"

#include <iostream>

#include "include/consolehelper.cuh"
#include "include/utils.cuh"

void _Assert(bool condition, string message, const char* file, const char* function, int line, bool shouldExit)
{
	if (!condition)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
			cout << endl << "Fatal error encountered: " << message << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
			ConsoleHelper::RevertConsoleColors();
		}
		if (shouldExit)
		{
			exit(EXIT_FAILURE);
		}
	}
}

void EmitWarning(string message)
{
	lock_guard<mutex> lock(s_consoleMutex);
	ConsoleHelper::SetConsoleForeground(ConsoleColor::YELLOW);
	cout << message << endl;
	ConsoleHelper::RevertConsoleColors();
}