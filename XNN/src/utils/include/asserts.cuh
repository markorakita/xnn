// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Assert functions.
// Created: 09/22/2020.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <mutex>
#include <string>

// Macro to check ship asserts.
// Asserts in both debug and ship if condition is not met.
#define ShipAssert(condition, message) _Assert(condition, message, __FILE__, __FUNCTION__, __LINE__, true)

// Macro to check debug asserts.
// Asserts in debug only if condition is not met.
#if !defined(NDEBUG)
#define DebugAssert(condition, message) \
		do { _Assert(condition, message, __FILE__, __FUNCTION__, __LINE__, false); } while (0)
#else
#define DebugAssert(condition, message) \
		do { (void)sizeof(condition); } while(0)
#endif

using namespace std;

// Mutex for console output.
static mutex s_consoleMutex;

// Emits warning to console.
void EmitWarning(string message);

// Asserts in both debug and ship if condition is not met.
// Should never be called directly, use macro!
void _Assert(bool condition, string message, const char* file, const char* function, int line, bool shouldExit);