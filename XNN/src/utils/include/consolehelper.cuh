// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Console helper.
// Created: 03/06/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

enum class ConsoleColor
{
	BLACK,
	DARKBLUE,
	DARKGREEN,
	DARKCYAN,
	DARKRED,
	DARKMAGENTA,
	DARKYELLOW,
	DARKGRAY,
	GRAY,
	BLUE,
	GREEN,
	CYAN,
	RED,
	MAGENTA,
	YELLOW,
	WHITE,
};

class ConsoleHelper
{
private:
	// True if we saved initial console state.
	static bool s_savedInitialConsoleState;

	// Initial console state.
	static unsigned short s_initialConsoleState;

	// Saves initial console state.
	static void SaveInitialConsoleState(void* consoleStdOutHandle);

public:
	// Sets console foreground.
	static void SetConsoleForeground(ConsoleColor foregroundColor);

	// Sets console background.
	static void SetConsoleBackground(ConsoleColor backgroundColor);

	// Reverts console foreground and background back to default ones.
	static void RevertConsoleColors();
};