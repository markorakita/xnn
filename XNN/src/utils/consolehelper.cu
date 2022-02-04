// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Console helper.
// Created: 03/06/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/consolehelper.cuh"

#define NOMINMAX
#include <windows.h>

bool ConsoleHelper::s_savedInitialConsoleState = false;
unsigned short ConsoleHelper::s_initialConsoleState = 0;

WORD GetForegroundColorValue(ConsoleColor foregroundColor)
{
	switch (foregroundColor)
	{
		case ConsoleColor::DARKBLUE:
			return FOREGROUND_BLUE;
		case ConsoleColor::DARKGREEN:
			return FOREGROUND_GREEN;
		case ConsoleColor::DARKCYAN:
			return FOREGROUND_GREEN | FOREGROUND_BLUE;
		case ConsoleColor::DARKRED:
			return FOREGROUND_RED;
		case ConsoleColor::DARKMAGENTA:
			return FOREGROUND_RED | FOREGROUND_BLUE;
		case ConsoleColor::DARKYELLOW:
			return FOREGROUND_RED | FOREGROUND_GREEN;
		case ConsoleColor::DARKGRAY:
			return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
		case ConsoleColor::GRAY:
			return FOREGROUND_INTENSITY;
		case ConsoleColor::BLUE:
			return FOREGROUND_INTENSITY | FOREGROUND_BLUE;
		case ConsoleColor::GREEN:
			return FOREGROUND_INTENSITY | FOREGROUND_GREEN;
		case ConsoleColor::CYAN:
			return FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE;
		case ConsoleColor::RED:
			return FOREGROUND_INTENSITY | FOREGROUND_RED;
		case ConsoleColor::MAGENTA:
			return FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_BLUE;
		case ConsoleColor::YELLOW:
			return FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN;
		case ConsoleColor::WHITE:
			return FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
		default:
			return 0;
	}
}

WORD GetBackgroundColorValue(ConsoleColor backgroundColor)
{
	switch (backgroundColor)
	{
		case ConsoleColor::DARKBLUE:
			return BACKGROUND_BLUE;
		case ConsoleColor::DARKGREEN:
			return BACKGROUND_GREEN;
		case ConsoleColor::DARKCYAN:
			return BACKGROUND_GREEN | BACKGROUND_BLUE;
		case ConsoleColor::DARKRED:
			return BACKGROUND_RED;
		case ConsoleColor::DARKMAGENTA:
			return BACKGROUND_RED | BACKGROUND_BLUE;
		case ConsoleColor::DARKYELLOW:
			return BACKGROUND_RED | BACKGROUND_GREEN;
		case ConsoleColor::DARKGRAY:
			return BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE;
		case ConsoleColor::GRAY:
			return BACKGROUND_INTENSITY;
		case ConsoleColor::BLUE:
			return BACKGROUND_INTENSITY | BACKGROUND_BLUE;
		case ConsoleColor::GREEN:
			return BACKGROUND_INTENSITY | BACKGROUND_GREEN;
		case ConsoleColor::CYAN:
			return BACKGROUND_INTENSITY | BACKGROUND_GREEN | BACKGROUND_BLUE;
		case ConsoleColor::RED:
			return BACKGROUND_INTENSITY | BACKGROUND_RED;
		case ConsoleColor::MAGENTA:
			return BACKGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_BLUE;
		case ConsoleColor::YELLOW:
			return BACKGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN;
		case ConsoleColor::WHITE:
			return BACKGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE;
		default:
			return 0;
	}
}

void ConsoleHelper::SaveInitialConsoleState(void* consoleStdOutHandle)
{
	CONSOLE_SCREEN_BUFFER_INFO consoleScreenBufferInfo;
	GetConsoleScreenBufferInfo(consoleStdOutHandle, &consoleScreenBufferInfo);

	s_initialConsoleState = consoleScreenBufferInfo.wAttributes;
	s_savedInitialConsoleState = true;
}

void ConsoleHelper::SetConsoleForeground(ConsoleColor foregroundColor)
{
	HANDLE consoleStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

	if (!s_savedInitialConsoleState)
	{
		SaveInitialConsoleState(consoleStdOutHandle);
	}

	SetConsoleTextAttribute(consoleStdOutHandle, GetForegroundColorValue(foregroundColor));
}

void ConsoleHelper::SetConsoleBackground(ConsoleColor backgroundColor)
{
	HANDLE consoleStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

	if (!s_savedInitialConsoleState)
	{
		SaveInitialConsoleState(consoleStdOutHandle);
	}

	SetConsoleTextAttribute(consoleStdOutHandle, GetBackgroundColorValue(backgroundColor));
}

void ConsoleHelper::RevertConsoleColors()
{
	// This means console is still intact.
	if (!s_savedInitialConsoleState)
	{
		return;
	}

	HANDLE consoleStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(consoleStdOutHandle, s_initialConsoleState);
}