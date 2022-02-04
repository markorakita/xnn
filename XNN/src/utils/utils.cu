// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/utils.cuh"

#include <algorithm>

#define NOMINMAX
#include <windows.h>

string ConvertToLowercase(string inputString)
{
	string outputString = inputString;
	if (inputString != "")
	{
		transform(outputString.begin(), outputString.end(), outputString.begin(), tolower);
	}

	return outputString;
}

string GetExtension(string fileName)
{
	size_t dotPosition = fileName.find_last_of('.');
	if (dotPosition != string::npos && dotPosition < fileName.size() - 1)
	{
		return ConvertToLowercase(fileName.substr(dotPosition + 1));
	}

	return "";
}

string GetFileName(string filePath)
{
	size_t slashPosition = filePath.find_last_of('\\');
	if (slashPosition == string::npos)
	{
		slashPosition = filePath.find_last_of('/');
	}
	if (slashPosition != string::npos && slashPosition < filePath.size() - 1)
	{
		return filePath.substr(slashPosition + 1);
	}

	return filePath;
}

string GetFileNameWithoutExtension(string filePath)
{
	string fileName = GetFileName(filePath);
	size_t dotPosition = fileName.find_last_of('.');
	if (dotPosition != string::npos)
	{
		return fileName.substr(0, dotPosition);
	}

	return fileName;
}

string ParseArgumentValue(int argc, char* argv[], string signature)
{
	for (int i = 0; i < argc - 1; ++i)
	{
		if (ConvertToLowercase(string(argv[i])) == signature)
		{
			return string(argv[i + 1]);
		}
	}

	return "";
}

bool ParseArgument(int argc, char *argv[], string signature, string& out, bool convertToLowercase /*= false*/)
{
	string argumentValue = ParseArgumentValue(argc, argv, signature);
	if (!argumentValue.empty())
	{
		if (convertToLowercase)
		{
			out = ConvertToLowercase(argumentValue);
		}
		else
		{
			out = argumentValue;
		}
		return true;
	}

	return false;
}

bool ParseArgument(int argc, char *argv[], string signature, uint& out)
{
	string argumentValue = ParseArgumentValue(argc, argv, signature);
	if (!argumentValue.empty())
	{
		out = stoul(argumentValue);
		return true;
	}

	return false;
}

bool ParseArgument(int argc, char* argv[], string signature, int& out)
{
	string argumentValue = ParseArgumentValue(argc, argv, signature);
	if (!argumentValue.empty())
	{
		out = stoi(argumentValue);
		return true;
	}

	return false;
}

void ParseArgument(int argc, char *argv[], string signature, bool& out)
{
	for (int i = 0; i < argc; ++i)
	{
		if (ConvertToLowercase(string(argv[i])) == signature)
		{
			out = true;
			return;
		}
	}

	out = false;
}

string GetCurrentTimeStamp()
{
	SYSTEMTIME time;
	GetLocalTime(&time);

	string month = to_string(time.wMonth);
	if (month.length() == 1)
	{
		month = "0" + month;
	}

	string day = to_string(time.wDay);
	if (day.length() == 1)
	{
		day = "0" + day;
	}

	string hour = to_string(time.wHour);
	if (hour.length() == 1)
	{
		hour = "0" + hour;
	}

	string minute = to_string(time.wMinute);
	if (minute.length() == 1)
	{
		minute = "0" + minute;
	}

	return month + "/" + day + "/" + to_string(time.wYear) + " " + hour + ":" + minute;
}