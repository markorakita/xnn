// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Factory for creating data parsers.
// Created: 11/29/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>

using namespace std;

// Data parser types.
enum class DataParserType
{
	JPEG,
	Unknown
};

class DataParser;

class DataParserFactory
{
private:
	// Map of created data parsers, mapped on data extension.
	map<DataParserType, DataParser*> m_dataParsers;

	// Gets data parser type for data extension.
	DataParserType GetDataParserType(string dataExtension);

	// Creates appropriate data parser depending on data parser type.
	// Returns: true if creating data parser is successful, false otherwise.
	bool CreateDataParser(DataParserType dataParserType, DataParser** outDataParser);

public:
	// Destructor.
	~DataParserFactory();

	// Returns appropriate data parser, depending on data extension.
	// (Data extension should be in lower case!)
	DataParser* GetDataParser(string dataExtension);
};
